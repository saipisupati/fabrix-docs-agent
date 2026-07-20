"""
live_docs.py — optional fetch of public docs.fabrix.ai pages at answer time.

Used for platform install / VM / prerequisites asks so the agent can ground on the
same pages ChatGPT opens (e.g. /installation_guides/), without scraping the whole
site on every question.

Fail-open: network/SSL errors return [] and the local KB path continues.
Disable with LIVE_DOCS_FETCH=0.
"""

from __future__ import annotations

import html as html_lib
import logging
import os
import re
import ssl
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from doc_urls import PUBLIC_DOCS_BASE

logger = logging.getLogger(__name__)

LIVE_DOCS_FETCH = os.environ.get("LIVE_DOCS_FETCH", "1").strip().lower() not in (
    "0", "false", "no", "off",
)
LIVE_DOCS_TIMEOUT = float(os.environ.get("LIVE_DOCS_TIMEOUT", "12"))

# Install / VM facet — same index ChatGPT uses
INSTALL_LIVE_URLS: tuple[str, ...] = (
    f"{PUBLIC_DOCS_BASE.rstrip('/')}/installation_guides/",
    f"{PUBLIC_DOCS_BASE.rstrip('/')}/installation_guides/deployment/",
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "footer", "noscript"):
            self._skip += 1
        if tag in ("p", "br", "li", "tr", "h1", "h2", "h3", "h4", "div"):
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer", "noscript") and self._skip:
            self._skip -= 1
        if tag in ("p", "li", "tr", "h1", "h2", "h3", "h4"):
            self._chunks.append("\n")

    def handle_data(self, data):
        if self._skip:
            return
        text = data.strip()
        if text:
            self._chunks.append(text)

    def text(self) -> str:
        raw = " ".join(self._chunks)
        raw = html_lib.unescape(raw)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def html_to_text(page_html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(page_html or "")
        parser.close()
    except Exception:
        # Extremely broken HTML — crude strip
        return re.sub(r"<[^>]+>", " ", page_html or "")
    return parser.text()


def fetch_url_text(url: str, timeout: float | None = None) -> str | None:
    """GET a public docs URL and return visible text, or None on failure."""
    timeout = LIVE_DOCS_TIMEOUT if timeout is None else timeout
    try:
        req = Request(
            url,
            headers={"User-Agent": "fabrix-docs-agent-live/1.0"},
            method="GET",
        )
        with urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            raw = resp.read()
            charset = "utf-8"
            ctype = resp.headers.get_content_charset()
            if ctype:
                charset = ctype
            page = raw.decode(charset, errors="replace")
        text = html_to_text(page)
        if len(text) < 80:
            logger.info("live_docs: thin page text from %s (%s chars)", url, len(text))
            return None
        # Cap so prompts stay bounded
        return text[:12000]
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as e:
        logger.info("live_docs: fetch failed %s (%s)", url, e)
        return None


def live_install_kb_entries(max_pages: int = 2) -> list[dict]:
    """
    Fetch installation_guides pages into synthetic KB-shaped entries.
    Empty list when disabled or all fetches fail.
    """
    if not LIVE_DOCS_FETCH:
        return []
    entries: list[dict] = []
    for i, url in enumerate(INSTALL_LIVE_URLS[:max_pages]):
        text = fetch_url_text(url)
        if not text:
            continue
        # Prefer the prerequisites / hardware slice when present
        low = text.lower()
        focus = text
        for marker in ("prerequisites", "software installation prerequisites", "cpu"):
            idx = low.find(marker)
            if idx >= 0:
                focus = text[max(0, idx - 80) : idx + 4000]
                break
        entries.append(
            {
                "id": f"live-install-{i}",
                "kind": "procedure",
                "title": "RDA Studio / platform installation prerequisites (live docs)",
                "text": focus,
                "source": "installation_guides/index.md"
                if i == 0
                else "installation_guides/deployment.md",
                "url": url,
                "score": 1.0,
                "example": "",
            }
        )
        logger.info("live_docs: loaded %s (%s chars)", url, len(focus))
    return entries
