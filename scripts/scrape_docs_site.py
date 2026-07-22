#!/usr/bin/env python3
"""
scrape_docs_site.py — full mirror of https://docs.fabrix.ai → local markdown.

Discovers pages from MkDocs search_index.json plus a BFS link crawl, extracts
each page's <article> body, converts HTML→Markdown, and writes under DOCS_ROOT
using the same path layout as the official MD export (Bots/foo.md, etc.).

Examples:
  python3 scripts/scrape_docs_site.py
  python3 scripts/scrape_docs_site.py --out /path/to/docs --workers 12
  python3 scripts/scrape_docs_site.py --dry-run

Env: DOCS_ROOT (default out), DOCS_SITE_ORIGIN (default https://docs.fabrix.ai)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.request import Request, urlopen

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from doc_urls import PUBLIC_DOCS_BASE  # noqa: E402

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

try:
    from markdownify import markdownify as html_to_md
except ImportError:  # pragma: no cover
    html_to_md = None  # type: ignore


SKIP_PREFIXES = (
    "assets/",
    "stylesheets/",
    "javascripts/",
    "search/",
    "img/",
    "images/",
    "fonts/",
    "css/",
    "js/",
    "data/datasets/",  # binary/demo dataset assets, not doc pages
)
SKIP_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".css",
    ".js",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".map",
    ".xml",
    ".json",
    ".gz",
    ".zip",
    ".pdf",
    ".csv",
    ".parquet",
)
ROOT_FILE_SLUGS = {
    "Datasets": "Datasets.md",
    "Formatting-Templates": "Formatting-Templates.md",
}


class _HrefCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        for k, v in attrs:
            if k == "href" and v:
                self.hrefs.append(v)


def _load_dotenv() -> None:
    env_path = os.path.join(ROOT, ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip("'").strip('"'))


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def fetch_bytes(url: str, timeout: float, ctx: ssl.SSLContext) -> tuple[bytes, str]:
    req = Request(
        url,
        headers={
            "User-Agent": "fabrix-docs-agent-scrape/1.0 (+local corpus mirror)",
            "Accept": "text/html,application/json,*/*",
        },
        method="GET",
    )
    with urlopen(req, timeout=timeout, context=ctx) as resp:
        raw = resp.read()
        ctype = (resp.headers.get_content_type() or "").lower()
        final = resp.geturl() or url
    return raw, final if ctype else final


def normalize_page_path(href: str, base: str) -> str | None:
    """Return site-relative page path without leading slash or hash, or None."""
    if not href or href.startswith(("mailto:", "javascript:", "data:", "tel:")):
        return None
    full = urljoin(base.rstrip("/") + "/", href)
    full, _frag = urldefrag(full)
    parsed = urlparse(full)
    origin = urlparse(base)
    if parsed.netloc and parsed.netloc != origin.netloc:
        return None
    path = parsed.path or "/"
    if path.startswith("/"):
        path = path[1:]
    low = path.lower()
    if any(low.startswith(p) for p in SKIP_PREFIXES):
        return None
    if any(low.endswith(s) for s in SKIP_SUFFIXES):
        return None
    if low.endswith(".html"):
        path = path[: -len(".html")]
    # collapse // and strip trailing slash for consistency (except home="")
    path = re.sub(r"/+", "/", path).strip("/")
    return path


def path_to_md_rel(page_path: str) -> str:
    """Map site path → markdown relative path under DOCS_ROOT."""
    if not page_path:
        return "index.md"
    parts = page_path.split("/")
    if len(parts) == 1 and parts[0] in ROOT_FILE_SLUGS:
        return ROOT_FILE_SLUGS[parts[0]]
    # section indexes: installation_guides → installation_guides/index.md
    # leaf pages: Bots/kafka-v2 → Bots/kafka-v2.md
    if len(parts) == 1:
        # Could be section root (Bots, Pipelines, …) → index.md inside
        return f"{parts[0]}/index.md"
    return f"{page_path}.md"


def discover_from_search_index(base: str, timeout: float, ctx: ssl.SSLContext) -> set[str]:
    url = base.rstrip("/") + "/search/search_index.json"
    raw, _ = fetch_bytes(url, timeout, ctx)
    data = json.loads(raw.decode("utf-8", errors="replace"))
    docs = data.get("docs") or []
    pages: set[str] = set()
    for d in docs:
        if not isinstance(d, dict):
            continue
        loc = (d.get("location") or "").split("#", 1)[0].strip()
        p = normalize_page_path(loc, base)
        if p is not None:
            pages.add(p)
    return pages


def extract_links(html: str, page_url: str, base: str) -> set[str]:
    pages: set[str] = set()
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "lxml" if _has_lxml() else "html.parser")
        for a in soup.find_all("a", href=True):
            p = normalize_page_path(a["href"], page_url)
            if p is not None:
                # normalize against site base
                p2 = normalize_page_path(p, base)
                if p2 is not None:
                    pages.add(p2)
        return pages
    parser = _HrefCollector()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        return pages
    for href in parser.hrefs:
        p = normalize_page_path(href, page_url)
        if p is not None:
            pages.add(p)
    return pages


def _has_lxml() -> bool:
    try:
        import lxml  # noqa: F401

        return True
    except ImportError:
        return False


def extract_article_html(html: str) -> tuple[str, str]:
    """Return (title, article_inner_html)."""
    title = ""
    if BeautifulSoup is None:
        m = re.search(r"<article[^>]*>([\s\S]*?)</article>", html, re.I)
        body = m.group(1) if m else html
        tm = re.search(r"<h1[^>]*>([\s\S]*?)</h1>", body, re.I)
        if tm:
            title = re.sub(r"<[^>]+>", "", tm.group(1)).strip()
        return title, body

    soup = BeautifulSoup(html, "lxml" if _has_lxml() else "html.parser")
    # Drop chrome that sometimes nests inside article
    for sel in ("script", "style", "noscript", "nav", "footer", ".md-source-file", ".headerlink"):
        for tag in soup.select(sel):
            tag.decompose()

    article = soup.select_one("article.md-content__inner") or soup.select_one("article")
    if article is None:
        article = soup.select_one(".md-content__inner") or soup.body or soup

    h1 = article.find("h1") if article else None
    if h1:
        title = h1.get_text(" ", strip=True)
    elif soup.title:
        title = soup.title.get_text(" ", strip=True)

    # Remove permalink glyphs
    for a in article.select("a.headerlink") if article else []:
        a.decompose()

    return title, str(article)


def article_to_markdown(title: str, article_html: str, source_url: str) -> str:
    if html_to_md is not None:
        md = html_to_md(
            article_html,
            heading_style="ATX",
            bullets="-",
            strip=["script", "style"],
        )
    else:
        # Minimal fallback
        text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", article_html)
        text = re.sub(r"(?i)<br\s*/?>", "\n", text)
        text = re.sub(r"(?i)</p>", "\n\n", text)
        text = re.sub(r"(?i)</h([1-6])>", "\n\n", text)
        text = re.sub(r"(?i)<li[^>]*>", "- ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        md = re.sub(r"[ \t]+\n", "\n", text)
        md = re.sub(r"\n{3,}", "\n\n", md)

    md = md.strip()
    # Ensure a top heading if missing
    if title and not re.match(r"^#\s+", md):
        md = f"# {title}\n\n{md}"
    header = (
        f"<!-- scraped from {source_url} -->\n"
        f"<!-- generated by scripts/scrape_docs_site.py -->\n\n"
    )
    return header + md + "\n"


def crawl_extra(
    seed_pages: set[str],
    base: str,
    timeout: float,
    ctx: ssl.SSLContext,
    max_extra_fetches: int = 800,
) -> set[str]:
    """
    BFS for pages missing from search_index.

    Starts from home + section indexes only (not every leaf), then expands
    newly discovered paths until the budget is hit.
    """
    known = set(seed_pages)
    section_seeds = {
        "",
        "Bots",
        "installation_guides",
        "beginners_guide",
        "Pipelines",
        "Datasource_Integrations",
        "ai_fabric",
        "Extensions",
        "reference_guides",
        "rda_releases",
        "Datasets",
        "Formatting-Templates",
    }
    for extra in section_seeds:
        known.add(extra)
    # Prefer crawling seeds first; only enqueue newly found paths
    queue = sorted(section_seeds, key=lambda p: (p.count("/"), p))
    seen_fetch: set[str] = set()
    fetches = 0
    while queue and fetches < max_extra_fetches:
        path = queue.pop(0)
        if path in seen_fetch:
            continue
        seen_fetch.add(path)
        url = base.rstrip("/") + ("/" if not path else f"/{path}/")
        try:
            raw, final = fetch_bytes(url, timeout, ctx)
            html = raw.decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError, OSError):
            continue
        fetches += 1
        found = extract_links(html, final, base)
        for p in found:
            if p not in known:
                known.add(p)
                queue.append(p)
    return known


def scrape_one(
    page_path: str,
    base: str,
    out_root: str,
    timeout: float,
    ctx: ssl.SSLContext,
    dry_run: bool,
) -> tuple[str, str, int]:
    """Returns (page_path, status, bytes_written)."""
    if not page_path:
        url = base.rstrip("/") + "/"
    else:
        url = base.rstrip("/") + f"/{page_path}/"

    try:
        raw, final = fetch_bytes(url, timeout, ctx)
    except HTTPError as e:
        return page_path, f"http_{e.code}", 0
    except (URLError, TimeoutError, OSError) as e:
        return page_path, f"err:{type(e).__name__}", 0

    html = raw.decode("utf-8", errors="replace")
    title, article_html = extract_article_html(html)
    md = article_to_markdown(title, article_html, final)
    rel = path_to_md_rel(page_path)
    dest = os.path.join(out_root, rel)
    if dry_run:
        return page_path, f"dry:{rel}", len(md.encode("utf-8"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(md)
    return page_path, f"ok:{rel}", len(md.encode("utf-8"))


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=os.environ.get("DOCS_ROOT", ""),
        help="Output DOCS_ROOT (default from .env DOCS_ROOT)",
    )
    parser.add_argument(
        "--base",
        default=os.environ.get("DOCS_SITE_ORIGIN", PUBLIC_DOCS_BASE),
        help="Docs site origin",
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-crawl",
        action="store_true",
        help="Only use search_index.json (skip BFS link expansion)",
    )
    parser.add_argument(
        "--max-crawl-fetches",
        type=int,
        default=900,
        help="Max pages to fetch during BFS discovery",
    )
    args = parser.parse_args()

    if not args.out:
        print("ERROR: --out or DOCS_ROOT required", file=sys.stderr)
        return 2
    if BeautifulSoup is None or html_to_md is None:
        print(
            "ERROR: need beautifulsoup4 + markdownify "
            "(pip install beautifulsoup4 markdownify lxml)",
            file=sys.stderr,
        )
        return 2

    ctx = _ssl_context()
    base = args.base.rstrip("/")
    print(f"Base: {base}")
    print(f"Out:  {args.out}")
    t0 = time.time()

    print("Discovering pages from search_index.json …")
    pages = discover_from_search_index(base, args.timeout, ctx)
    print(f"  search_index unique pages: {len(pages)}")

    if not args.no_crawl:
        print("BFS link crawl for extras …")
        before = len(pages)
        pages = crawl_extra(pages, base, args.timeout, ctx, args.max_crawl_fetches)
        print(f"  after crawl: {len(pages)} (+{len(pages) - before})")

    pages_list = sorted(pages, key=lambda p: (p.count("/"), p))
    print(f"Scraping {len(pages_list)} pages with {args.workers} workers …")

    ok = fail = 0
    written = 0
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {
            pool.submit(
                scrape_one,
                p,
                base,
                args.out,
                args.timeout,
                ctx,
                args.dry_run,
            ): p
            for p in pages_list
        }
        done = 0
        for fut in as_completed(futs):
            page_path, status, nbytes = fut.result()
            done += 1
            if status.startswith("ok") or status.startswith("dry"):
                ok += 1
                written += nbytes
            else:
                fail += 1
                failures.append(f"{page_path or '(home)'}: {status}")
            if done % 50 == 0 or done == len(pages_list):
                print(f"  progress {done}/{len(pages_list)} ok={ok} fail={fail}")

    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed:.1f}s — ok={ok} fail={fail} "
        f"bytes≈{written:,} dry_run={args.dry_run}"
    )
    if failures:
        print("Failures (first 30):")
        for line in failures[:30]:
            print(" ", line)
    manifest = os.path.join(ROOT, "data", "scrape_manifest.json")
    if not args.dry_run:
        os.makedirs(os.path.dirname(manifest), exist_ok=True)
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "base": base,
                    "out": args.out,
                    "pages": len(pages_list),
                    "ok": ok,
                    "fail": fail,
                    "elapsed_s": round(elapsed, 1),
                    "failures": failures,
                    "page_paths": pages_list,
                },
                f,
                indent=2,
            )
        print(f"Manifest: {manifest}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
