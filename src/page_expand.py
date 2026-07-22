"""
page_expand.py — load full markdown pages from DOCS_ROOT when retrieve agrees on a path.

Closes the "right URL, wrong chunk" gap: after vector retrieve, expand high-confidence
doc paths to full page text (tables, CLI blocks) without per-question hardcoding.
"""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import unquote, urlparse

from config import BOTS_DIR, DOCS_ROOT
from doc_urls import BOTS_REL_PREFIX, public_doc_url

logger = logging.getLogger(__name__)

EXPANDABLE_PREFIXES: tuple[str, ...] = (
    f"{BOTS_REL_PREFIX}/",
    "beginners_guide/",
    "installation_guides/",
    "reference_guides/",
    "Datasource_Integrations/",
    "Pipelines/",
    "ai_fabric/",
    "Extensions/",
)

DEMOTE_PATH_MARKERS: tuple[str, ...] = (
    "update_image_repository",
    "update_docker_regsitry",
)

DEFAULT_PAGE_MAX_CHARS = 16_000
DEFAULT_TOTAL_MAX_CHARS = 20_000
DEFAULT_LIMIT_PAGES = 2


def _strip_html_comment(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text or "", flags=re.DOTALL).strip()


def normalize_source_path(
    raw: str,
    *,
    chunk_type: str = "",
) -> str | None:
    """
    Map KB source, chunk metadata, or docs URL to a repo-relative path under DOCS_ROOT.
    Returns path like Bots/kafka-v2.md or beginners_guide/scheduled_pipelines.md.
    """
    if not raw:
        return None
    s = raw.strip().replace("\\", "/")
    if not s:
        return None

    if s.startswith("http://") or s.startswith("https://"):
        parsed = urlparse(s)
        if "docs.fabrix.ai" not in (parsed.netloc or ""):
            return None
        path = unquote(parsed.path or "").strip("/")
        if not path:
            return "index.md"
        if not path.endswith(".md"):
            path = f"{path}.md"
        return path

    if s.startswith("/"):
        s = s.lstrip("/")

    if any(m in s.lower() for m in DEMOTE_PATH_MARKERS):
        return None

    # Already a relative doc path
    if "/" in s and s.endswith(".md"):
        return s

    # Narrative chunk: beginners_guide/foo.md
    if "/" in s:
        return s if s.endswith(".md") else f"{s}.md"

    # Bot catalog chunk metadata.source is basename only (e.g. kafka-v2.md)
    if s.endswith(".md"):
        if chunk_type == "bot" or (DOCS_ROOT and os.path.isfile(os.path.join(BOTS_DIR, s))):
            return f"{BOTS_REL_PREFIX}/{s}"
        if DOCS_ROOT and os.path.isfile(os.path.join(DOCS_ROOT, s)):
            return s

    return None


def _path_from_kb_entry(entry: dict) -> str | None:
    for key in ("source", "url"):
        val = entry.get(key) or ""
        p = normalize_source_path(str(val))
        if p:
            return p
    return None


def _path_from_chunk(chunk: dict) -> str | None:
    meta = chunk.get("metadata") or {}
    src = meta.get("source") or ""
    ctype = meta.get("type") or ""
    p = normalize_source_path(str(src), chunk_type=ctype)
    if p:
        return p
    if ctype == "bot" and src:
        return normalize_source_path(src, chunk_type="bot")
    return None


def _is_expandable_path(path: str) -> bool:
    if not path:
        return False
    if any(m in path.lower() for m in DEMOTE_PATH_MARKERS):
        return False
    return any(path.startswith(prefix) for prefix in EXPANDABLE_PREFIXES) or path.endswith(
        ".md"
    )


def collect_expand_candidates(
    kb_entries: list[dict],
    chunks: list[dict],
) -> dict[str, float]:
    """Score doc paths by retrieval rank (higher = more confident)."""
    scores: dict[str, float] = {}
    for i, e in enumerate(kb_entries or []):
        p = _path_from_kb_entry(e)
        if p and _is_expandable_path(p):
            scores[p] = scores.get(p, 0.0) + (12 - i)
    for i, c in enumerate(chunks or []):
        p = _path_from_chunk(c)
        if p and _is_expandable_path(p):
            scores[p] = scores.get(p, 0.0) + (8 - i)
    return scores


def _resolve_filesystem_path(rel_path: str) -> str | None:
    if not DOCS_ROOT or not rel_path:
        return None
    rel_path = rel_path.replace("\\", "/")
    candidates = [
        os.path.join(DOCS_ROOT, rel_path),
    ]
    if rel_path.startswith(f"{BOTS_REL_PREFIX}/"):
        candidates.append(os.path.join(BOTS_DIR, os.path.basename(rel_path)))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def expand_page(rel_path: str, max_chars: int = DEFAULT_PAGE_MAX_CHARS) -> str | None:
    """Read full markdown for rel_path under DOCS_ROOT; None if missing."""
    fpath = _resolve_filesystem_path(rel_path)
    if not fpath:
        return None
    try:
        with open(fpath, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        logger.info("page_expand: read failed %s (%s)", rel_path, e)
        return None
    text = _strip_html_comment(text)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... page truncated for context limit ...]"
    return text


def expand_context(
    kb_entries: list[dict],
    chunks: list[dict],
    *,
    limit_pages: int = DEFAULT_LIMIT_PAGES,
    page_max_chars: int = DEFAULT_PAGE_MAX_CHARS,
    total_max_chars: int = DEFAULT_TOTAL_MAX_CHARS,
) -> list[dict]:
    """
    Return expanded pages: [{path, url, text}, ...].

    Expand when:
    - path appears in 2+ ranked hits, OR
    - Bots/ rank-1 style strong hit (score >= 8), OR
    - any path with score >= 12
    """
    scores = collect_expand_candidates(kb_entries, chunks)
    if not scores:
        return []

    selected: list[str] = []
    for path, score in sorted(scores.items(), key=lambda x: (-x[1], x[0])):
        count_hint = score >= 16  # roughly 2+ hits
        bot_strong = path.startswith(f"{BOTS_REL_PREFIX}/") and score >= 8
        if count_hint or bot_strong or score >= 12:
            selected.append(path)
        if len(selected) >= limit_pages:
            break

    if not selected and scores:
        # Single best path if clearly bot catalog or top narrative
        best_path, best_score = max(scores.items(), key=lambda x: x[1])
        if best_score >= 6:
            selected = [best_path]

    pages: list[dict] = []
    total = 0
    for path in selected[:limit_pages]:
        remaining = total_max_chars - total
        if remaining < 2000:
            break
        cap = min(page_max_chars, remaining)
        text = expand_page(path, max_chars=cap)
        if not text:
            continue
        pages.append({
            "path": path,
            "url": public_doc_url(path),
            "text": text,
        })
        total += len(text)

    if pages:
        logger.info(
            "page_expand: loaded %s page(s): %s",
            len(pages),
            ", ".join(p["path"] for p in pages),
        )
    return pages


def pages_to_kb_entries(pages: list[dict]) -> list[dict]:
    """Synthetic KB entries prepended before retrieve hits (like live_docs)."""
    entries: list[dict] = []
    for i, page in enumerate(pages):
        entries.append({
            "id": f"full-page-{i}",
            "kind": "full_page",
            "title": f"Full doc page: {page['path']}",
            "text": page["text"],
            "source": page["path"],
            "url": page.get("url") or public_doc_url(page["path"]),
            "score": 1.0,
            "example": "",
        })
    return entries
