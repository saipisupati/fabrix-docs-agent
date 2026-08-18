"""
freshness.py — per-page scrape hashes, rebuild gate, and retired-source filter.

Not live scrape-on-/ask. Batch scrape writes data/scrape_manifest.json;
retrieve() / retrieve_kb() drop sources listed in data/retired_sources.json.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from config import (
    COLLECTION_NAME,
    FRESHNESS_STATUS_PATH,
    QDRANT_DIR,
    RETIRED_SOURCES_PATH,
    SCRAPE_MANIFEST_PATH,
)

# Tests may monkeypatch this.
RETIRED_PATH = RETIRED_SOURCES_PATH


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data or b"").hexdigest()


def page_record(
    path: str,
    data: bytes,
    status: str = "ok",
    md_rel: str = "",
) -> dict[str, Any]:
    return {
        "path": path,
        "md_rel": md_rel,
        "sha256": hash_bytes(data) if data else "",
        "bytes": len(data or b""),
        "scraped_at": utc_now(),
        "status": status,
    }


def diff_manifests(old: dict | None, new: dict | None) -> dict[str, Any]:
    old_m = (old or {}).get("pages_meta") or {}
    new_m = (new or {}).get("pages_meta") or {}
    changed, added, removed = [], [], []
    for path, rec in new_m.items():
        if path not in old_m:
            added.append(path)
        elif (rec or {}).get("sha256") != (old_m[path] or {}).get("sha256"):
            changed.append(path)
    for path in old_m:
        if path not in new_m:
            removed.append(path)
    return {
        "changed_paths": sorted(changed),
        "added_paths": sorted(added),
        "removed_paths": sorted(removed),
        "has_content_changes": bool(changed or added or removed),
    }


def should_rebuild(manifest: dict | None) -> bool:
    if not manifest:
        return True
    if "has_content_changes" in manifest:
        return bool(manifest["has_content_changes"])
    return True


def md_rel_to_ingest_source(md_rel: str) -> str:
    """Map scraped markdown path to ingest metadata.source."""
    rel = (md_rel or "").replace("\\", "/").strip()
    if not rel:
        return ""
    if rel.startswith("Bots/"):
        return rel.rsplit("/", 1)[-1]
    return rel


def ingest_sources_from_manifest(manifest: dict | None) -> tuple[set[str], set[str]]:
    """Return (delete_sources, upsert_sources) from manifest page diffs."""
    man = manifest or {}
    pages_meta = man.get("pages_meta") or {}

    def page_paths_to_sources(page_paths: list[str] | set[str]) -> set[str]:
        out: set[str] = set()
        for page_path in page_paths or []:
            rec = pages_meta.get(page_path) or {}
            md_rel = rec.get("md_rel") or ""
            src = md_rel_to_ingest_source(md_rel)
            if src:
                out.add(src)
        return out

    delete_paths = set(man.get("removed_paths") or []) | set(man.get("changed_paths") or [])
    upsert_paths = set(man.get("changed_paths") or []) | set(man.get("added_paths") or [])
    return page_paths_to_sources(delete_paths), page_paths_to_sources(upsert_paths)


def manifest_has_page_hashes(manifest: dict | None) -> bool:
    return bool((manifest or {}).get("pages_meta"))


def load_json(path: str, default: Any = None) -> Any:
    if not os.path.isfile(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_manifest(path: str = SCRAPE_MANIFEST_PATH) -> dict:
    data = load_json(path, default={})
    return data if isinstance(data, dict) else {}


def write_freshness_status(payload: dict, path: str = FRESHNESS_STATUS_PATH) -> None:
    dump_json(path, payload)


def load_retired(path: str | None = None) -> dict:
    p = path or RETIRED_PATH
    data = load_json(p, default={"sources": [], "updated_at": None})
    if not isinstance(data, dict):
        return {"sources": [], "updated_at": None}
    srcs = [str(s).strip() for s in (data.get("sources") or []) if str(s).strip()]
    return {"sources": srcs, "updated_at": data.get("updated_at")}


def retired_set(path: str | None = None) -> set[str]:
    return {s.replace("\\", "/") for s in load_retired(path)["sources"]}


def save_retired(sources: list[str], path: str | None = None) -> dict:
    p = path or RETIRED_PATH
    payload = {
        "sources": sorted({s.replace("\\", "/").strip() for s in sources if s and str(s).strip()}),
        "updated_at": utc_now(),
    }
    dump_json(p, payload)
    return payload


def retire_source(source: str, path: str | None = None) -> dict:
    cur = load_retired(path)
    srcs = list(cur["sources"])
    s = (source or "").replace("\\", "/").strip()
    if s and s not in srcs:
        srcs.append(s)
    return save_retired(srcs, path)


def unretire_source(source: str, path: str | None = None) -> dict:
    s = (source or "").replace("\\", "/").strip()
    cur = load_retired(path)
    srcs = [x for x in cur["sources"] if x != s]
    return save_retired(srcs, path)


def source_is_retired(source: str, retired: set[str] | None = None) -> bool:
    if retired is None:
        retired = retired_set()
    if not retired:
        return False
    s = (source or "").replace("\\", "/").strip()
    if not s:
        return False
    if s in retired:
        return True
    base = s.rsplit("/", 1)[-1]
    if base in retired:
        return True
    for r in retired:
        rb = r.rsplit("/", 1)[-1]
        if s == r or s.endswith("/" + r) or base == rb:
            return True
    return False


def filter_retired_chunks(chunks: list[dict], retired: set[str] | None = None) -> list[dict]:
    if retired is None:
        retired = retired_set()
    if not retired:
        return chunks
    out = []
    for c in chunks:
        meta = c.get("metadata") or {}
        src = meta.get("source") or meta.get("source_file") or ""
        if source_is_retired(str(src), retired):
            continue
        out.append(c)
    return out


def filter_retired_entries(entries: list[dict], retired: set[str] | None = None) -> list[dict]:
    if retired is None:
        retired = retired_set()
    if not retired:
        return entries
    out = []
    for e in entries:
        src = e.get("source") or ""
        if source_is_retired(str(src), retired):
            continue
        out.append(e)
    return out


def qdrant_lock_held(qdrant_dir: str | None = None) -> bool:
    return os.path.isfile(os.path.join(qdrant_dir or QDRANT_DIR, ".lock"))


def load_kb_status(qdrant_client=None) -> dict[str, Any]:
    from kb.store import load_kb

    man = load_manifest()
    retired = load_retired()
    status = load_json(FRESHNESS_STATUS_PATH, default={}) or {}
    kb = load_kb()
    chunk_count = None
    if qdrant_client is not None:
        try:
            chunk_count = int(qdrant_client.count(collection_name=COLLECTION_NAME).count)
        except Exception:
            chunk_count = None
    return {
        "qdrant_collection": COLLECTION_NAME,
        "chunk_count": chunk_count,
        "kb_topics": len(kb.topics) if kb else 0,
        "kb_entities": len(kb.entities) if kb else 0,
        "last_scrape_at": man.get("scraped_at"),
        "last_rebuild": status if isinstance(status, dict) else {},
        "changed_paths": list(man.get("changed_paths") or [])[:50],
        "added_paths": list(man.get("added_paths") or [])[:50],
        "removed_paths": list(man.get("removed_paths") or [])[:50],
        "has_content_changes": man.get("has_content_changes"),
        "retired": retired.get("sources") or [],
        "retired_updated_at": retired.get("updated_at"),
        "manifest_ok": man.get("ok"),
        "manifest_fail": man.get("fail"),
        "manifest_pages": man.get("pages"),
        "qdrant_locked": qdrant_lock_held(),
    }
