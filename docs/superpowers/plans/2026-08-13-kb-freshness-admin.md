# KB freshness + admin layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-page scrape freshness tracking, gate the existing rebuild pipeline on real doc changes, let admins retire/unretire sources from retrieval, and expose a thin `/admin` API + status page — without rewriting scrape/ingest/KB.

**Architecture:** Keep `scripts/scrape_docs_site.py` → `scripts/sync_docs_and_rebuild.py` → `scripts/run_freshness_pipeline.py` as the batch path. New module `src/freshness.py` owns manifest page records, diffs, and `data/retired_sources.json`. `query_qdrant.retrieve()` and `kb.store.retrieve_kb()` drop retired sources. FastAPI `/admin/*` is a thin layer over those files + a background refresh subprocess. Chat-styled `chat/admin.html` is read-mostly UI.

**Tech Stack:** existing Python 3.12 / FastAPI / Qdrant local path / hashlib SHA-256 / same `API_KEY` header as `/ask`.

## Global Constraints

- Additive cycle style: no RAG rewrite, no scrape-on-every-`/ask`.
- Never `if question == "..."` in `src/agent.py`.
- Stop uvicorn before ingest/rebuild (Qdrant file lock) — admin refresh must document this and refuse if the lock is held, or spawn after a warning.
- `data/scrape_manifest.json` today is run-level only: `base`, `out`, `pages`, `ok`, `fail`, `elapsed_s`, `failures`, `page_paths` (653 strings). No etag/hash/mtime.
- Chunk `metadata.source` is the ingest source name (e.g. `Bots/cfxdm.md`); retire keys use that same string.
- Gate all `/admin/*` with existing `_check_api_key` in `src/api.py` (require `API_KEY` set in env for admin routes even if `/ask` is open).
- Full `python3 tests/run_quality_harness.py` after Phase C (retrieve path) and again after Phase F.
- Hard rule from `docs/CONTINUOUS_QUALITY.md`: one class per round; generic filters.

---

### Task 1: Per-page freshness tracking (Phase A)

**Files:**
- Create: `src/freshness.py`
- Create: `tests/test_freshness.py`
- Modify: `scripts/scrape_docs_site.py` (manifest write ~478–495)
- Modify: `src/config.py` (paths)

**Interfaces:**
- Consumes: scrape output files under `args.out`; previous `data/scrape_manifest.json`
- Produces:
  - `PageRecord = {path: str, sha256: str, bytes: int, scraped_at: str, status: str}`
  - `manifest["pages_meta"]: dict[str, PageRecord]`
  - `manifest["changed_paths"]: list[str]`
  - `manifest["removed_paths"]: list[str]`
  - `manifest["added_paths"]: list[str]`
  - `diff_manifests(old: dict, new: dict) -> dict`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_freshness.py
from freshness import hash_bytes, diff_manifests, page_record


def test_hash_bytes_stable():
    assert hash_bytes(b"hello") == hash_bytes(b"hello")
    assert hash_bytes(b"hello") != hash_bytes(b"Hello")


def test_diff_detects_changed_added_removed():
    old = {
        "pages_meta": {
            "a.md": {"path": "a.md", "sha256": "111", "bytes": 1, "scraped_at": "t0", "status": "ok"},
            "gone.md": {"path": "gone.md", "sha256": "222", "bytes": 1, "scraped_at": "t0", "status": "ok"},
        }
    }
    new = {
        "pages_meta": {
            "a.md": {"path": "a.md", "sha256": "999", "bytes": 2, "scraped_at": "t1", "status": "ok"},
            "b.md": {"path": "b.md", "sha256": "333", "bytes": 1, "scraped_at": "t1", "status": "ok"},
        }
    }
    d = diff_manifests(old, new)
    assert d["changed_paths"] == ["a.md"]
    assert d["added_paths"] == ["b.md"]
    assert d["removed_paths"] == ["gone.md"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 tests/test_freshness.py` (or `python3 -c` importing `freshness`)

Expected: `ModuleNotFoundError: freshness`

- [ ] **Step 3: Implement `src/freshness.py`**

```python
import hashlib
from datetime import datetime, timezone


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def page_record(path: str, data: bytes, status: str = "ok") -> dict:
    return {
        "path": path,
        "sha256": hash_bytes(data) if data else "",
        "bytes": len(data or b""),
        "scraped_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": status,
    }


def diff_manifests(old: dict, new: dict) -> dict:
    old_m = (old or {}).get("pages_meta") or {}
    new_m = (new or {}).get("pages_meta") or {}
    changed, added, removed = [], [], []
    for path, rec in new_m.items():
        if path not in old_m:
            added.append(path)
        elif rec.get("sha256") != old_m[path].get("sha256"):
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
```

Add to `src/config.py`:

```python
SCRAPE_MANIFEST_PATH = os.path.join(_PROJECT_ROOT, "data", "scrape_manifest.json")
RETIRED_SOURCES_PATH = os.path.join(_PROJECT_ROOT, "data", "retired_sources.json")
FRESHNESS_STATUS_PATH = os.path.join(_PROJECT_ROOT, "data", "freshness_status.json")
```

- [ ] **Step 4: Wire scrape manifest write**

In `scripts/scrape_docs_site.py`, load previous manifest before overwrite. For each successful page write, compute `page_record`. Keep existing top-level `pages/ok/fail/failures/page_paths`. Merge:

```python
from freshness import diff_manifests, page_record  # sys.path insert ROOT/src

prev = {}
if os.path.isfile(manifest):
    with open(manifest, encoding="utf-8") as f:
        prev = json.load(f)
# pages_meta built during the scrape loop
diff = diff_manifests(prev, {"pages_meta": pages_meta})
payload = {
    "base": base,
    "out": args.out,
    "pages": len(pages_list),
    "ok": ok,
    "fail": fail,
    "elapsed_s": round(elapsed, 1),
    "failures": failures,
    "page_paths": pages_list,
    "pages_meta": pages_meta,
    **diff,
    "prev_scraped_at": prev.get("scraped_at"),
    "scraped_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
```

Preserve backward compatibility: readers that only use `page_paths` still work.

- [ ] **Step 5: Re-run tests**

Run: `PYTHONPATH=src python3 -m pytest tests/test_freshness.py -q` if pytest installed, else `python3 tests/test_freshness.py` with a `if __name__` runner.

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/freshness.py src/config.py scripts/scrape_docs_site.py tests/test_freshness.py
git commit -m "feat: per-page scrape hashes and manifest diff"
```

---

### Task 2: Gate existing freshness pipeline (Phase B)

**Files:**
- Modify: `scripts/run_freshness_pipeline.py`
- Modify: `scripts/sync_docs_and_rebuild.py` (optional `--if-changed` flag)
- Modify: `tests/test_freshness.py`

**Interfaces:**
- Consumes: `diff_manifests` / `manifest["has_content_changes"]` after scrape
- Produces: skip ingest+`build_kb` when scrape ran and `has_content_changes` is false; still run `--check-live` + scrape + audit; write `data/freshness_status.json`

- [ ] **Step 1: Failing test for skip logic**

```python
from freshness import should_rebuild


def test_should_rebuild_true_on_changes():
    assert should_rebuild({"has_content_changes": True}) is True


def test_should_rebuild_false_when_unchanged():
    assert should_rebuild({"has_content_changes": False, "fail": 0}) is False


def test_should_rebuild_true_if_diff_missing():
    # first run / old manifests without pages_meta → rebuild
    assert should_rebuild({"ok": 10, "page_paths": ["a"]}) is True
```

- [ ] **Step 2: Implement `should_rebuild(manifest: dict) -> bool`**

```python
def should_rebuild(manifest: dict) -> bool:
    if not manifest:
        return True
    if "has_content_changes" in manifest:
        return bool(manifest["has_content_changes"])
    return True
```

- [ ] **Step 3: Gate `run_freshness_pipeline.py`**

Do **not** rewrite the step list. After `[py, sync, "--scrape"]`:

```python
import json
from freshness import should_rebuild
from config import SCRAPE_MANIFEST_PATH, FRESHNESS_STATUS_PATH

with open(SCRAPE_MANIFEST_PATH, encoding="utf-8") as f:
    man = json.load(f)
rebuild = should_rebuild(man)
status = {
    "last_scrape_at": man.get("scraped_at"),
    "has_content_changes": man.get("has_content_changes"),
    "changed_n": len(man.get("changed_paths") or []),
    "rebuild": rebuild,
}
if not rebuild:
    print("No content changes — skipping ingest/build_kb/evals")
    # write status, return 0 after audit-only
```

If `rebuild` is true, continue existing ingest → build_kb → benchmark → bakeoff.

Add CLI flag `FORCE_REBUILD=1` / `--force` to ignore the gate.

- [ ] **Step 4: Manual check**

Run scrape twice in a row (second should skip rebuild). Do not run full ingest in CI for this task unless `--force`.

- [ ] **Step 5: Commit**

```bash
git add src/freshness.py scripts/run_freshness_pipeline.py tests/test_freshness.py
git commit -m "feat: skip docs rebuild when scrape hashes are unchanged"
```

---

### Task 3: Retire mechanism (Phase C)

**Files:**
- Modify: `src/freshness.py` (`load_retired`, `save_retired`, `is_retired`, `retire`, `unretire`)
- Modify: `src/query_qdrant.py` `retrieve()` after chunks are built, before/after rerank
- Modify: `src/kb/store.py` `retrieve_kb()` after `ordered` list exists
- Create: `data/retired_sources.json` default `{"sources": [], "updated_at": null}`
- Modify: `tests/test_freshness.py`
- Modify: `tests/test_wiring_detect.py` or new `tests/test_retire_filter.py`

**Interfaces:**
- Consumes: chunk `metadata["source"]`; KB entry `source` / `source` path
- Produces: filtered lists; persist `{"sources": ["Bots/cfxdm.md"], "updated_at": iso}`

- [ ] **Step 1: Failing tests**

```python
from freshness import filter_retired_chunks, retire_source, unretire_source, load_retired
import freshness as fr


def test_filter_retired_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(fr, "RETIRED_PATH", str(tmp_path / "retired.json"))
    retire_source("Bots/secret.md")
    chunks = [
        {"text": "a", "metadata": {"source": "Bots/secret.md"}},
        {"text": "b", "metadata": {"source": "Bots/cfxdm.md"}},
    ]
    out = filter_retired_chunks(chunks)
    assert [c["metadata"]["source"] for c in out] == ["Bots/cfxdm.md"]
    unretire_source("Bots/secret.md")
    assert len(filter_retired_chunks(chunks)) == 2
```

Match source with suffix/basename fallback so `cfxdm.md` retires `Bots/cfxdm.md`:

```python
def source_is_retired(source: str, retired: set[str]) -> bool:
    s = (source or "").replace("\\", "/")
    if s in retired:
        return True
    base = s.rsplit("/", 1)[-1]
    return base in retired or s in retired
```

- [ ] **Step 2: Filter in retrieve**

In `query_qdrant.retrieve`, after building `chunks` and **before** `rerank_by_bot_name(...)[:top_k]`:

```python
from freshness import filter_retired_chunks
chunks = filter_retired_chunks(chunks)
```

Same for remote branch. Over-fetch already uses `candidate_limit`; filtering then trim to `top_k` is enough.

In `kb.store.retrieve_kb`, drop entries whose `source` is retired before scoring, or after `ordered` is built.

- [ ] **Step 3: Unit tests pass**

Run: `PYTHONPATH=src python3 -m pytest tests/test_retire_filter.py tests/test_freshness.py -q`

Expected: PASS

- [ ] **Step 4: Regression**

Stop API. `python3 tests/run_quality_harness.py` (or at least `eval_production.py` + `BREAK_CYCLE=33 tests/eval_break.py`).

Expected: prod 100% PASS, break ≥95% / 0 FAIL (empty retired list = no behavior change).

- [ ] **Step 5: Commit**

```bash
git add src/freshness.py src/query_qdrant.py src/kb/store.py tests/test_retire_filter.py data/retired_sources.json
git commit -m "feat: exclude retired sources from Qdrant and KB retrieve"
```

---

### Task 4: Admin visibility (Phase D)

**Files:**
- Modify: `src/api.py`
- Create: `chat/admin.html` (reuse CSS variables from `chat/index.html`)
- Create: `src/kb_status.py` (optional helper) or keep helpers in `freshness.py`

**Interfaces:**
- `GET /admin/kb-status` → JSON
- Produces:

```python
{
  "qdrant_collection": "fabrix_docs",
  "chunk_count": 0,          # from Qdrant count if client available
  "kb_topics": 0,
  "last_scrape_at": "...",
  "last_rebuild": {...},     # freshness_status.json
  "changed_paths": [],
  "retired": [],
  "manifest_ok": 554,
  "manifest_fail": 99,
}
```

- [ ] **Step 1: Endpoint behind API key**

Admin routes **require** `API_KEY` env to be set; if unset, return 503 `"API_KEY not configured for admin"`. Then `_check_api_key`.

```python
@app.get("/admin/kb-status")
def kb_status(x_api_key: Optional[str] = Header(default=None)):
    if not API_KEY:
        raise HTTPException(503, "API_KEY not configured for admin")
    _check_api_key(x_api_key)
    return load_kb_status(request.app.state.qdrant)
```

`load_kb_status` reads manifest, retired file, freshness_status, `load_kb()` counts, optional `client.count(COLLECTION_NAME)`.

- [ ] **Step 2: `chat/admin.html`**

Minimal page: header like Docs Agent, table of retired sources, last scrape time, changed path count, refresh button (Phase E can wire POST). GET status with `X-API-Key` from `localStorage` prompt or query `?key=` (do not commit keys).

Serve it the same way as chat: `python3 -m http.server 5173 --directory chat` → `/admin.html`. Admin page calls `http://localhost:8080/admin/kb-status`.

- [ ] **Step 3: Smoke**

`curl -H "X-API-Key: $API_KEY" http://127.0.0.1:8080/admin/kb-status`

Expected: JSON 200

- [ ] **Step 4: Commit**

```bash
git add src/api.py src/freshness.py chat/admin.html
git commit -m "feat: read-only /admin/kb-status and admin page"
```

---

### Task 5: Trigger + retire APIs (Phase E)

**Files:**
- Modify: `src/api.py`
- Modify: `chat/admin.html`
- Modify: `scripts/demo_start.sh` only if needed to document admin URL

**Interfaces:**
- `POST /admin/refresh` body `{"force": false}` — subprocess `python3 scripts/run_freshness_pipeline.py` (warn: stop writers; if Qdrant locked, 409)
- `POST /admin/retire` body `{"source": "Bots/cfxdm.md"}`
- `POST /admin/unretire` body `{"source": "Bots/cfxdm.md"}`

All require `API_KEY`.

- [ ] **Step 1: Implement handlers**

Refresh: write `data/freshness_status.json` `{running: true}` then `subprocess.Popen` with `FORCE_REBUILD=1` if force. Do not block the request for the full ingest; return `{accepted: true, pid: n}`.

Retire/unretire: call `retire_source` / `unretire_source`, return `{retired: [...]}`.

- [ ] **Step 2: Admin UI buttons**

Retire input + list with Unretire. Refresh button with confirm: “Stop other API writers first.”

- [ ] **Step 3: Auth tests**

```python
from fastapi.testclient import TestClient
# missing key → 401; with key → 200
```

If TestClient is heavy, use a small function test on `_check_api_key`.

- [ ] **Step 4: Commit**

```bash
git add src/api.py chat/admin.html
git commit -m "feat: admin refresh and retire/unretire endpoints"
```

---

### Task 6: Testing + docs (Phase F)

**Files:**
- Modify: `tests/eval_break.py` (cycle 34, 2–3 cases) only if a live retrieve leak is demonstrable; otherwise keep unit tests as the retire gate
- Modify: `docs/CONTINUOUS_QUALITY.md` companion link
- Modify: `docs/QUALITY_LOOP.md` or `docs/DEPLOY_CHECKLIST.md` one line: set `API_KEY` for admin
- Create: `docs/KB_FRESHNESS_ADMIN.md` short operator README

**Operator README contents:**
- Scrape still batch-only
- `python3 scripts/run_freshness_pipeline.py` vs `/admin/refresh`
- Retire is retrieval-time (no re-ingest required)
- Unretire is immediate
- Qdrant lock: stop uvicorn before forced rebuild ingest

- [ ] **Step 1: End-to-end change detection**

1. Snapshot manifest hashes
2. Edit one scraped markdown file on disk (or mock `pages_meta`)
3. `should_rebuild` true
4. Restore file → second diff false

- [ ] **Step 2: Retire live check**

With API up: ask a question that normally cites `Bots/cfxdm.md`; retire that source; same question must not list that source. Unretire; source may return.

- [ ] **Step 3: Full harness**

```bash
# stop API
python3 tests/run_quality_harness.py
```

Expected: exit 0 or known-good bars (prod 100%, break ≥95% 0 FAIL).

- [ ] **Step 4: Commit**

```bash
git add docs/KB_FRESHNESS_ADMIN.md docs/CONTINUOUS_QUALITY.md tests/
git commit -m "docs: KB freshness admin operator guide and regression notes"
```

---

## Out of scope

- Streaming scrape on `/ask`
- Incremental Qdrant upsert of only changed pages (full ingest remains until a later cycle)
- New design system / auth provider (Okta, etc.)
- Deleting vectors from disk on retire (filter-only is enough)

## Sizing

| Phase | Time |
|-------|------|
| A hashes + diff | ~0.5 day |
| B gate pipeline | ~0.5 day |
| C retire filter | ~0.5 day |
| D status page | ~0.5–1 day |
| E POST + auth | ~0.5 day |
| F harness + E2E | ~0.5–1 day |
| **Total** | **~3–4 days** |
