# KB freshness + admin (post-demo)

Operator-facing guide. **Not live scrape-on-`/ask`.** Retrieval still uses local Qdrant + `data/kb`. Freshness is a batch job.

## What shipped

1. Per-page SHA-256 on `data/scrape_manifest.json` (`pages_meta`, `changed_paths` / `added_paths` / `removed_paths`) — written by `scripts/scrape_docs_site.py`
2. `scripts/run_freshness_pipeline.py` skips ingest/`build_kb`/evals when hashes are unchanged (`--force` or `FORCE_REBUILD=1` to override). When hashes did change, ingest runs `--incremental` (full wipe only with `--force`).
3. `data/retired_sources.json` — retrieval-time filter in `retrieve()` and `retrieve_kb()`
4. `/admin/kb-status`, `/admin/refresh`, `/admin/retire`, `/admin/unretire` behind `API_KEY`
5. Operator console: `chat/admin.html`

## Admin UI

Serve the same way as chat (`python3 -m http.server 5173 --directory chat`). There is **no** Admin link on the demo chat opening page — bookmark `/admin.html`.

```bash
# Chat + API locally, then open admin
./scripts/demo_start.sh
# http://127.0.0.1:5173/admin.html
```

1. Set `API_KEY` on the uvicorn process.
2. Paste the same value into the API key field and Save (stored in this browser only as `fabrix_admin_key`).
3. Status cards load from `GET /admin/kb-status` (auto-refresh every 30s while the tab is visible).
4. **Retire** a `metadata.source` path or basename; **Unretire** from the table. Filter is retrieve-time only — Qdrant points stay.
5. **Scrape + gate** runs check-live → scrape → audit, then incremental ingest if page hashes changed.
6. **Force rebuild** always wipes Qdrant and runs full ingest + `build_kb` + evals. Stop uvicorn first (Qdrant file lock).

When `API_KEY` is set on the API, chat `/ask` also requires `X-API-Key`. Save the key once on the admin page (`fabrix_admin_key` in localStorage); the chat UI reuses that key automatically.

After refresh, the page polls status every 5s for about two minutes. Pipeline stdout stays in the shell/CI log; the UI does not tail it.

## Operator commands

```bash
# Stop uvicorn first if you need ingest (Qdrant file lock)
python3 scripts/run_freshness_pipeline.py           # incremental if hashes changed
python3 scripts/run_freshness_pipeline.py --force   # full wipe + ingest
python3 src/ingest_qdrant.py --incremental          # changed/added/removed pages only
python3 src/ingest_qdrant.py --full                 # wipe collection
```

## Admin API

All `/admin/*` require `X-API-Key` matching `API_KEY`. If `API_KEY` is unset, admin returns 503.

| Method | Path | Body |
|--------|------|------|
| GET | `/admin/kb-status` | — |
| POST | `/admin/retire` | `{"source": "Bots/cfxdm.md"}` |
| POST | `/admin/unretire` | `{"source": "Bots/cfxdm.md"}` |
| POST | `/admin/refresh` | `{"force": false}` |

`/admin/refresh` returns `{accepted, pid}` immediately. Scrape/audit can run while the API is up; ingest/`build_kb` will fail until uvicorn is stopped. Status includes `qdrant_locked` so the UI can show that warning.

## Tests (no full harness)

```bash
PYTHONPATH=src python3 tests/test_freshness.py
```

Bot catalog pages scrape as `Bots/cfxdm.md` but ingest stores `metadata.source` as `cfxdm.md`. Incremental delete/upsert uses that mapping.

## Not in v1

- Deleting Qdrant points on retire
- Scraping inside `POST /ask`
- Searchable full source inventory or pipeline log viewer in the browser
- Admin link on the demo chat opening page
