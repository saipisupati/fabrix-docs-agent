# KB freshness + admin (post-demo)

Operator-facing guide. Implementation plan: [superpowers/plans/2026-08-13-kb-freshness-admin.md](superpowers/plans/2026-08-13-kb-freshness-admin.md)

**Not live scrape-on-`/ask`.** Retrieval still uses local Qdrant + `data/kb`. Freshness is a batch job.

## What shipped

1. Per-page SHA-256 on `data/scrape_manifest.json` (`pages_meta`, `changed_paths` / `added_paths` / `removed_paths`)
2. `scripts/run_freshness_pipeline.py` skips ingest/`build_kb`/evals when hashes are unchanged (`--force` or `FORCE_REBUILD=1` to override)
3. `data/retired_sources.json` — retrieval-time filter in `retrieve()` and `retrieve_kb()`
4. `/admin/kb-status`, `/admin/refresh`, `/admin/retire`, `/admin/unretire` behind `API_KEY`
5. Thin `chat/admin.html` (same look as the demo chat)

## Operator commands

```bash
# Stop uvicorn first if you need ingest (Qdrant file lock)
python3 scripts/run_freshness_pipeline.py
python3 scripts/run_freshness_pipeline.py --force

# Chat + admin (admin needs API_KEY in the API process env)
./scripts/demo_start.sh
# http://127.0.0.1:5173/admin.html
```

Retire a path (exact `metadata.source` or basename): `Bots/cfxdm.md`. Unretire from the same page. Retire does **not** delete Qdrant points — they are dropped at retrieve time.

## Admin API

All `/admin/*` require `X-API-Key` matching `API_KEY`. If `API_KEY` is unset, admin returns 503.

| Method | Path | Body |
|--------|------|------|
| GET | `/admin/kb-status` | — |
| POST | `/admin/retire` | `{"source": "Bots/cfxdm.md"}` |
| POST | `/admin/unretire` | `{"source": "Bots/cfxdm.md"}` |
| POST | `/admin/refresh` | `{"force": false}` |

`/admin/refresh` returns `{accepted, pid}` immediately. Scrape/audit can run while the API is up; ingest/`build_kb` will fail until uvicorn is stopped.

## Tests (no full harness)

```bash
PYTHONPATH=src python3 tests/test_freshness.py
PYTHONPATH=src python3 tests/test_retire_filter.py
```

## Not in v1

- Incremental upsert of only changed pages
- Deleting Qdrant points on retire
- Scraping inside `POST /ask`
