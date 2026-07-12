# Production deploy checklist

Checklist for hosting the Fabrix docs Q&A API and embedding the ask widget on a documentation site.

Related docs:

- Widget embed snippet: [DOCS_SITE_INTEGRATION.md](DOCS_SITE_INTEGRATION.md)
- Demo script: [DEMO_SCRIPT.md](DEMO_SCRIPT.md)
- Quality gate: [QUALITY_LOOP.md](QUALITY_LOOP.md)
- Local setup: [README.md](../README.md)

---

## A. Prerequisites

- [ ] API host decided (VM, container, or managed service). No Dockerfile in repo today; v1 path is bare `uvicorn`.
- [ ] Documentation site deploy path agreed (for widget embed)
- [ ] `OPENROUTER_API_KEY` provisioned (embeddings)
- [ ] `OPENAI_API_KEY` provisioned (generation)
- [ ] Qdrant DB available on the host:
  - Copy `data/qdrant_db/` from a machine that ran `ingest_qdrant.py`, **or**
  - Re-run ingest on the server with correct `BOTS_DIR` / `DOCS_ROOT` / `CFXQL_FILE` paths
- [ ] Structured KB available on the host:
  - Copy `data/kb/` from a machine that ran `python3 src/build_kb.py`, **or**
  - Rebuild KB on the server after ingest (stop API first if upserting into the same Qdrant path)
- [ ] Pre-ingest audit passes: `python3 scripts/audit_ingest_sources.py` (expect 543 files)
- [ ] Quality harness passes: `python3 tests/run_quality_harness.py` (exit 0, readiness streak ≥ 2)

---

## B. API server deploy

### Install

```bash
git clone <repo-url>
cd fabrix-docs-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

Create `.env` on the server (never commit):

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENROUTER_API_KEY` | yes | Embedding queries via OpenRouter |
| `OPENAI_API_KEY` | yes | Answer generation (gpt-4o-mini) |
| `DOCS_SITE_ORIGIN` | yes | CORS: set to your docs site origin (e.g. `https://docs.example.com`) |
| `API_KEY` | optional | If set, clients must send `X-API-Key` on `POST /ask` |
| `LLM_MODEL` | optional | Default `gpt-4o-mini` |
| `BOTS_DIR` / `DOCS_ROOT` / `CFXQL_FILE` | if re-ingesting | Paths to public MD export |

- [ ] All required env vars set
- [ ] `data/qdrant_db/` present on host
- [ ] `data/kb/` present on host (`kb.json` + `embeddings.npz`)

### Start the API

Production (no reload):

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8080
```

- [ ] Process manager configured (systemd, supervisor, etc.) so the API restarts on failure
- [ ] TLS termination in front of the API (nginx, ALB, Cloudflare, etc.). The widget **requires HTTPS** in production.

### Qdrant constraint

The API uses local file Qdrant (`QdrantClient(path=data/qdrant_db)`). Only one process can open the DB at a time.

- [ ] Do not run eval scripts on the prod host while the API is up
- [ ] For concurrent access later, migrate to a Qdrant server (out of scope for v1)

---

## C. Post-deploy verification

Replace `<api-host>` with your HTTPS base URL (no trailing slash).

### Health

```bash
curl https://<api-host>/health
```

Expected: `{"status":"ok"}`

### Ask endpoint

```bash
curl -X POST https://<api-host>/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What parameters does the count loop bot take?"}'
```

Expected: JSON with `answer` and `sources` array with documentation links.

### Optional API key

If `API_KEY` is set:

```bash
curl -X POST https://<api-host>/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-api-key>" \
  -d '{"question": "What parameters does the count loop bot take?"}'
```

- [ ] Request without key returns 401 when `API_KEY` is configured

### CORS

- [ ] Browser request from your docs site origin succeeds

### Eval (off prod host only)

On a separate machine or CI job with its own Qdrant copy:

```bash
python3 tests/run_quality_harness.py
```

---

## D. Widget embed

See [DOCS_SITE_INTEGRATION.md](DOCS_SITE_INTEGRATION.md) for full snippet.

- [ ] Copy `widget/ask-widget.js` and `widget/ask-widget.css` into docs site static assets
- [ ] Wire assets into your static site or docs generator
- [ ] Set `data-api-url="https://<api-host>"` (no trailing slash)
- [ ] If using `API_KEY`, configure `window.FabrixAskConfig` before loading the script
- [ ] Browser test: ask a question, confirm answer renders and source links work

---

## E. Ongoing ops

- [ ] Doc update process: `audit_ingest_sources.py` → `ingest_qdrant.py` → `build_kb.py` → redeploy `data/qdrant_db/` + `data/kb/` → restart API
- [ ] Monitor `/health` uptime
- [ ] Monitor OpenAI / OpenRouter quota and `/ask` error rate
- [ ] Secrets stay server-side only (never in widget or public HTML)

---

## What stays server-side

Never expose in the widget or public HTML:

- `OPENROUTER_API_KEY`
- `OPENAI_API_KEY`
- `API_KEY`
- Qdrant path / `data/qdrant_db/`

The widget only calls `POST {apiUrl}/ask`.
