# Production deploy checklist

Gate checklist for hosting the Fabrix docs Q&A API and embedding the widget on [docs.fabrix.ai](https://docs.fabrix.ai).

Related docs:

- Widget embed snippet: [DOCS_SITE_INTEGRATION.md](DOCS_SITE_INTEGRATION.md)
- Local setup and env vars: [README.md](../README.md)

---

## A. Prerequisites

- [ ] Docs repo access and deploy path for docs.fabrix.ai (owner: Dheeraj)
- [ ] API host decided (VM, container, or internal service). No Dockerfile in repo today; v1 path is bare `uvicorn`.
- [ ] `OPENROUTER_API_KEY` provisioned (embeddings)
- [ ] `OPENAI_API_KEY` provisioned (generation)
- [ ] Qdrant DB available on the host:
  - Copy `data/qdrant_db/` from a machine that ran `ingest_qdrant.py`, **or**
  - Re-run ingest on the server with correct `BOTS_DIR` / `DOCS_ROOT` / `CFXQL_FILE` paths
- [ ] Pre-ingest audit passes: `python3 scripts/audit_ingest_sources.py` (expect 540 files)

---

## B. API server deploy

### Install

```bash
git clone <fabrix-docs-agent-repo>
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
| `DOCS_SITE_ORIGIN` | yes | CORS: set to `https://docs.fabrix.ai` |
| `API_KEY` | optional | If set, clients must send `X-API-Key` on `POST /ask` |
| `LLM_MODEL` | optional | Default `gpt-4o-mini` |
| `BOTS_DIR` / `DOCS_ROOT` / `CFXQL_FILE` | if re-ingesting | Override machine-specific paths in `config.py` |

- [ ] All required env vars set
- [ ] `data/qdrant_db/` present on host

### Start the API

Production (no reload):

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8080
```

- [ ] Process manager configured (systemd, supervisor, etc.) so the API restarts on failure
- [ ] TLS termination in front of the API (nginx, ALB, Cloudflare, etc.). The widget **requires HTTPS** in production.

### Qdrant constraint

The API uses local file Qdrant (`QdrantClient(path=data/qdrant_db)`). Only one process can open the DB at a time.

- [ ] Do not run `run_eval_agent.py` or other Qdrant clients on the prod host while the API is up
- [ ] For concurrent access later, migrate to a Qdrant server (out of scope for v1)

---

## C. Post-deploy verification

Replace `<api-host>` with your HTTPS base URL (no trailing slash).

### Health

```bash
curl https://<api-host>/health
```

Expected: `{"status":"ok"}`

- [ ] Health check returns 200

### Ask endpoint

```bash
curl -X POST https://<api-host>/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What parameters does the count loop bot take?"}'
```

Expected: JSON with `answer` and `sources` array including a link to `docs.fabrix.ai`.

- [ ] Ask returns a grounded answer with source links

### Optional API key

If `API_KEY` is set:

```bash
curl -X POST https://<api-host>/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"question": "What parameters does the count loop bot take?"}'
```

- [ ] Request without key returns 401 when `API_KEY` is configured

### CORS

- [ ] Browser request from `https://docs.fabrix.ai` succeeds (test after widget embed, or via devtools on the docs site)

### Eval (off prod host only)

On a separate machine or CI job with its own Qdrant copy:

```bash
python3 tests/run_eval_agent.py
```

- [ ] Agent eval run separately from prod (optional sanity check)

---

## D. Widget embed on docs.fabrix.ai

See [DOCS_SITE_INTEGRATION.md](DOCS_SITE_INTEGRATION.md) for full snippet.

- [ ] Copy `widget/ask-widget.js` and `widget/ask-widget.css` into docs site static assets
- [ ] Wire assets into MkDocs (exact hook depends on docs repo: `extra_javascript` / `extra_css` in `mkdocs.yml`, or theme override)
- [ ] Set `data-api-url="https://<api-host>"` (no trailing slash)
- [ ] If using `API_KEY`, configure `window.FabrixAskConfig` before loading the script
- [ ] Browser test: ask a question, confirm answer renders and source links open on docs.fabrix.ai

---

## E. Ongoing ops

- [ ] Doc update process defined: `audit_ingest_sources.py` → `ingest_qdrant.py` → redeploy `data/qdrant_db/` → restart API
- [ ] Monitor `/health` uptime
- [ ] Monitor OpenAI / OpenRouter quota and `/ask` error rate
- [ ] Secrets stay server-side only (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `API_KEY` never in widget or public HTML)

---

## What stays server-side

Never expose in the widget or public HTML:

- `OPENROUTER_API_KEY`
- `OPENAI_API_KEY`
- `API_KEY` (prefer env on API only; widget gets key only if you control injection)
- Qdrant path / `data/qdrant_db/`

The widget only calls `POST {apiUrl}/ask`.
