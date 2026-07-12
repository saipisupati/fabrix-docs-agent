# Fabrix Docs Agent

A retrieval-augmented generation (RAG) agent that answers questions about [Fabrix.ai](https://docs.fabrix.ai) public documentation: the RDA bot catalog, CFXQL reference, and platform guides.

## Status

Local agent over the public doc export: **543** MD source files (214 bot catalog + 328 narrative guides across 8 folders + 3 root pages + `cfxql.md`) → **~6,387 chunks** in Qdrant plus a structured KB (`data/kb/`). Embeddings: `sentence-transformers/all-minilm-l6-v2` via OpenRouter. Generation: OpenAI `gpt-4o-mini`.

**Quality harness** (see [docs/QUALITY_LOOP.md](docs/QUALITY_LOOP.md)):

| Suite | Bar | Latest |
|-------|-----|--------|
| `eval_production.py` | 100% PASS | 22/22 |
| `eval_break.py` (full cycle 1+2) | ≥95% PASS, 0 FAIL | 27/27 |
| `eval_readiness.py` | GREEN twice (pass ≥95%, p95 ≤ 45s) | GREEN ×2 |

Run `python3 tests/run_quality_harness.py` (stop the API first; local Qdrant allows one process at a time). Exit 0 means the raised bar is met.

Operator docs: [docs/QUALITY_LOOP.md](docs/QUALITY_LOOP.md), [docs/DEPLOY_CHECKLIST.md](docs/DEPLOY_CHECKLIST.md), [docs/NOTES.md](docs/NOTES.md).

## How it works (local path: primary)

1. **Chunk + embed + store** (`src/ingest_qdrant.py`): loads the bot catalog from `BOTS_DIR`, CFXQL from `CFXQL_FILE`, and narrative guides from `DOCS_INCLUDE_DIRS` under `DOCS_ROOT`, chunks with strategy-specific logic, embeds via OpenRouter, and stores in local Qdrant at `data/qdrant_db/`. Requires `OPENROUTER_API_KEY`.
2. **Structured knowledge base** (`src/build_kb.py`): extracts topics/entities/facts/procedures from the same public MD corpus into `data/kb/kb.json` + `embeddings.npz`, and optionally upserts into Qdrant collection `fabrix_kb`. Run after ingest (or whenever docs change).
3. **Agent** (`src/agent.py`): merged scope + facet planning → KB-first retrieve (chunk fallback) → path-first unified answer (`**Documented Fabrix path**` + numbered steps + `Next (inferred)` for gaps). Generic **integration-family fidelity** keeps bots/sources on the products named in the question. Optional ops critique + timing on every response. `examples` / `gaps` / `sources` / `used_inference` are separate fields for the UI; `sources` is empty when out of scope or ungrounded.
4. **Query + generate** (`src/query_qdrant.py`): embeds the question, retrieves top-k chunks, builds a grounded prompt, generates via `gpt-4o-mini`. Used as chunk fallback and by older eval scripts.

**CFXQL chunking strategies** (`CHUNKING_STRATEGY` in `ingest_qdrant.py`):
- `hand_rolled` (default): hardcoded splits at this doc's headers, most accurate, doesn't generalize
- `heuristic`: generic header detection on plain text, generalizes but noisier
- `size_based`: character-count splitting, breaks the `comparison_01` eval case, avoid

## Remote path (optional)

`src/ingest_and_test_remote.py` uploads raw markdown to a hosted Qdrant endpoint. Chunking and embedding happen server-side using `BAAI/bge-large-en-v1.5`. Set `REMOTE_BASE_URL` in `.env`. Uses a different embedding model than the local path.

## Legacy path (not used)

`src/ingest.py` / `src/query.py`: original Chroma + TF-IDF prototype, kept for reference. Needs `chromadb` installed separately.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root (gitignored):

```bash
cp .env.example .env
# Edit .env: API keys plus BOTS_DIR / CFXQL_FILE (required for ingest)
```

All scripts load `.env` automatically via `src/config.py`. You can still `export` vars in your shell instead if you prefer.

### Optional environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `BOTS_DIR` | Path to bot catalog `.md` files | **required** (set in `.env`) |
| `DOCS_ROOT` | Root of MD doc export (parent of `Bots/`) | derived from `BOTS_DIR` or set in `.env` |
| `DOCS_INCLUDE_DIRS` | Comma-separated narrative folders to ingest | 8 folders (see `config.py`) |
| `DOCS_ROOT_FILES` | Comma-separated root-level `.md` files to ingest | `index.md,Datasets.md,Formatting-Templates.md` |
| `CFXQL_FILE` | Path to CFXQL reference markdown | **required** (set in `.env`) |
| `EMBEDDING_MODEL` | OpenRouter embedding model | `sentence-transformers/all-minilm-l6-v2` |
| `COLLECTION_NAME` | Qdrant collection name | `fabrix_docs` |
| `KB_COLLECTION_NAME` | Qdrant KB collection name | `fabrix_kb` |
| `REMOTE_BASE_URL` | Optional hosted Qdrant / fastembed wrapper URL | unset (local path) |
| `LLM_MODEL` | OpenAI generation model | `gpt-4o-mini` |
| `DOCS_SITE_ORIGIN` | CORS origin for your docs site | `*` (local dev) |
| `API_KEY` | Optional `X-API-Key` auth on `POST /ask` | unset (no auth) |

Set `BOTS_DIR`, `DOCS_ROOT`, and `CFXQL_FILE` in `.env` (see `.env.example`).

## Running it

**Local (recommended)**

```bash
# pre-ingest checkpoint: fail if sources escape the public docs export
python3 scripts/audit_ingest_sources.py

# chunk + embed + store
python3 src/ingest_qdrant.py

# build structured KB (after ingest; writes data/kb/)
python3 src/build_kb.py

# query with generation
python3 src/query_qdrant.py "what parameters does the count loop bot take?"

# retrieval-only baseline (no generation, writes tests/eval_baseline_results.txt)
python3 tests/run_eval_baseline.py

# automated generation eval (writes tests/eval_generation_results.txt)
python3 tests/run_eval_generation.py

# agent CLI (auto-routing, no manual filters)
python3 src/agent.py "What parameters does the count loop bot take?"

# API server
uvicorn src.api:app --reload --port 8080
# POST /ask {"question": "..."} -> {answer, sources, examples, gaps, scope, used_inference, timing}
# GET  /health -> {"status": "ok"}

# beta chat UI (API must be running on :8080)
python3 -m http.server 5173 --directory chat
# open http://localhost:5173/

# agent eval (no category hints; writes tests/eval_agent_results.txt)
python3 tests/run_eval_agent.py

# KB/scope/inference/examples eval (writes tests/eval_kb_results.txt)
python3 tests/run_eval_kb.py

# production-style Fabrix ops battery (multi-facet + critique; stop API first)
python3 tests/eval_production.py

# adversarial break battery (stop API first)
python3 tests/eval_break.py

# readiness gate: PASS rate + p95 latency (stop API first)
python3 tests/eval_readiness.py

# quality harness: production + full break + readiness streak (stop API first)
# see docs/QUALITY_LOOP.md; exit 0 only when raised bar holds (GREEN ×2)
python3 tests/run_quality_harness.py

# random bot retrieval spot check (15 bots, seed 42)
python3 tests/run_eval_bot_sample.py

# full pipeline eval (manual grading)
python3 tests/run_eval.py
```

**Remote (optional)**

```bash
# Requires REMOTE_BASE_URL in .env
python3 src/ingest_and_test_remote.py
```

**Utilities**

```bash
# chunk-only validation across a Bots/ folder (no embed/store)
python3 src/batch_ingest_bots.py /path/to/Bots/

# chunk-only validation across narrative folders (no embed/store)
python3 scripts/batch_ingest_narrative.py

# verify eval bot names exist in BOTS_DIR
python3 scripts/audit_eval_sources.py

# verify ingest sources stay within public docs export (optional: --verify-urls)
python3 scripts/audit_ingest_sources.py

# compare fastembed models on eval retrieval (local CPU, no Qdrant re-ingest)
pip install fastembed
python3 src/test_fastembed_eval.py --models BAAI/bge-small-en-v1.5
```

## Docs site integration

Optional: embed the ask widget on a documentation site and point it at a hosted API. See [docs/DOCS_SITE_INTEGRATION.md](docs/DOCS_SITE_INTEGRATION.md) for the paste snippet, CORS (`DOCS_SITE_ORIGIN`), and optional API key setup. Deploy checklist: [docs/DEPLOY_CHECKLIST.md](docs/DEPLOY_CHECKLIST.md).

Run the quality harness until exit 0 before production deploy. See [docs/QUALITY_LOOP.md](docs/QUALITY_LOOP.md).

Standalone chat UI for local or hosted testing: [docs/HOSTING_BETA.md](docs/HOSTING_BETA.md) (chat UI hosting guide).

## Eval

**Primary gate:** `tests/run_quality_harness.py` runs production → full break → readiness and tracks a readiness GREEN streak. Playbook: [docs/QUALITY_LOOP.md](docs/QUALITY_LOOP.md). Individual suites (stop API first):

| Script | Purpose |
|--------|---------|
| `tests/eval_production.py` | Production-style Fabrix ops battery (22 cases) |
| `tests/eval_break.py` | Adversarial battery: format stress, traps, contamination (27 cases; `BREAK_CYCLE=2` for cycle 2 only) |
| `tests/eval_readiness.py` | PASS rate + p95 latency gate (14-case subset) |
| `tests/run_quality_harness.py` | Raised bar: production 100% + break ≥95%/0 FAIL + readiness GREEN ×2 |

**Legacy / development evals:**

- `tests/eval_set.py`: hand-built cases (lookup, comparison, multi-part, guide, install, ai_fabric, pipeline, datasource, extensions, releases, negative/hallucination, plus scope / inference / examples / gaps)
- `tests/run_eval_baseline.py`: retrieval-only scoring against `eval_set.py`
- `tests/run_eval_generation.py`: oracle full-pipeline scoring (category + filter hints passed manually)
- `tests/run_eval_agent.py`: agent path via `agent.answer()` with no manual filters
- `tests/run_eval_kb.py`: KB agent checks (scope, inference, examples, gaps)
- `tests/run_eval_bot_sample.py`: random bot retrieval spot check (15 bots, seed 42)
- `tests/run_eval.py`: manual grading via `query_qdrant.ask()`

All eval result files (`tests/eval_*_results.txt`, `tests/quality_harness_digest.md`, `tests/readiness_streak.json`) are gitignored.

## Deploy: copy Qdrant + KB

`data/qdrant_db/` and `data/kb/` are gitignored. On a server, either rebuild or copy from a machine that already ran ingest + KB build:

```bash
# on build machine
tar czf fabrix_runtime_data.tar.gz data/qdrant_db data/kb

# on target host (API stopped; local Qdrant is single-process)
tar xzf fabrix_runtime_data.tar.gz
# then start uvicorn as usual
```

Or rebuild on the server:

```bash
python3 scripts/audit_ingest_sources.py
python3 src/ingest_qdrant.py
python3 src/build_kb.py   # stop API first if upserting fabrix_kb into the same Qdrant path
```

**Answer contract:** one unified `answer` (ChatGPT-style). When Fabrix technical synthesis is used, `used_inference=true` and the answer includes a short disclosure line. `examples` / `gaps` / `sources` are separate fields (chat shows Examples → Sources → Gaps below the bubble). `sources` is `[]` when the agent abstains or finds no grounding.

### Technical inference smoke

Ask Fabrix-only synthesis questions (not generic OS tips). Expect `scope=related` (or `in_scope`), a single coherent **path-first** answer (named Fabrix objects before host prerequisites), `used_inference=true` (or the disclosure line), and Examples below, not a big `## Inferred` report section:

```bash
python3 src/agent.py "How would I build a Fabrix pipeline that pulls ServiceNow ticket data and writes it to a persistent stream?"
python3 src/agent.py "In Agentic AI on Fabrix, when should I use a Toolset versus a Persona?"
python3 src/agent.py "How would I chain a Kubernetes inventory collection into a dashboard-friendly dataset?"
python3 tests/run_quality_harness.py   # full gate (stop API first)
```

## Known issues

- `BOTS_DIR` / `CFXQL_FILE` must be set in `.env` before ingest (no machine-specific defaults in repo)
- Two embedding models across local vs remote paths, no shared config
- Local Qdrant file lock: only one process (API or eval) can open `data/qdrant_db/` at a time
- `data/qdrant_db/` and `data/kb/` are gitignored: rebuild via ingest + `build_kb.py`, or copy for deploy
