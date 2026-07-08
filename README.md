# Fabrix Docs Agent

A retrieval-augmented generation (RAG) agent that answers questions about Fabrix.ai documentation, covering the RDA bot catalog, CFXQL reference, and platform guides.

See [docs/NOTES.md](docs/NOTES.md) for project log and findings.

## Status

Working prototype: full pipeline (chunk → embed → store → retrieve → generate) runs end-to-end against the full public doc set: **543/543** MD source files (214 bot catalog + 328 narrative guides across 8 folders + 3 root pages + `cfxql.md`): **~6,387 chunks** total. Root pages (`index.md`, `Datasets.md`, `Formatting-Templates.md`) are ingested via `DOCS_ROOT_FILES`. Embeddings use `sentence-transformers/all-minilm-l6-v2` via OpenRouter; local queries auto-read `data/qdrant_db/embedding_model.txt` so dims stay matched. Generation uses OpenAI `gpt-4o-mini`. Agent eval: **19 PASS, 1 PARTIAL, 0 FAIL** (20 cases, 2026-06-26). A separate remote ingest path targets a shared Qdrant server (see below).

## How it works (local path: primary)

1. **Chunk + embed + store** (`src/ingest_qdrant.py`): loads the bot catalog from `BOTS_DIR`, CFXQL from `CFXQL_FILE`, and narrative guides from `DOCS_INCLUDE_DIRS` under `DOCS_ROOT`, chunks with strategy-specific logic, embeds via OpenRouter, and stores in local Qdrant at `data/qdrant_db/`. Requires `OPENROUTER_API_KEY`.
2. **Query + generate** (`src/query_qdrant.py`): embeds the question (model from `embedding_model.txt` on local path), retrieves top-k chunks, builds a grounded prompt, generates via `gpt-4o-mini`. Requires `OPENROUTER_API_KEY` and `OPENAI_API_KEY`.

**CFXQL chunking strategies** (`CHUNKING_STRATEGY` in `ingest_qdrant.py`):
- `hand_rolled` (default): hardcoded splits at this doc's headers, most accurate, doesn't generalize
- `heuristic`: generic header detection on plain text, generalizes but noisier
- `size_based`: character-count splitting, breaks the `comparison_01` eval case, avoid

## Remote path (in progress)

`src/ingest_and_test_remote.py` uploads raw markdown to a hosted Qdrant server. Chunking and embedding happen server-side using `BAAI/bge-large-en-v1.5`. Requires VPN. Uses a different embedding model than the local path; some large bot files still timeout at 120s.

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
OPENROUTER_API_KEY=your_key_here   # embeddings
OPENAI_API_KEY=your_key_here       # generation
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
| `LLM_MODEL` | OpenAI generation model | `gpt-4o-mini` |
| `DOCS_SITE_ORIGIN` | CORS origin for docs.fabrix.ai | `*` (local dev) |
| `API_KEY` | Optional `X-API-Key` auth on `POST /ask` | unset (no auth) |

Set `BOTS_DIR`, `DOCS_ROOT`, and `CFXQL_FILE` in `.env` (see `.env.example`).

## Running it

**Local (recommended)**

```bash
# pre-ingest checkpoint: fail if sources escape the public docs export
python3 scripts/audit_ingest_sources.py

# chunk + embed + store
python3 src/ingest_qdrant.py

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
# POST /ask {"question": "..."} -> {answer, sources}
# GET  /health -> {"status": "ok"}

# agent eval (no category hints; writes tests/eval_agent_results.txt)
python3 tests/run_eval_agent.py

# random bot retrieval spot check (15 bots, seed 42)
python3 tests/run_eval_bot_sample.py

# full pipeline eval (manual grading)
python3 tests/run_eval.py
```

**Remote (VPN required)**

```bash
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

Embed the ask widget on [docs.fabrix.ai](https://docs.fabrix.ai) and point it at a hosted API. See [docs/DOCS_SITE_INTEGRATION.md](docs/DOCS_SITE_INTEGRATION.md) for the paste snippet, CORS (`DOCS_SITE_ORIGIN`), and optional API key setup. Production deploy steps: [docs/DEPLOY_CHECKLIST.md](docs/DEPLOY_CHECKLIST.md). Demo run-of-show: [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md).

**Beta chat hosting** (standalone page for tester feedback before production embed): [docs/HOSTING_BETA.md](docs/HOSTING_BETA.md).

## Eval

- `tests/eval_set.py`: 20 hand-built cases (lookup, comparison, multi-part, guide, install, ai_fabric, pipeline, datasource, extensions, releases, negative/hallucination)
- `tests/run_eval_baseline.py`: automated retrieval-only scoring against `eval_set.py` (source hit + fact coverage in top-k chunks). Requires `OPENROUTER_API_KEY` and an ingested `data/qdrant_db/`. Results go to `tests/eval_baseline_results.txt` (gitignored).
- `tests/run_eval_generation.py`: oracle full-pipeline scoring (category + filter hints passed manually). Requires `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, and ingested `data/qdrant_db/`. Results go to `tests/eval_generation_results.txt` (gitignored).
- `tests/run_eval_agent.py`: agent path via `agent.answer()` with no manual filters (real-user proxy). Same keys and Qdrant DB required. Results go to `tests/eval_agent_results.txt` (gitignored).
- `tests/run_eval_bot_sample.py`: retrieval-only random bot sample (default 15 bots, seed 42). Grades top-1 bot hit for generic parameter questions. Results go to `tests/eval_bot_sample_results.txt` (gitignored).
- `tests/run_eval.py`: runs each case through `query_qdrant.ask()` for manual pass/fail/partial grading

## Open questions / known issues

- `BOTS_DIR` / `CFXQL_FILE` must be set in `.env` before ingest (no machine-specific defaults in repo)
- Two embedding models across paths (MiniLM local, BGE-large remote), no shared config
- Remote ingestion timeouts on larger bot catalog files
- Local Qdrant file lock: only one process (API or eval) can open `data/qdrant_db/` at a time
- `data/qdrant_db/` is gitignored: rebuild locally via ingest
