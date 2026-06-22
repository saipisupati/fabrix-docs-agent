# Fabrix Docs Agent

A retrieval-augmented generation (RAG) agent that answers questions about Fabrix.ai documentation, covering the RDA bot catalog, CFXQL reference, and platform guides.

See [docs/NOTES.md](docs/NOTES.md) for project log and findings.

## Status

Working prototype — full pipeline (chunk → embed → store → retrieve → generate) runs end-to-end against the real corpus: 214 bot catalog files (~1,834 bots) plus the CFXQL reference (`cfxql.md`), ~1,847 chunks total. Embeddings use `sentence-transformers/all-minilm-l6-v2` via OpenRouter; generation uses OpenAI `gpt-4o-mini`. A separate remote ingest path targets a shared Qdrant server (see below).

## How it works (local path — primary)

1. **Chunk + embed + store** (`src/ingest_qdrant.py`) — loads sample files from `data/raw/` plus the real bot catalog from `BOTS_DIR`, chunks with strategy-specific logic, embeds via OpenRouter, and stores in local Qdrant at `data/qdrant_db/`. Requires `OPENROUTER_API_KEY`.
2. **Query + generate** (`src/query_qdrant.py`) — embeds the question with the same model, retrieves top-k chunks, builds a grounded prompt, generates via `gpt-4o-mini`. Requires `OPENROUTER_API_KEY` and `OPENAI_API_KEY`.

**CFXQL chunking strategies** (`CHUNKING_STRATEGY` in `ingest_qdrant.py`):
- `hand_rolled` (default) — hardcoded splits at this doc's headers, most accurate, doesn't generalize
- `heuristic` — generic header detection on plain text, generalizes but noisier
- `size_based` — character-count splitting, breaks the `comparison_01` eval case, avoid

## Remote path (in progress)

`src/ingest_and_test_remote.py` uploads raw markdown to a hosted Qdrant server (`10.95.121.54:8000`). Chunking and embedding happen server-side using `BAAI/bge-large-en-v1.5`. Requires VPN. Uses a different embedding model than the local path; some large bot files still timeout at 120s.

## Legacy path (not used)

`src/ingest.py` / `src/query.py` — original Chroma + TF-IDF prototype, kept for reference. Needs `chromadb` installed separately.

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
| `BOTS_DIR` | Path to bot catalog `.md` files | machine-specific path in `config.py` |
| `CFXQL_FILE` | Path to CFXQL reference markdown | machine-specific path in `config.py` |
| `EMBEDDING_MODEL` | OpenRouter embedding model | `sentence-transformers/all-minilm-l6-v2` |
| `LLM_MODEL` | OpenAI generation model | `gpt-4o-mini` |

Override `BOTS_DIR` and `CFXQL_FILE` when running on a machine other than the one that wrote the defaults.

## Running it

**Local (recommended)**

```bash
# chunk + embed + store
python3 src/ingest_qdrant.py

# query with generation
python3 src/query_qdrant.py "what parameters does the count loop bot take?"

# retrieval-only baseline (no generation, writes tests/eval_baseline_results.txt)
python3 tests/run_eval_baseline.py

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
```

## Eval

- `tests/eval_set.py` — 7 hand-built cases (lookup, comparison, multi-part, negative/hallucination)
- `tests/run_eval_baseline.py` — automated retrieval-only scoring against `eval_set.py` (source hit + fact coverage in top-k chunks). Requires `OPENROUTER_API_KEY` and an ingested `data/qdrant_db/`. Results go to `tests/eval_baseline_results.txt` (gitignored).
- `tests/run_eval.py` — runs each case through `query_qdrant.ask()` for manual pass/fail/partial grading

## Open questions / known issues

- Default `BOTS_DIR` / `CFXQL_FILE` paths in `config.py` are machine-specific — override via env vars on other machines
- Two embedding models across paths (MiniLM local, BGE-large remote), no shared config
- Remote ingestion timeouts on larger bot catalog files
- `data/qdrant_db/` is gitignored — rebuild locally via ingest
