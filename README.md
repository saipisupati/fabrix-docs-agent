# Fabrix Docs Agent

A retrieval-augmented generation (RAG) agent that answers questions about Fabrix.ai documentation, covering the RDA bot catalog, CFXQL reference, and platform guides,

## Status

Working prototype, full pipeline (chunk → embed → store → retrieve → generate) runs end-to-end against the real corpus: 214 bot catalog files (~1,834 bots) plus the CFXQL reference, ~1,844 chunks total, embeddings use `sentence-transformers/all-minilm-l6-v2` via OpenRouter, generation uses OpenAI `gpt-4o-mini`, a separate remote ingest path targets a shared Qdrant server (see below), currently in progress,

**Known limitation:** not self-contained yet, bot catalog and CFXQL source paths are hardcoded in the scripts (`REAL_BOTS_DIR`, `CFXQL_FILE`), update those before running on another machine,

## How it works (local path — primary)

1. **Chunk** (`src/ingest_qdrant.py` → `load_and_chunk_all()`) — loads sample files from `data/raw/` plus the real bot catalog from `REAL_BOTS_DIR`, bot files use `chunk_bot_catalog_markdown()` (cleans HTML/CSS, splits on `##` headers, one chunk per bot), CFXQL uses `CHUNKING_STRATEGY` (default: `hand_rolled`),
2. **Embed + store** (`src/ingest_with_real_embeddings.py`) — embeds all chunks in batches via OpenRouter, stores in local Qdrant at `data/qdrant_db/`, requires `OPENROUTER_API_KEY`,
3. **Query + generate** (`src/query_qdrant.py`) — embeds the question with the same model, retrieves top-k chunks, builds a grounded prompt, generates via `gpt-4o-mini`, requires `OPENROUTER_API_KEY` and `OPENAI_API_KEY`,

**CFXQL chunking strategies** (`CHUNKING_STRATEGY` in `ingest_qdrant.py`):
- `hand_rolled` (default) — hardcoded splits at this doc's headers, most accurate, doesn't generalize,
- `heuristic` — generic header detection on plain text, generalizes but noisier,
- `size_based` — character-count splitting, breaks the `comparison_01` eval case, avoid,

## Remote path (in progress)

`src/ingest_and_test_remote.py` uploads raw markdown to a hosted Qdrant server (`10.95.121.54:8000`), chunking and embedding happen server-side using `BAAI/bge-large-en-v1.5`, requires VPN, uses a different embedding model than the local path, some large bot files still timeout at 120s,

## Legacy path (not used)

`src/ingest.py` / `src/query.py` — original Chroma + TF-IDF prototype, kept for reference, needs `chromadb` installed separately,

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY=your_key_here   # embeddings
export OPENAI_API_KEY=your_key_here       # generation
```

## Running it

**Local (recommended)**

```bash
# chunk + embed + store
python3 src/ingest_with_real_embeddings.py

# query with generation
python3 src/query_qdrant.py "what parameters does the count loop bot take?"

# eval suite (manual grading)
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

`tests/eval_set.py` — 7 hand-built cases (lookup, comparison, multi-part, negative/hallucination), `tests/run_eval.py` runs each through `query_qdrant.ask()` for manual pass/fail/partial grading, not automated yet,

## Open questions / known issues

- Hardcoded local paths (`REAL_BOTS_DIR`, `CFXQL_FILE`) — should move to env vars,
- Two embedding models across paths (MiniLM local, BGE-large remote), no shared config,
- Remote ingestion timeouts on larger bot catalog files,
- Eval scoring is manual,
- Legacy Chroma path and unused sample `.txt` bot files in `data/raw/` should be removed,
- `ingest_qdrant.py` `main()` still has TF-IDF code — use `ingest_with_real_embeddings.py` instead,
- `data/qdrant_db/` is tracked in git and grows with each re-ingest,
