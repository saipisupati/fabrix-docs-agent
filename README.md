# Fabrix Docs Agent

A retrieval-augmented generation (RAG) agent that answers questions about Fabrix.ai documentation, covering the RDA bot catalog, CFXQL reference, and platform guides,

## Status

Early prototype, the pipeline (chunk → embed → store → retrieve → generate) works end-to-end on a small sample in `data/raw/`, bot catalog markdown chunking is validated across 214 files but not yet wired into the main ingest path, generation runs via OpenRouter but embeddings are still TF-IDF placeholder,

## How it works

**Primary path (Qdrant)**

1. **Ingest** (`src/ingest_qdrant.py`) — loads `.txt` files from `data/raw/`, chunks by doc type (bot catalog, CFXQL reference, or generic narrative), embeds with TF-IDF, stores in `data/qdrant_db/`
2. **Query** (`src/query_qdrant.py`) — retrieves top-k chunks from Qdrant, builds a grounded prompt, generates an answer via OpenRouter (`OPENROUTER_API_KEY` required)

**Bot catalog (markdown, separate for now)**

- `src/clean_markdown.py` — strips frontmatter, CSS, and HTML from `.md` bot pages
- `src/ingest_qdrant.py` → `chunk_bot_catalog_markdown()` — splits on `##` headers, one chunk per bot, extracts `bot_name` / `prefix` / `cfxql_type`
- `src/batch_ingest_bots.py` — batch-runs markdown chunking across an entire `Bots/` folder (214 files validated)

**CFXQL reference chunking** — configurable via `CHUNKING_STRATEGY` in `ingest_qdrant.py`: `hand_rolled`, `heuristic` (default), or `size_based`

**Legacy path (Chroma)** — `src/ingest.py` / `src/query.py`, original prototype, LLM call not yet wired up,

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY=your_key_here   # needed for query_qdrant.py
```

## Running it

```bash
# sample ingest (3 files in data/raw/)
python src/ingest_qdrant.py

# query with generation
python src/query_qdrant.py "what parameters does the count loop bot take?"

# batch chunk all bot markdown files (does not embed/store yet)
python src/batch_ingest_bots.py /path/to/Bots/

# run eval suite (manual grading)
python tests/run_eval.py
```

## Eval

`tests/eval_set.py` has 7 hand-built cases (lookup, comparison, multi-part, negative/hallucination), `tests/run_eval.py` runs each through `query_qdrant.ask()` for pass/fail grading,

## Open questions

- Swap TF-IDF for a real embedding model (see SWAP POINT in `ingest_qdrant.py`, comparison scripts in `src/test_embedding_comparison.py`)
- Wire `batch_ingest_bots.py` output into the Qdrant ingest path
- Switch CFXQL reference to markdown chunking (`src/test_markdown_chunking.py`)
- Consolidate Chroma vs Qdrant into one path
- Expand `data/raw/` beyond the current 3-file sample
