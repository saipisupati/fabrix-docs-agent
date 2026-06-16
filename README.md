# Fabrix Docs Agent

A retrieval-augmented generation (RAG) agent that answers questions about
Fabrix.ai documentation (https://docs.fabrix.ai/) — covering the RDA bot
catalog, CFXQL reference, and platform guides.

## Status

Early prototype. Pipeline (chunk → embed → store → retrieve → generate) is
working end-to-end on a small sample of docs. Not yet connected to a
production LLM provider.

## How it works

1. **Ingest** (`src/ingest.py`) — loads raw doc files from `data/raw/`,
   splits them into chunks (different strategy for narrative guides vs.
   the templated bot catalog), embeds each chunk, and stores them in a
   local Chroma vector store at `data/chroma_db/`.
2. **Query** (`src/query.py`) — takes a user question, embeds it, retrieves
   the most relevant chunks from Chroma (optionally filtered by metadata
   like `cfxql_type` or `type`), builds a prompt, and generates an answer
   grounded in the retrieved chunks.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set your model API key as an environment variable (provider TBD — see
`src/query.py` for current config).

## Running it

```bash
python src/ingest.py     # builds the vector store from data/raw/
python src/query.py "what parameters does the count loop bot take?"
```

## Open questions

- Final model/provider (currently scaffolded for swap-in, not yet confirmed)
- Framework choice (LangChain currently used for chunking utilities)
- Source of docs: public site (docs.fabrix.ai) vs. internal export
