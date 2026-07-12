#!/usr/bin/env python3
"""
build_kb.py — extract structured KB from public docs, embed, persist under data/kb/.

Run (API should be stopped if you also want Qdrant fabrix_kb upsert):
  python3 src/build_kb.py

Requires BOTS_DIR / DOCS_ROOT / CFXQL_FILE and OPENROUTER_API_KEY for embeddings.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config import EMBEDDING_MODEL, load_dotenv
from kb.extract import build_knowledge_base
from kb.store import (
    embed_entries,
    save_embeddings,
    save_kb,
    upsert_qdrant_kb,
)

load_dotenv()


def main() -> None:
    if "OPENROUTER_API_KEY" not in os.environ:
        print("Error: OPENROUTER_API_KEY is required to embed KB entries.")
        sys.exit(1)

    print("Extracting structured knowledge base from public docs...")
    kb = build_knowledge_base()
    kb.embedding_model = EMBEDDING_MODEL
    print(
        f"  topics={len(kb.topics)} entities={len(kb.entities)} "
        f"facts={len(kb.facts)} procedures={len(kb.procedures)} relations={len(kb.relations)}"
    )
    if not kb.entities and not kb.facts:
        print("Error: KB is empty. Check BOTS_DIR / DOCS_ROOT / CFXQL_FILE in .env")
        sys.exit(1)

    save_kb(kb)
    print("  Wrote data/kb/kb.json")

    entries = kb.searchable_entries()
    print(f"Embedding {len(entries)} searchable KB entries with {EMBEDDING_MODEL}...")
    matrix = embed_entries(entries, EMBEDDING_MODEL)
    save_embeddings(entries, matrix)
    print("  Wrote data/kb/embeddings.npz")

    print("Attempting Qdrant collection fabrix_kb upsert...")
    ok = upsert_qdrant_kb(entries, matrix, EMBEDDING_MODEL)
    print("  Qdrant KB:", "ok" if ok else "skipped (use embeddings.npz at query time)")

    print("Done.")


if __name__ == "__main__":
    main()
