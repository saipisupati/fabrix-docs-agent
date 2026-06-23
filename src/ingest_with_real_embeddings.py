"""Deprecated: use ingest_qdrant.py instead."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

print("Note: ingest_with_real_embeddings.py is deprecated, use ingest_qdrant.py instead.\n")

from ingest_qdrant import main

if __name__ == "__main__":
    main()
