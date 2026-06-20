"""Shared paths and model settings. Override via environment variables."""

import os

QDRANT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "qdrant_db")
COLLECTION_NAME = "fabrix_docs"
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-minilm-l6-v2"
)
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"

_DEFAULT_BOTS_DIR = (
    "/Users/supersaiyan.06/Downloads/rdaf_docs/rdaf_docs/bot_library/target/docs/Bots"
)
_DEFAULT_CFXQL_FILE = (
    "/Users/supersaiyan.06/Downloads/rdaf_docs/rdaf_docs/bot_library/target/docs/"
    "reference_guides/cfxql.md"
)

BOTS_DIR = os.environ.get("BOTS_DIR", _DEFAULT_BOTS_DIR)
CFXQL_FILE = os.environ.get("CFXQL_FILE", _DEFAULT_CFXQL_FILE)
