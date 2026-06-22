"""Shared paths and model settings. Override via environment variables or .env file."""

import os

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
_ENV_PATH = os.path.join(_PROJECT_ROOT, ".env")


def load_dotenv():
    """Load project-root .env into os.environ (does not override existing vars)."""
    if not os.path.isfile(_ENV_PATH):
        return
    with open(_ENV_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()

QDRANT_DIR = os.path.join(_PROJECT_ROOT, "data", "qdrant_db")
COLLECTION_NAME = "fabrix_docs"
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-minilm-l6-v2"
)
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"

EMBED_BATCH_SIZE = 100
QDRANT_UPLOAD_BATCH_SIZE = 200

_DEFAULT_BOTS_DIR = (
    "/Users/supersaiyan.06/Downloads/rdaf_docs/rdaf_docs/bot_library/target/docs/Bots"
)
_DEFAULT_DOCS_ROOT = os.path.dirname(_DEFAULT_BOTS_DIR)
_DEFAULT_CFXQL_FILE = (
    "/Users/supersaiyan.06/Downloads/rdaf_docs/rdaf_docs/bot_library/target/docs/"
    "reference_guides/cfxql.md"
)
_DEFAULT_DOCS_INCLUDE_DIRS = (
    "beginners_guide,reference_guides,installation_guides,Pipelines,"
    "Datasource_Integrations,ai_fabric,Extensions,rda_releases"
)

BOTS_DIR = os.environ.get("BOTS_DIR", _DEFAULT_BOTS_DIR)
DOCS_ROOT = os.environ.get("DOCS_ROOT", _DEFAULT_DOCS_ROOT)
CFXQL_FILE = os.environ.get("CFXQL_FILE", _DEFAULT_CFXQL_FILE)
DOCS_INCLUDE_DIRS = [
    d.strip()
    for d in os.environ.get("DOCS_INCLUDE_DIRS", _DEFAULT_DOCS_INCLUDE_DIRS).split(",")
    if d.strip()
]
