"""
config.py, shared paths and env settings for the whole project.

Loads .env on import. Override paths via env vars when you're not on the machine
that wrote the defaults. REMOTE_BASE_URL switches query retrieval to the VPN server.
"""

import os

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
_ENV_PATH = os.path.join(_PROJECT_ROOT, ".env")


def load_dotenv():
    # load .env once; never overrides vars already set in the shell
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
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "fabrix_docs")
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-minilm-l6-v2"
)
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"

EMBED_BATCH_SIZE = 100
QDRANT_UPLOAD_BATCH_SIZE = 200
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
REMOTE_BASE_URL = os.environ.get("REMOTE_BASE_URL", "")



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
# root-level pages (index, Datasets, Formatting-Templates) not under a subfolder
_DEFAULT_DOCS_ROOT_FILES = "index.md,Datasets.md,Formatting-Templates.md"
DOCS_ROOT_FILES = [
    f.strip()
    for f in os.environ.get("DOCS_ROOT_FILES", _DEFAULT_DOCS_ROOT_FILES).split(",")
    if f.strip()
]
