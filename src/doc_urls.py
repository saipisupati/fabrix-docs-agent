"""
doc_urls.py, turn chunk metadata into clickable docs.fabrix.ai links.

Used by the agent and API so answers cite the real public doc pages.
"""

from urllib.parse import quote

PUBLIC_DOCS_BASE = "https://docs.fabrix.ai"
CFXQL_REL_PATH = "reference_guides/cfxql.md"
BOTS_REL_PREFIX = "Bots"


def public_doc_url(rel_path):
    # repo path like beginners_guide/foo.md → https://docs.fabrix.ai/beginners_guide/foo/
    path = rel_path.replace("\\", "/")
    if path.endswith(".md"):
        path = path[:-3]
    parts = path.split("/")
    if parts[-1] in ("index", "index_release"):
        parts = parts[:-1]
    encoded = "/".join(quote(part, safe="") for part in parts if part)
    if not encoded:
        return f"{PUBLIC_DOCS_BASE}/"
    return f"{PUBLIC_DOCS_BASE}/{encoded}/"


def chunk_metadata_to_url(metadata):
    # bots, cfxql, and narrative guides each map slightly differently
    source = (metadata.get("source") or "").replace("\\", "/")
    chunk_type = metadata.get("type", "")

    if source == "cfxql.md" or metadata.get("cfxql_type") in (
        "Full",
        "Restricted",
        "intro",
    ):
        return public_doc_url(CFXQL_REL_PATH)

    if chunk_type == "bot":
        return public_doc_url(f"{BOTS_REL_PREFIX}/{source}")

    if "/" in source:
        return public_doc_url(source)

    if source.endswith(".md"):
        return public_doc_url(f"{BOTS_REL_PREFIX}/{source}")

    return public_doc_url(source)
