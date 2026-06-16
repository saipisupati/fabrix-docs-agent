"""
ingest_qdrant.py — builds the Qdrant vector store from raw documentation files.

Run this once (and again any time data/raw/ changes):
    python src/ingest_qdrant.py

What it does:
  1. Loads every .txt file in data/raw/
  2. Chunks each file based on its type:
     - bot catalog files -> one chunk per bot entry
     - cfxql_reference.txt -> section-aware chunking (Full vs Restricted,
       split at the doc's own headers, not by raw character count)
     - anything else -> generic size-based chunking with overlap
  3. Embeds every chunk
  4. Stores chunks + embeddings + metadata in a local Qdrant collection
     at data/qdrant_db/

NOTE ON EMBEDDINGS: this currently uses TfidfVectorizer as a placeholder
embedding method (no API key required, runs fully offline). Swap in a real
embedding model once the provider is confirmed - see the comment marked
SWAP POINT below. The rest of the pipeline does not need to change.
"""

import os
import re
import pickle
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
QDRANT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "qdrant_db")
VECTORIZER_PATH = os.path.join(QDRANT_DIR, "vectorizer.pkl")
COLLECTION_NAME = "fabrix_docs"

# Files that should use bot-catalog chunking (one chunk per bot entry)
# rather than size-based or section-based chunking. Update this list as
# more bot catalog pages get added to data/raw/.
BOT_CATALOG_FILES = {"c_extension_loop_bots.txt", "exec_and_dm_sink_bots.txt"}

BOT_START = re.compile(r'(?=Bot [@#*][^\s]+)')
BOT_NAME = re.compile(r'Bot ([@#*][^\s]+)')


def chunk_narrative(text, source_name):
    """Generic fallback: split long-form docs by size with overlap.
    Used for any narrative doc that isn't the CFXQL reference."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return [
        {"text": t, "metadata": {"source": source_name, "type": "narrative",
         "cfxql_type": "reference", "bot_name": "n/a", "prefix": "n/a"}}
        for t in splitter.split_text(text)
    ]


def chunk_cfxql_reference(text, source_name):
    """Section-aware chunking for the CFXQL reference specifically.
    Splits at the doc's own Full CFXQL / Restricted CFXQL headers instead
    of raw character count, so cfxql_type metadata is accurate per chunk
    and Full/Restricted content never mix in the same chunk."""
    intro_and_bot_types, rest = text.split("Full CFXQL\n", 1)
    full_section, restricted_section = rest.split("Restricted CFXQL\n", 1)
    full_section = "Full CFXQL\n" + full_section
    restricted_section = "Restricted CFXQL\n" + restricted_section

    chunks = [{
        "text": intro_and_bot_types.strip(),
        "metadata": {"source": source_name, "type": "narrative",
                     "cfxql_type": "intro", "bot_name": "n/a", "prefix": "n/a"}
    }]

    full_parts = full_section.split("Equality Operators\n", 1)
    chunks.append({
        "text": full_parts[0].strip(),
        "metadata": {"source": source_name, "type": "narrative",
                     "cfxql_type": "Full", "bot_name": "n/a", "prefix": "n/a"}
    })
    if len(full_parts) > 1:
        chunks.append({
            "text": ("Full CFXQL Equality Operators\n" + full_parts[1]).strip(),
            "metadata": {"source": source_name, "type": "narrative",
                         "cfxql_type": "Full", "bot_name": "n/a", "prefix": "n/a"}
        })

    chunks.append({
        "text": restricted_section.strip(),
        "metadata": {"source": source_name, "type": "narrative",
                     "cfxql_type": "Restricted", "bot_name": "n/a", "prefix": "n/a"}
    })
    return chunks


def chunk_bot_catalog(text, source_name):
    """Split bot catalog pages into one chunk per bot entry, extracting metadata
    (bot name, prefix, CFXQL type) directly from the templated structure."""
    chunks = []
    for part in BOT_START.split(text):
        part = part.strip()
        if not part:
            continue
        name_match = BOT_NAME.search(part)
        bot_name = name_match.group(1) if name_match else "unknown"
        prefix = bot_name[0] if bot_name[0] in "@#*" else "unknown"
        if "Restricted CFXQL" in part:
            cfxql_type = "Restricted"
        elif "Full CFXQL" in part:
            cfxql_type = "Full"
        else:
            cfxql_type = "unspecified"
        chunks.append({"text": part, "metadata": {
            "bot_name": bot_name, "prefix": prefix, "cfxql_type": cfxql_type,
            "type": "bot", "source": source_name
        }})
    return chunks


def load_and_chunk_all():
    all_chunks = []
    for filename in sorted(os.listdir(RAW_DIR)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(RAW_DIR, filename)
        with open(path) as f:
            text = f.read()

        if filename in BOT_CATALOG_FILES:
            chunks = chunk_bot_catalog(text, filename)
        elif filename == "cfxql_reference.txt":
            chunks = chunk_cfxql_reference(text, filename)  # section-aware
        else:
            chunks = chunk_narrative(text, filename)  # generic fallback

        all_chunks.extend(chunks)
        print(f"  {filename}: {len(chunks)} chunks")

    return all_chunks


def main():
    print("Loading and chunking documents from data/raw/ ...")
    all_chunks = load_and_chunk_all()
    print(f"Total chunks: {len(all_chunks)}\n")

    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    print("Embedding chunks...")
    # ============================================================
    # SWAP POINT: replace this block with a real embedding model
    # once the provider is confirmed, e.g.:
    #
    #   from langchain_mistralai import MistralAIEmbeddings
    #   embedder = MistralAIEmbeddings(model="mistral-embed")
    #   chunk_vectors = embedder.embed_documents(texts)
    #
    # Nothing else in this file needs to change.
    # ============================================================
    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(texts).toarray().tolist()
    vector_size = len(chunk_vectors[0])

    os.makedirs(QDRANT_DIR, exist_ok=True)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Setting up Qdrant collection (vector size={vector_size})...")
    client = QdrantClient(path=QDRANT_DIR)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    # Qdrant stores chunks as "points": an id, a vector, and a payload
    # (payload = metadata + the original text, since Qdrant doesn't
    # have a separate "documents" field the way Chroma does)
    points = []
    for i, (vec, meta, text) in enumerate(zip(chunk_vectors, metadatas, texts)):
        payload = {**meta, "text": text}
        points.append(PointStruct(id=i, vector=vec, payload=payload))

    client.upsert(collection_name=COLLECTION_NAME, points=points)

    count = client.count(COLLECTION_NAME).count
    print(f"\nDone. Stored {count} chunks at {QDRANT_DIR}")


if __name__ == "__main__":
    main()
