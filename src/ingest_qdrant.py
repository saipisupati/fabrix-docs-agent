
"""
ingest_qdrant.py — chunk, embed, and store the Fabrix docs corpus in local Qdrant.

    python src/ingest_qdrant.py

Requires: OPENROUTER_API_KEY
"""

import os
import re
import sys
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, os.path.dirname(__file__))
from chunk_heuristic import chunk_by_heuristic_sections
from clean_markdown import clean_markdown, extract_bot_metadata
from config import (
    BOTS_DIR,
    COLLECTION_NAME,
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    EMBEDDINGS_URL,
    QDRANT_DIR,
    QDRANT_UPLOAD_BATCH_SIZE,
)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

CHUNKING_STRATEGY = "hand_rolled"

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
    """Section-aware chunking for the CFXQL reference specifically."""
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


def chunk_bot_catalog_markdown(filepath, source_name):
    """Markdown-aware bot catalog chunking."""
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    with open(filepath, encoding="utf-8") as f:
        raw_text = f.read()

    cleaned_text = clean_markdown(raw_text)
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "h2")])
    raw_chunks = splitter.split_text(cleaned_text)

    chunks = []
    for chunk in raw_chunks:
        h2 = chunk.metadata.get("h2", "")
        if not h2.startswith("Bot "):
            continue

        bot_meta = extract_bot_metadata(h2)
        if "Restricted CFXQL" in chunk.page_content:
            cfxql_type = "Restricted"
        elif "Full CFXQL" in chunk.page_content:
            cfxql_type = "Full"
        else:
            cfxql_type = "unspecified"

        chunks.append({
            "text": chunk.page_content,
            "metadata": {
                "bot_name": bot_meta["bot_name"],
                "prefix": bot_meta["prefix"],
                "cfxql_type": cfxql_type,
                "type": "bot",
                "source": source_name,
            }
        })

    return chunks


def chunk_cfxql_heuristic(text, source_name):
    """Wraps the generic heuristic chunker with cfxql_type tags."""
    raw_chunks = chunk_by_heuristic_sections(text, source_name)
    for chunk in raw_chunks:
        header = chunk["metadata"].get("section_header", "").lower()
        if "restricted" in header or "restricted cfxql" in chunk["text"][:50].lower():
            chunk["metadata"]["cfxql_type"] = "Restricted"
        elif "full" in header or "full cfxql" in chunk["text"][:50].lower():
            chunk["metadata"]["cfxql_type"] = "Full"
        else:
            chunk["metadata"]["cfxql_type"] = "unspecified"
        chunk["metadata"]["bot_name"] = "n/a"
        chunk["metadata"]["prefix"] = "n/a"
    return raw_chunks


def load_and_chunk_all():
    all_chunks = []
    for filename in sorted(os.listdir(RAW_DIR)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(RAW_DIR, filename)) as f:
            text = f.read()

        if filename == "cfxql_reference.txt":
            if CHUNKING_STRATEGY == "hand_rolled":
                chunks = chunk_cfxql_reference(text, filename)
            elif CHUNKING_STRATEGY == "heuristic":
                chunks = chunk_cfxql_heuristic(text, filename)
            else:
                chunks = chunk_narrative(text, filename)
        else:
            chunks = chunk_narrative(text, filename)

        all_chunks.extend(chunks)
        strategy = CHUNKING_STRATEGY if filename == "cfxql_reference.txt" else "n/a"
        print(f"  {filename}: {len(chunks)} chunks  (strategy={strategy})")

    if os.path.isdir(BOTS_DIR):
        md_files = sorted(f for f in os.listdir(BOTS_DIR) if f.endswith(".md"))
        print(f"\nLoading bot catalog from BOTS_DIR={BOTS_DIR} ...")
        for filename in md_files:
            filepath = os.path.join(BOTS_DIR, filename)
            try:
                chunks = chunk_bot_catalog_markdown(filepath, filename)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"  FAILED on {filename}: {e}")
        print(f"  Loaded {len(md_files)} bot files")
    else:
        print(f"\n(BOTS_DIR not found at {BOTS_DIR}, skipping bot catalog)")

    return all_chunks


def embed_batch(texts, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                EMBEDDINGS_URL,
                headers={
                    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                    "Content-Type": "application/json",
                },
                json={"model": EMBEDDING_MODEL, "input": texts},
                timeout=60,
            )
            if response.status_code != 200:
                raise RuntimeError(f"{response.status_code}: {response.text[:200]}")
            data = response.json()["data"]
            data.sort(key=lambda d: d["index"])
            return [d["embedding"] for d in data]
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt + 1}/{max_retries} after error: {e}")
            else:
                raise


def embed_all_chunks(all_chunks):
    texts = [c["text"] for c in all_chunks]
    all_vectors = []
    num_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    for batch_num in range(num_batches):
        start = batch_num * EMBED_BATCH_SIZE
        end = min(start + EMBED_BATCH_SIZE, len(texts))
        batch_texts = texts[start:end]
        print(f"  Embedding batch {batch_num + 1}/{num_batches} ({len(batch_texts)} chunks)...")
        all_vectors.extend(embed_batch(batch_texts))

    return all_vectors


def store_in_qdrant(all_chunks, vectors):
    vector_size = len(vectors[0])
    os.makedirs(QDRANT_DIR, exist_ok=True)

    with open(os.path.join(QDRANT_DIR, "embedding_model.txt"), "w") as f:
        f.write(EMBEDDING_MODEL)

    print(f"\nSetting up Qdrant collection (vector size={vector_size})...")
    client = QdrantClient(path=QDRANT_DIR)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    points = [
        PointStruct(id=i, vector=vec, payload={**chunk["metadata"], "text": chunk["text"]})
        for i, (chunk, vec) in enumerate(zip(all_chunks, vectors))
    ]

    for i in range(0, len(points), QDRANT_UPLOAD_BATCH_SIZE):
        batch = points[i:i + QDRANT_UPLOAD_BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"  Uploaded {min(i + QDRANT_UPLOAD_BATCH_SIZE, len(points))}/{len(points)} points")

    return client.count(COLLECTION_NAME).count


def main():
    if "OPENROUTER_API_KEY" not in os.environ:
        print("Error: OPENROUTER_API_KEY is required.")
        sys.exit(1)

    print("Loading and chunking documents...")
    all_chunks = load_and_chunk_all()
    print(f"Total chunks: {len(all_chunks)}\n")

    if not all_chunks:
        print("No chunks produced — check BOTS_DIR and data/raw/.")
        sys.exit(1)

    print(f"Embedding with {EMBEDDING_MODEL} (batches of {EMBED_BATCH_SIZE})...")
    vectors = embed_all_chunks(all_chunks)
    print(f"Done embedding. Vector size: {len(vectors[0])}")

    count = store_in_qdrant(all_chunks, vectors)
    print(f"\nDone. Stored {count} chunks at {QDRANT_DIR}")


if __name__ == "__main__":
    main()
