
"""
ingest_qdrant.py —-> chunk, embed, and store the Fabrix docs corpus in local Qdrant.

    python src/ingest_qdrant.py

Requires: OPENROUTER_API_KEY
"""

import os
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
    CFXQL_FILE,
    COLLECTION_NAME,
    DOCS_INCLUDE_DIRS,
    DOCS_ROOT,
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    EMBEDDINGS_URL,
    QDRANT_DIR,
    QDRANT_UPLOAD_BATCH_SIZE,
)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

CHUNKING_STRATEGY = "hand_rolled"
MAX_CHUNK_CHARS = 8000
MAX_EMBED_BATCH_CHARS = 120_000

# Legacy sample bot pages in data/raw/ — superseded by BOTS_DIR markdown catalog
SKIP_RAW_FILES = {"c_extension_loop_bots.txt", "exec_and_dm_sink_bots.txt"}


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


def chunk_cfxql_markdown(filepath):
    """Markdown-aware chunking for the real CFXQL reference (cfxql.md)."""
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    cleaned = clean_markdown(text)
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "h2"), ("####", "h4")]
    )
    source_name = os.path.basename(filepath)
    chunks = []

    for chunk in splitter.split_text(cleaned):
        h2 = chunk.metadata.get("h2", "").lower()
        h4 = chunk.metadata.get("h4", "").lower()
        if "restricted" in h2 or "restricted" in h4:
            cfxql_type = "Restricted"
        elif "full" in h2 or "full" in h4:
            cfxql_type = "Full"
        elif h2 or h4:
            cfxql_type = "intro"
        else:
            cfxql_type = "unspecified"

        chunks.append({
            "text": chunk.page_content,
            "metadata": {
                "source": source_name,
                "type": "narrative",
                "cfxql_type": cfxql_type,
                "bot_name": "n/a",
                "prefix": "n/a",
                **chunk.metadata,
            },
        })

    return chunks


def chunk_narrative_markdown(filepath, rel_source, doc_section):
    """Markdown-aware chunking for platform/narrative docs (non-bot)."""
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    cleaned = clean_markdown(text)
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "h2"), ("###", "h3"), ("####", "h4")]
    )
    chunks = []
    for chunk in splitter.split_text(cleaned):
        if not chunk.page_content.strip():
            continue
        chunks.append({
            "text": chunk.page_content,
            "metadata": {
                "type": "narrative",
                "source": rel_source,
                "doc_section": doc_section,
                "cfxql_type": "n/a",
                "bot_name": "n/a",
                "prefix": "n/a",
                **chunk.metadata,
            },
        })
    return chunks


def _cfxql_rel_path():
    if os.path.isfile(CFXQL_FILE):
        return os.path.relpath(CFXQL_FILE, DOCS_ROOT).replace("\\", "/")
    return "reference_guides/cfxql.md"


def load_narrative_docs():
    """Walk DOCS_INCLUDE_DIRS under DOCS_ROOT; skip cfxql.md (ingested separately)."""
    all_chunks = []
    skip_rel = {_cfxql_rel_path()}
    print(f"\nLoading narrative docs from DOCS_ROOT={DOCS_ROOT}")
    print(f"  include: {DOCS_INCLUDE_DIRS}")

    for subdir in DOCS_INCLUDE_DIRS:
        dir_path = os.path.join(DOCS_ROOT, subdir)
        if not os.path.isdir(dir_path):
            print(f"  {subdir}: skipped (not found)")
            continue

        folder_files = 0
        folder_chunks = 0
        errors = []
        for root, _, files in os.walk(dir_path):
            for filename in sorted(files):
                if not filename.endswith(".md"):
                    continue
                filepath = os.path.join(root, filename)
                rel_source = os.path.relpath(filepath, DOCS_ROOT).replace("\\", "/")
                if rel_source in skip_rel:
                    continue
                doc_section = rel_source.split("/")[0]
                try:
                    chunks = chunk_narrative_markdown(filepath, rel_source, doc_section)
                    all_chunks.extend(chunks)
                    folder_files += 1
                    folder_chunks += len(chunks)
                except Exception as e:
                    errors.append((rel_source, str(e)))
        print(f"  {subdir}: {folder_files} files, {folder_chunks} chunks", end="")
        if errors:
            print(f", {len(errors)} errors")
            for rel, err in errors:
                print(f"    FAILED {rel}: {err}")
        else:
            print()

    return all_chunks


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
    cfxql_loaded = False

    if os.path.isfile(CFXQL_FILE):
        chunks = chunk_cfxql_markdown(CFXQL_FILE)
        all_chunks.extend(chunks)
        print(f"  {os.path.basename(CFXQL_FILE)}: {len(chunks)} chunks  (strategy=markdown)")
        cfxql_loaded = True
    else:
        print(f"  (CFXQL_FILE not found at {CFXQL_FILE}, using data/raw/ fallback)")

    for filename in sorted(os.listdir(RAW_DIR)):
        if not filename.endswith(".txt"):
            continue
        if filename in SKIP_RAW_FILES:
            print(f"  {filename}: skipped (superseded by BOTS_DIR markdown catalog)")
            continue
        if filename == "cfxql_reference.txt" and cfxql_loaded:
            print(f"  {filename}: skipped (using CFXQL_FILE markdown)")
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

    if DOCS_INCLUDE_DIRS and os.path.isdir(DOCS_ROOT):
        all_chunks.extend(load_narrative_docs())
    elif DOCS_INCLUDE_DIRS:
        print(f"\n(DOCS_ROOT not found at {DOCS_ROOT}, skipping narrative docs)")

    return all_chunks


def split_oversized_chunks(chunks):
    """Split any chunk exceeding MAX_CHUNK_CHARS before embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_CHARS, chunk_overlap=100
    )
    normalized = []
    for chunk in chunks:
        if len(chunk["text"]) <= MAX_CHUNK_CHARS:
            normalized.append(chunk)
            continue
        for part in splitter.split_text(chunk["text"]):
            normalized.append({"text": part, "metadata": dict(chunk["metadata"])})
    oversized = sum(1 for c in chunks if len(c["text"]) > MAX_CHUNK_CHARS)
    if oversized:
        print(f"  Split {oversized} oversized chunks (>{MAX_CHUNK_CHARS} chars)")
    return normalized


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
            body = response.json()
            if "data" not in body:
                raise RuntimeError(f"Unexpected response: {str(body)[:200]}")
            data = body["data"]
            data.sort(key=lambda d: d["index"])
            return [d["embedding"] for d in data]
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt + 1}/{max_retries} after error: {e}")
            else:
                raise


def embed_all_chunks(all_chunks):
    texts = [c["text"] for c in all_chunks]
    batches = list(_char_limited_batches(texts, MAX_EMBED_BATCH_CHARS, EMBED_BATCH_SIZE))
    all_vectors = []
    for batch_num, batch_texts in enumerate(batches, 1):
        print(f"  Embedding batch {batch_num}/{len(batches)} ({len(batch_texts)} chunks)...")
        all_vectors.extend(embed_batch(batch_texts))
    return all_vectors


def _char_limited_batches(texts, max_chars, max_items):
    """Yield batches capped by total characters and item count."""
    batch = []
    batch_chars = 0
    for text in texts:
        if batch and (batch_chars + len(text) > max_chars or len(batch) >= max_items):
            yield batch
            batch = []
            batch_chars = 0
        batch.append(text)
        batch_chars += len(text)
    if batch:
        yield batch


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
    all_chunks = split_oversized_chunks(load_and_chunk_all())
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
