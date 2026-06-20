
"""
ingest_qdrant.py — chunking logic for the Fabrix docs corpus.

Primary ingest (chunk + embed + store):
    python src/ingest_with_real_embeddings.py

This file owns load_and_chunk_all() and all chunking strategies.
main() below is a legacy TF-IDF-only path — do not use it with
query_qdrant.py, which expects MiniLM embeddings from OpenRouter.
"""

import os
import re
import sys
import pickle
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, os.path.dirname(__file__))
from chunk_heuristic import chunk_by_heuristic_sections
from clean_markdown import clean_markdown, extract_bot_metadata


RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Real bot catalog source - the actual markdown files from the
# confirmed-public Bots/ folder. Set this to your real path.
REAL_BOTS_DIR = "/Users/supersaiyan.06/Downloads/rdaf_docs/rdaf_docs/bot_library/target/docs/Bots"

QDRANT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "qdrant_db")
VECTORIZER_PATH = os.path.join(QDRANT_DIR, "vectorizer.pkl")
COLLECTION_NAME = "fabrix_docs"

# Which chunking strategy to use for the CFXQL reference. Change this
# one line to compare strategies:
#   "hand_rolled" - hardcoded splits, works great here, won't generalize
#   "heuristic"    - generic header detection, works on any doc, noisier
#   "size_based"   - original naive approach, ignores structure
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


def chunk_bot_catalog_markdown(filepath, source_name):
    """Markdown-aware bot catalog chunking: cleans HTML/CSS noise, splits
    on ## headers (one per bot), and extracts bot_name/prefix from each
    header. Generalizes across any bot catalog file without per-file
    customization - tested successfully on cfxdm.md (182 bots), kafka.md
    (2 bots), and jira.md (5 bots)."""
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    with open(filepath, encoding="utf-8") as f:
        raw_text = f.read()

    cleaned_text = clean_markdown(raw_text)

    headers_to_split_on = [("##", "h2")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    raw_chunks = splitter.split_text(cleaned_text)

    chunks = []
    for chunk in raw_chunks:
        h2 = chunk.metadata.get("h2", "")
        if not h2.startswith("Bot "):
            continue  # skip the intro chunk before the first bot

        bot_meta = extract_bot_metadata(h2)

        # determine cfxql_type the same way as the plain-text version
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
    """Wraps the generic heuristic chunker, then adds a best-guess
    cfxql_type tag based on the section header, so metadata filtering
    still works the same way as the other strategies."""
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
        print(f"  {filename}: {len(chunks)} chunks  (strategy={CHUNKING_STRATEGY if filename == 'cfxql_reference.txt' else 'n/a'})")

    # Load the REAL bot catalog (.md files), if the folder exists.
    if os.path.isdir(REAL_BOTS_DIR):
        md_files = sorted(f for f in os.listdir(REAL_BOTS_DIR) if f.endswith(".md"))
        print(f"\nLoading real bot catalog from {REAL_BOTS_DIR} ...")
        for filename in md_files:
            filepath = os.path.join(REAL_BOTS_DIR, filename)
            try:
                chunks = chunk_bot_catalog_markdown(filepath, filename)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"  FAILED on {filename}: {e}")
        print(f"  Loaded {len(md_files)} real bot files")
    else:
        print(f"\n(Real bots directory not found at {REAL_BOTS_DIR}, skipping)")

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
