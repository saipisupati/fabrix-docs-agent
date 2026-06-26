"""
ingest.py, LEGACY v0: Chroma + TF-IDF. Do not use for production.

Use ingest_qdrant.py instead. Kept for reference only.
"""

import os
import re
import pickle
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
VECTORIZER_PATH = os.path.join(CHROMA_DIR, "vectorizer.pkl")
COLLECTION_NAME = "fabrix_docs"

# Files that should use bot-catalog chunking (one chunk per bot entry)
# rather than size-based narrative chunking. Update this list as more
# bot catalog pages get added to data/raw/.
BOT_CATALOG_FILES = {"c_extension_loop_bots.txt", "exec_and_dm_sink_bots.txt"}

BOT_START = re.compile(r'(?=Bot [@#*][^\s]+)')
BOT_NAME = re.compile(r'Bot ([@#*][^\s]+)')


def chunk_narrative(text, source_name):
    """Split long-form docs (guides, reference pages) by size with overlap."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = []
    for t in splitter.split_text(text):
        chunks.append({
            "text": t,
            "metadata": {
                "source": source_name,
                "type": "narrative",
                "cfxql_type": "reference",
                "bot_name": "n/a",
                "prefix": "n/a",
            }
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
        chunks.append({
            "text": part,
            "metadata": {
                "bot_name": bot_name,
                "prefix": prefix,
                "cfxql_type": cfxql_type,
                "type": "bot",
                "source": source_name,
            }
        })
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
        else:
            chunks = chunk_narrative(text, filename)

        all_chunks.extend(chunks)
        print(f"  {filename}: {len(chunks)} chunks")

    return all_chunks


def main():
    print("Loading and chunking documents from data/raw/ ...")
    all_chunks = load_and_chunk_all()
    print(f"Total chunks: {len(all_chunks)}\n")

    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]

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

    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print("Storing in Chroma...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    collection.add(documents=texts, embeddings=chunk_vectors, metadatas=metadatas, ids=ids)

    print(f"\nDone. Stored {collection.count()} chunks at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
