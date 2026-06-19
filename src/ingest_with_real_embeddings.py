"""
ingest_with_real_embeddings.py —> replaces TF-IDF with a real embedding
model (sentence-transformers/all-MiniLM-L6-v2) across the FULL real dataset (1867 chunks).

Requires: OPENROUTER_API_KEY environment variable set.
"""

import os
import sys
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

sys.path.insert(0, os.path.dirname(__file__))
from ingest_qdrant import load_and_chunk_all, QDRANT_DIR, COLLECTION_NAME

EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
MODEL = "sentence-transformers/all-minilm-l6-v2"
BATCH_SIZE = 100


def embed_batch(texts, model=MODEL, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                EMBEDDINGS_URL,
                headers={
                    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                    "Content-Type": "application/json",
                },
                json={"model": model, "input": texts},
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

    num_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_num in range(num_batches):
        start = batch_num * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(texts))
        batch_texts = texts[start:end]

        print(f"  Embedding batch {batch_num + 1}/{num_batches} ({len(batch_texts)} chunks)...")
        batch_vectors = embed_batch(batch_texts)
        all_vectors.extend(batch_vectors)

    return all_vectors


def main():
    print("Loading and chunking documents (sample files + real bot catalog)...")
    all_chunks = load_and_chunk_all()
    print(f"Total chunks to embed: {len(all_chunks)}\n")

    print(f"Embedding all chunks with {MODEL} (in batches of {BATCH_SIZE})...")
    vectors = embed_all_chunks(all_chunks)
    vector_size = len(vectors[0])
    print(f"\nDone embedding. Vector size: {vector_size}")

    with open(os.path.join(QDRANT_DIR, "embedding_model.txt"), "w") as f:
        f.write(MODEL)

    print(f"\nSetting up Qdrant collection (vector size={vector_size})...")
    client = QdrantClient(path=QDRANT_DIR)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    points = []
    for i, (chunk, vec) in enumerate(zip(all_chunks, vectors)):
        payload = {**chunk["metadata"], "text": chunk["text"]}
        points.append(PointStruct(id=i, vector=vec, payload=payload))

    upload_batch_size = 200
    for i in range(0, len(points), upload_batch_size):
        batch = points[i:i + upload_batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"  Uploaded {min(i + upload_batch_size, len(points))}/{len(points)} points to Qdrant")

    count = client.count(COLLECTION_NAME).count
    print(f"\nDone. Stored {count} chunks with real embeddings ({MODEL}) at {QDRANT_DIR}")


if __name__ == "__main__":
    main()
