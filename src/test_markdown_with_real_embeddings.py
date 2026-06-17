"""
test_markdown_with_real_embeddings.py — combines markdown-based chunking
with REAL semantic embeddings via OpenRouter, to see if real embeddings
fix the gap we found with TF-IDF.
"""

import os
import sys
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import MarkdownHeaderTextSplitter

EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"


def chunk_markdown_file(filepath):
    with open(filepath, encoding="utf-8") as f:
        text = f.read()
    headers_to_split_on = [("##", "h2"), ("####", "h4")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    return splitter.split_text(text)


def embed_texts(texts, model):
    response = requests.post(
        EMBEDDINGS_URL,
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={"model": model, "input": texts},
    )
    if response.status_code != 200:
        raise RuntimeError(f"Embedding request failed ({response.status_code}): {response.text}")
    data = response.json()["data"]
    data.sort(key=lambda d: d["index"])
    return [d["embedding"] for d in data]


def test_model(model_name, chunks, query):
    print(f"\n{'='*70}")
    print(f"EMBEDDING MODEL: {model_name}")
    print('='*70)

    texts = [c.page_content for c in chunks]

    print("Embedding all chunks...")
    chunk_vectors = embed_texts(texts, model_name)

    print("Embedding query...")
    query_vector = embed_texts([query], model_name)[0]

    sims = cosine_similarity([query_vector], chunk_vectors)[0]

    print(f'\nQuery: "{query}"\n')
    print("All chunks, ranked by score:")
    ranked = sorted(enumerate(sims), key=lambda x: -x[1])
    for idx, score in ranked:
        h2 = chunks[idx].metadata.get("h2", "none")
        h4 = chunks[idx].metadata.get("h4", "")
        print(f"  {idx:2d}: score={score:.3f}  h2={h2:<20} h4={h4}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_markdown_with_real_embeddings.py /path/to/cfxql.md")
        sys.exit(1)

    filepath = sys.argv[1]
    chunks = chunk_markdown_file(filepath)
    print(f"Loaded {len(chunks)} chunks from {filepath}")

    query = "what is the difference between full and restricted cfxql"

    test_model("mistralai/mistral-embed-2312", chunks, query)
    test_model("nvidia/llama-nemotron-embed-vl-1b-v2:free", chunks, query)


if __name__ == "__main__":
    main()
