"""
test_embedding_comparison.py: Phase 2: compares several embedding models
side by side on both ACCURACY (does the right chunk rank highly on our
regression question) and COST (real OpenRouter pricing), producing a
clean comparison table.
"""

import os
import sys
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import MarkdownHeaderTextSplitter

EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"

MODELS_TO_TEST = [
    {"id": "mistralai/mistral-embed-2312", "price_per_token": 0.0000001},
    {"id": "nvidia/llama-nemotron-embed-vl-1b-v2:free", "price_per_token": 0.0},
    {"id": "baai/bge-large-en-v1.5", "price_per_token": 0.00000001},
    {"id": "openai/text-embedding-3-small", "price_per_token": 0.00000002},
]

CORRECT_HEADERS = {"Full CFXQL", "Restricted CFXQL"}


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


def score_accuracy(chunks, sims):
    ranked = sorted(enumerate(sims), key=lambda x: -x[1])
    ranks = {}
    for rank, (idx, score) in enumerate(ranked, 1):
        h2 = chunks[idx].metadata.get("h2", "")
        h4 = chunks[idx].metadata.get("h4", "")
        if h2 in CORRECT_HEADERS and not h4:
            ranks[h2] = rank
    return ranks


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_embedding_comparison.py /path/to/cfxql.md")
        sys.exit(1)

    filepath = sys.argv[1]
    chunks = chunk_markdown_file(filepath)
    query = "what is the difference between full and restricted cfxql"

    results = []

    for model_info in MODELS_TO_TEST:
        model_id = model_info["id"]
        print(f"Testing {model_id}...")
        try:
            texts = [c.page_content for c in chunks]
            chunk_vectors = embed_texts(texts, model_id)
            query_vector = embed_texts([query], model_id)[0]
            sims = cosine_similarity([query_vector], chunk_vectors)[0]

            ranks = score_accuracy(chunks, sims)
            full_rank = ranks.get("Full CFXQL", "N/A")
            restricted_rank = ranks.get("Restricted CFXQL", "N/A")

            results.append({
                "model": model_id,
                "price_per_million": model_info["price_per_token"] * 1_000_000,
                "full_cfxql_rank": full_rank,
                "restricted_cfxql_rank": restricted_rank,
            })
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                "model": model_id,
                "price_per_million": model_info["price_per_token"] * 1_000_000,
                "full_cfxql_rank": "ERROR",
                "restricted_cfxql_rank": "ERROR",
            })

    print(f"\n{'='*90}")
    print("COMPARISON TABLE")
    print('='*90)
    print(f"{'Model':<45} {'$/1M tokens':>12} {'Full rank':>10} {'Restricted rank':>16}")
    print("-" * 90)
    for r in results:
        print(f"{r['model']:<45} {r['price_per_million']:>12.4f} {str(r['full_cfxql_rank']):>10} {str(r['restricted_cfxql_rank']):>16}")
    print("\n(Lower rank number = better. Rank 1-2 means the correct chunk")
    print("was the top result. N/A means that header wasn't found at all.)")


if __name__ == "__main__":
    main()
