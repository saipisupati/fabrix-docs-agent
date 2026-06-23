"""
test_faizan_models.py : tests 6 embedding models recommended by Faizan,
across multiple real questions via OpenRouter.

Requires: OPENROUTER_API_KEY environment variable set.
"""

import os
import sys
import requests
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import MarkdownHeaderTextSplitter

EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"

FAIZAN_MODELS = {
    "BAAI/bge-large-en-v1.5": "baai/bge-large-en-v1.5",
    "BAAI/bge-base-en-v1.5": "baai/bge-base-en-v1.5",
    "BAAI/bge-small-en-v1.5": None,
    "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/all-minilm-l6-v2",
    "sentence-transformers/all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": None,
}

TEST_QUESTIONS = [
    {
        "question": "what is the difference between full and restricted cfxql?",
        "expect_headers": {"Full CFXQL", "Restricted CFXQL"},
    },
    {
        "question": "what operators does restricted cfxql support?",
        "expect_headers": {"Restricted CFXQL"},
    },
    {
        "question": "how do i check if a value is null in cfxql?",
        "expect_headers": {"Unary Operators for NULL Value Checks"},
    },
]


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


def test_model(model_name, chunks):
    texts = [c.page_content for c in chunks]
    chunk_vectors = embed_texts(texts, model_name)

    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print('='*70)

    all_ranks = []
    for test in TEST_QUESTIONS:
        question = test["question"]
        expected = test["expect_headers"]

        query_vector = embed_texts([question], model_name)[0]
        sims = cosine_similarity([query_vector], chunk_vectors)[0]
        ranked = sorted(enumerate(sims), key=lambda x: -x[1])

        best_rank = None
        for rank, (idx, score) in enumerate(ranked, 1):
            h2 = chunks[idx].metadata.get("h2", "none")
            h4 = chunks[idx].metadata.get("h4", "")
            label = h4 if h4 else h2
            if label in expected and best_rank is None:
                best_rank = rank

        all_ranks.append(best_rank)
        print(f'  "{question[:50]}..." -> best expected chunk rank: {best_rank}')

    avg_rank = sum(all_ranks) / len(all_ranks)
    print(f"  Average rank across {len(TEST_QUESTIONS)} questions: {avg_rank:.2f}")
    return avg_rank


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_faizan_models.py /path/to/cfxql.md")
        sys.exit(1)

    filepath = sys.argv[1]
    chunks = chunk_markdown_file(filepath)
    print(f"Loaded {len(chunks)} chunks from {filepath}\n")

    print("Checking which of Faizan's requested models are available via OpenRouter:")
    for hf_name, or_name in FAIZAN_MODELS.items():
        status = or_name if or_name else "NOT AVAILABLE on OpenRouter - test via VPN/fastembed instead"
        print(f"  {hf_name} -> {status}")

    results = {}
    for hf_name, or_name in FAIZAN_MODELS.items():
        if or_name is None:
            continue
        try:
            avg_rank = test_model(or_name, chunks)
            results[hf_name] = avg_rank
        except Exception as e:
            print(f"  FAILED on {or_name}: {e}")
            results[hf_name] = None

    print(f"\n{'='*70}")
    print("SUMMARY (lower average rank = better)")
    print('='*70)
    for hf_name, avg_rank in sorted(results.items(), key=lambda x: (x[1] is None, x[1])):
        rank_str = f"{avg_rank:.2f}" if avg_rank is not None else "FAILED"
        print(f"  {hf_name:<55} avg rank: {rank_str}")

    not_tested = [name for name, or_name in FAIZAN_MODELS.items() if or_name is None]
    if not_tested:
        print(f"\nNOT tested (unavailable on OpenRouter, need direct fastembed access):")
        for name in not_tested:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
