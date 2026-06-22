"""
test_fastembed_eval.py: compare fastembed embedding models on eval_set retrieval.

Embeds the full local doc set in memory (no Qdrant re-ingest per model) and scores
each eval case using the same filters/top_k as run_eval_baseline.py.

Usage:
    pip install fastembed
    python3 src/test_fastembed_eval.py

Optional:
    python3 src/test_fastembed_eval.py --models BAAI/bge-small-en-v1.5 BAAI/bge-base-en-v1.5
"""

import argparse
import os
import sys
import time

import numpy as np
from fastembed import TextEmbedding
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from eval_scoring import grade_retrieval, score_retrieval
from eval_set import EVAL_SET, retrieval_params
from ingest_qdrant import load_and_chunk_all, split_oversized_chunks
from query_qdrant import bot_name_hints, rerank_by_bot_name

DEFAULT_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "snowflake/snowflake-arctic-embed-s",
    "mixedbread-ai/mxbai-embed-large-v1",
    "BAAI/bge-large-en-v1.5",
]

EMBED_BATCH = 256


def chunk_matches_filter(metadata, filter_dict):
    if not filter_dict:
        return True
    return all(metadata.get(k) == v for k, v in filter_dict.items())


def embed_corpus(embedder, texts):
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        vectors.extend(list(embedder.embed(batch)))
        done = min(start + EMBED_BATCH, len(texts))
        if done % 1024 < EMBED_BATCH or done == len(texts):
            print(f"    embedded {done}/{len(texts)} chunks", flush=True)
    return np.array(vectors, dtype=np.float32)


def embed_query(embedder, question):
    return np.array(list(embedder.embed([question]))[0], dtype=np.float32)


def in_memory_retrieve(question, query_vector, chunk_vectors, chunks, top_k, filter_dict):
    hints = bot_name_hints(question)
    candidate_limit = max(top_k * 10, 300) if hints else top_k

    indices = [
        i for i, c in enumerate(chunks)
        if chunk_matches_filter(c["metadata"], filter_dict)
    ]
    if not indices:
        return []

    sub_vectors = chunk_vectors[indices]
    sims = cosine_similarity([query_vector], sub_vectors)[0]
    ranked = sorted(zip(indices, sims), key=lambda x: -x[1])[:candidate_limit]

    results = [
        {
            "text": chunks[i]["text"],
            "metadata": chunks[i]["metadata"],
            "score": float(score),
        }
        for i, score in ranked
    ]
    return rerank_by_bot_name(question, results)[:top_k]


def eval_model(embedder, chunks, chunk_vectors, cases):
    grades = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
    details = []

    for case in cases:
        params = retrieval_params(case)
        q_vec = embed_query(embedder, case["question"])
        retrieved = in_memory_retrieve(
            case["question"],
            q_vec,
            chunk_vectors,
            chunks,
            top_k=params["top_k"],
            filter_dict=params["filter_dict"],
        )
        score = score_retrieval(case, retrieved)
        grade = grade_retrieval(score)
        grades[grade] += 1
        details.append((case["id"], grade, score.get("source_hit"), score.get("fact_score")))

    return grades, details


def main():
    parser = argparse.ArgumentParser(description="Compare fastembed models on eval retrieval")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="fastembed model names to compare",
    )
    args = parser.parse_args()

    print("Loading and chunking corpus...")
    chunks = split_oversized_chunks(load_and_chunk_all())
    texts = [c["text"] for c in chunks]
    print(f"  {len(chunks)} chunks\n")

    cases = [c for c in EVAL_SET if c["category"] != "negative"]
    print(f"Scoring {len(cases)} retrieval cases (negative cases skipped)\n")

    summary = []
    for model_name in args.models:
        print(f"{'=' * 72}")
        print(f"MODEL: {model_name}")
        print("=" * 72)
        t0 = time.time()
        try:
            print("  Loading model...")
            embedder = TextEmbedding(model_name=model_name)
            print("  Embedding corpus...")
            chunk_vectors = embed_corpus(embedder, texts)
            grades, details = eval_model(embedder, chunks, chunk_vectors, cases)
            elapsed = time.time() - t0
            dim = chunk_vectors.shape[1]
            pass_count = grades["PASS"]
            print(f"\n  dim={dim}  PASS={pass_count}/{len(cases)}  "
                  f"PARTIAL={grades['PARTIAL']}  FAIL={grades['FAIL']}  "
                  f"({elapsed:.0f}s)")
            for case_id, grade, source_hit, fact_score in details:
                fs = f"{fact_score:.0%}" if fact_score is not None else "n/a"
                print(f"    [{case_id}] {grade}  source_hit={source_hit}  facts={fs}")
            summary.append({
                "model": model_name,
                "dim": dim,
                "pass": pass_count,
                "partial": grades["PARTIAL"],
                "fail": grades["FAIL"],
                "seconds": elapsed,
                "error": None,
            })
        except Exception as e:
            print(f"  FAILED: {e}")
            summary.append({
                "model": model_name,
                "dim": None,
                "pass": None,
                "partial": None,
                "fail": None,
                "seconds": time.time() - t0,
                "error": str(e),
            })

    print(f"\n{'=' * 72}")
    print("SUMMARY (retrieval PASS count / 8 scored cases)")
    print("=" * 72)
    print(f"{'Model':<45} {'dim':>5} {'PASS':>6} {'time':>8}")
    print("-" * 72)
    for row in sorted(summary, key=lambda r: (-1 if r["pass"] is None else -r["pass"], r["model"])):
        if row["error"]:
            print(f"{row['model']:<45} {'—':>5} {'ERR':>6} {row['seconds']:>7.0f}s")
        else:
            print(f"{row['model']:<45} {row['dim']:>5} {row['pass']:>6} {row['seconds']:>7.0f}s")
    print("\nBaseline (OpenRouter MiniLM + Qdrant): 8/8 PASS")


if __name__ == "__main__":
    main()
