"""
query_qdrant.py — retrieve relevant chunks from Qdrant and generate an answer.

Requires:
    OPENROUTER_API_KEY — query embedding (sentence-transformers/all-minilm-l6-v2)
    OPENAI_API_KEY       — generation (gpt-4o-mini)
"""

import os
import sys
import time

import requests
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

sys.path.insert(0, os.path.dirname(__file__))
from config import COLLECTION_NAME, EMBEDDING_MODEL, EMBEDDINGS_URL, LLM_MODEL, QDRANT_DIR

MODEL = LLM_MODEL

def embed_question(question):
    response = requests.post(
        EMBEDDINGS_URL,
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={"model": EMBEDDING_MODEL, "input": [question]},
    )
    return response.json()["data"][0]["embedding"]

def build_filter(filter_dict):
    if not filter_dict:
        return None
    conditions = [
        FieldCondition(key=k, match=MatchValue(value=v))
        for k, v in filter_dict.items()
    ]
    return Filter(must=conditions)


def retrieve(question, client, top_k=3, filter_dict=None):
    query_vector = embed_question(question)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=build_filter(filter_dict),
    )
    chunks = []
    for point in results.points:
        payload = point.payload
        chunks.append({
            "text": payload["text"],
            "metadata": {k: v for k, v in payload.items() if k != "text"},
            "score": point.score,
        })
    return chunks


def build_prompt(question, retrieved_chunks):
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk["metadata"]
        source = meta.get("bot_name") if meta.get("type") == "bot" else meta.get("source")
        context_parts.append(f"[{i}] Source: {source}\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    return f"""You are a helpful assistant for Fabrix.ai documentation.
Answer the user's question using ONLY the documentation excerpts provided below.
If the answer is not in the excerpts, say "I couldn't find that in the documentation."
Always cite which excerpt your answer comes from using [1], [2], etc.

DOCUMENTATION EXCERPTS:
{context}

USER QUESTION:
{question}

ANSWER:"""


def generate(prompt, max_retries=3):
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                print(f"  Rate limited, waiting 30s before retry {attempt + 1}/{max_retries}...")
                time.sleep(30)
            else:
                raise


def ask(question, top_k=3, filter_dict=None):
    client = QdrantClient(path=QDRANT_DIR)

    chunks = retrieve(question, client, top_k=top_k, filter_dict=filter_dict)

    print(f'\nQuestion: "{question}"')
    if filter_dict:
        print(f"Filter: {filter_dict}")
    print(f"\nRetrieved {len(chunks)} chunks:")
    for i, c in enumerate(chunks, 1):
        meta = c["metadata"]
        ref = meta.get("bot_name") if meta.get("type") == "bot" else meta.get("source")
        print(f"  [{i}] score={c['score']:.3f}  {ref}")

    prompt = build_prompt(question, chunks)
    print("\nAsking the model...")
    answer = generate(prompt)

    print(f"\n--- ANSWER ---\n{answer}\n")
    return answer


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        ask("what parameters does the count loop bot take?", filter_dict={"type": "bot"})
        ask("what is the difference between full and restricted cfxql?")
