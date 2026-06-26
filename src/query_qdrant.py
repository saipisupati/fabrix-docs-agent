"""
query_qdrant.py: retrieve relevant chunks from Qdrant and generate an answer.

Requires:
    OPENROUTER_API_KEY: query embedding (sentence-transformers/all-minilm-l6-v2)
    OPENAI_API_KEY      : generation (gpt-4o-mini)
"""

import os
import re
import sys
import time

import requests
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDINGS_URL,
    LLM_MODEL,
    QDRANT_DIR,
    REMOTE_BASE_URL,
)

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


def bot_name_hints(question):
    """Extract bot-name slugs from natural-language questions."""
    q = question.lower()
    hints = re.findall(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)+", q)
    for match in re.finditer(r"(\w+)\s+loop\b", q):
        hints.append(f"{match.group(1)}-loop")
    for match in re.finditer(r"(\w+)\s+condition\b", q):
        hints.append(f"{match.group(1)}-condition")
    return list(dict.fromkeys(hints))


def rerank_by_bot_name(question, chunks):
    hints = bot_name_hints(question)
    if not hints:
        return chunks

    def sort_key(chunk):
        name = chunk["metadata"].get("bot_name", "").lower()
        match = any(hint in name for hint in hints)
        return (match, chunk["score"])

    return sorted(chunks, key=sort_key, reverse=True)


def prune_lookup_chunks(question, chunks, category):
    """For lookup questions, drop sibling bots that share a slug with rank-1."""
    if category != "lookup" or not chunks:
        return chunks
    hints = bot_name_hints(question)
    if not hints:
        return chunks[:2]
    top_name = chunks[0]["metadata"].get("bot_name", "").lower()
    if not all(h in top_name for h in hints):
        return chunks[:2]
    return [c for c in chunks if c["metadata"].get("bot_name", "").lower() == top_name]


def retrieve_remote(question, top_k=5, filter_dict=None):
    response = requests.post(
        f"{REMOTE_BASE_URL}/search",
        headers={"Content-Type": "application/json"},
        json={"collection_name": COLLECTION_NAME, "query": question, "limit": top_k},
        timeout=30,
    )
    if not response.ok:
        return []
    results = response.json().get("results", [])
    chunks = []
    for r in results:
        meta = r.get("metadata", {})
        chunks.append({
            "text": r.get("text", ""),
            "metadata": {
                "source": meta.get("source_file", ""),
                "bot_name": meta.get("heading", ""),
                "type": meta.get("type", ""),
                "doc_section": meta.get("doc_section", ""),
            },
            "score": r.get("score", 0),
        })
    return chunks


def retrieve(question, client, top_k=3, filter_dict=None):
    if REMOTE_BASE_URL:
        hints = bot_name_hints(question)
        remote_limit = max(top_k * 10, 300) if hints else top_k
        chunks = retrieve_remote(question, top_k=remote_limit, filter_dict=filter_dict)
        return rerank_by_bot_name(question, chunks)[:top_k]

    hints = bot_name_hints(question)
    candidate_limit = max(top_k * 10, 300) if hints else top_k

    query_vector = embed_question(question)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=candidate_limit,
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
    return rerank_by_bot_name(question, chunks)[:top_k]


def build_prompt(question, retrieved_chunks, category=None):
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk["metadata"]
        source = meta.get("bot_name") if meta.get("type") == "bot" else meta.get("source")
        context_parts.append(f"[{i}] Source: {source}\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    extra_instructions = ""
    if category == "lookup":
        extra_instructions = """
- Excerpt [1] is the primary source; answer about that specific bot.
- List parameters from that bot's parameter table only; do not mix tables from other bots.
- Include all parameters from the table relevant to the question and any required companion parameters shown in the excerpt.
"""
    elif category == "negative":
        extra_instructions = """
- Only answer if an excerpt explicitly documents the exact procedure or fact asked about.
- If excerpts are only tangentially related (e.g. admin setup vs end-user password reset), say "I couldn't find that in the documentation."
"""
    elif category == "multi_part":
        extra_instructions = """
- The question has multiple parts; address each part explicitly.
- Use all relevant excerpts; cite every excerpt you draw from.
- Include related requirements (e.g. companion bots, exit conditions) when they appear in the excerpts.
"""

    return f"""You are a helpful assistant for Fabrix.ai documentation.
Answer the user's question using ONLY the documentation excerpts provided below.
If the answer is not in the excerpts, say "I couldn't find that in the documentation."
Do not infer answers from loosely related excerpts; if none directly address the question, abstain.
Always cite which excerpt your answer comes from using [1], [2], etc.
{extra_instructions}
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


def run_pipeline(question, client, top_k=3, filter_dict=None, category=None):
    """Retrieve chunks and generate an answer. Returns (chunks, answer)."""
    chunks = retrieve(question, client, top_k=top_k, filter_dict=filter_dict)
    chunks = prune_lookup_chunks(question, chunks, category)
    prompt = build_prompt(question, chunks, category=category)
    answer = generate(prompt)
    return chunks, answer


def ask(question, top_k=3, filter_dict=None, category=None):
    client = QdrantClient(path=QDRANT_DIR)

    chunks = retrieve(question, client, top_k=top_k, filter_dict=filter_dict)
    chunks = prune_lookup_chunks(question, chunks, category)

    print(f'\nQuestion: "{question}"')
    if filter_dict:
        print(f"Filter: {filter_dict}")
    print(f"\nRetrieved {len(chunks)} chunks:")
    for i, c in enumerate(chunks, 1):
        meta = c["metadata"]
        ref = meta.get("bot_name") if meta.get("type") == "bot" else meta.get("source")
        print(f"  [{i}] score={c['score']:.3f}  {ref}")

    prompt = build_prompt(question, chunks, category=category)
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
