"""
query_qdrant.py, embed the question, pull chunks from Qdrant, ask gpt-4o-mini.

Used by agent.py and runnable standalone. Local path uses file Qdrant;
REMOTE_BASE_URL switches to the VPN fastembed wrapper instead.
"""

import os
import re
import sys
import time
from functools import lru_cache

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


def _local_embedding_model():
    # read model from ingest output so query vector size matches the collection
    path = os.path.join(QDRANT_DIR, "embedding_model.txt")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            model = f.read().strip()
            if model:
                return model
    return EMBEDDING_MODEL


@lru_cache(maxsize=256)
def _embed_question_cached(model: str, question: str):
    url = EMBEDDINGS_URL
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "input": [question]}

    last_err = None
    for attempt in range(2):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            last_err = e
            if attempt == 0:
                time.sleep(1)
                continue
            raise

    raise last_err  # pragma: no cover


def embed_question(question: str):
    model = EMBEDDING_MODEL if REMOTE_BASE_URL else _local_embedding_model()
    return _embed_question_cached(model, question)

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
    # bump chunks whose bot_name matches slugs parsed from the question
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


_PARAM_TABLE_HEADER_RE = re.compile(r"^\|\s*Parameter\s+Name\s*\|", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(s: str) -> str:
    return _HTML_TAG_RE.sub("", s or "").strip()


def parse_bot_param_table(text: str):
    """
    Best-effort parser for the bot parameter table embedded in bot chunks.

    Returns: list[dict] rows: {name, required, type, default, description}
    """
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines()]

    # Find the markdown table header row.
    start = None
    for i, ln in enumerate(lines):
        if _PARAM_TABLE_HEADER_RE.match(ln):
            start = i
            break
    if start is None:
        return []

    rows = []
    for ln in lines[start + 1 :]:
        if not ln.startswith("|"):
            # stop when table ends
            if rows:
                break
            continue

        # Skip separator rows like |---|---|
        if re.match(r"^\|\s*-+\s*\|", ln):
            continue

        parts = [p.strip() for p in ln.strip("|").split("|")]
        if len(parts) < 4:
            continue

        name_raw, type_raw, default_raw, desc_raw = (parts + ["", "", "", ""])[:4]
        name_text = _strip_html(name_raw)
        desc_text = _strip_html(desc_raw)

        # Required markers show up as asterisk or HTML span with red marker.
        required = False
        if "*" in name_raw or "*" in name_text:
            required = True
        name_text = name_text.replace("*", "").strip()

        rows.append(
            {
                "name": name_text,
                "required": required,
                "type": _strip_html(type_raw),
                "default": _strip_html(default_raw),
                "description": desc_text,
            }
        )

    return rows


def retrieve_remote(question, top_k=5, filter_dict=None):
    # VPN path: server embeds + searches; we just POST raw question text
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
    # local Qdrant or remote wrapper, then bot-name rerank and trim to top_k
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
- When the question asks how to load, use, or configure something, list every required parameter (marked * in the table), not only the parameter named in the question.
- For each listed parameter, include its description from the table (e.g. columns as a comma-separated list).
"""
    elif category == "bot_catalog":
        extra_instructions = """
- The excerpts are RDA Extension / bot catalog list pages. Treat extension names as the available bot groups.
- Explain that bots are organized by extension, then list extension names that appear in the excerpts (group by A-B, C, etc. if shown).
- Point the user to the Extension List and Bots pages on docs.fabrix.ai for the full catalog.
- Do NOT say you couldn't find it if extension names are present in the excerpts.
"""
    elif category == "overview":
        extra_instructions = """
- The user asked a broad / getting-started / tools-features question. Give a short high-level overview from the excerpts.
- Cover what RDA Fabric is, key design ideas (e.g. operate close to data), and main building blocks / tools if present
  (ingestion, pipelines, bots, CFXQL, dashboards, integrations, AI fabric).
- If they asked about "tools" or "features", map those to the documented platform capabilities above — do not look for a literal "tools" page.
- End with 2-4 concrete follow-up topics they can ask about next (architecture, bots, CFXQL, integrations, installation).
- Do NOT abstain just because the question is broad or uses words like tools/features; synthesize from architecture / beginners-guide excerpts.
- Keep the answer concise (about one short paragraph + a short bullet list is fine).
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
- When the question has two parts joined by "and" (e.g. what it does and what it expects), give two explicit sentences, one per part, including behavioral context from the excerpt intro (not just the parameter table).
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
    # gpt-4o-mini with simple 429 backoff
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
    # retrieve → prune lookup siblings → build prompt → generate
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
