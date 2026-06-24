"""
agent.py: docs Q&A agent with deterministic query routing and answer judge.

Usage:
    python3 src/agent.py "What parameters does the count loop bot take?"
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys

from dataclasses import dataclass
from openai import OpenAI
from qdrant_client import QdrantClient

sys.path.insert(0, os.path.dirname(__file__))

from config import DOCS_INCLUDE_DIRS, LLM_MODEL, QDRANT_DIR
from doc_urls import BOTS_REL_PREFIX, chunk_metadata_to_url, public_doc_url
from query_qdrant import bot_name_hints, generate, run_pipeline

logger = logging.getLogger(__name__)

CFXQL_KEYWORDS = ("cfxql", "full cfxql", "restricted cfxql", "operators")
COMPARISON_KEYWORDS = ("difference between", " vs ", "compared to", "which")
NEGATIVE_KEYWORDS = (
    "billing",
    "subscription",
    "password",
    "pricing",
    "account",
    "sso",
    "maximum number",
)

SECTION_KEYWORD_RULES = (
    (("install", "setup", "deploy", "backup", "retention"), "installation_guides"),
    (("llm", "pool", "ai fabric", "model"), "ai_fabric"),
    (("pipeline",), "Pipelines"),
    (("extension", "extensions"), "Extensions"),
    (("cfxoia",), "rda_releases"),
    (
        (
            "endpoint",
            "stream",
            "ingest",
            "gateway",
            "architecture",
            "design principle",
            "fabric",
            "rdaf",
            "messaging",
        ),
        "beginners_guide",
    ),
    (
        ("datasource", "integration", "integrate", "kafka", "jira", "servicenow", "kubernetes"),
        "Datasource_Integrations",
    ),
)


@dataclass
class RetrievalPlan:
    intent: str
    type_filter: str | None
    doc_section: str | None
    top_k: int
    category_hint: str | None


@dataclass
class AgentResponse:
    answer: str
    sources: list[dict]
    sufficient: bool


def filter_dict_from_plan(plan):
    out = {}
    if plan.type_filter:
        out["type"] = plan.type_filter
    if plan.doc_section:
        out["doc_section"] = plan.doc_section
    return out or None


def _contains_keyword(question_lower, keywords):
    return any(kw in question_lower for kw in keywords)


def _match_section(question_lower):
    for keywords, doc_section in SECTION_KEYWORD_RULES:
        if _contains_keyword(question_lower, keywords):
            return doc_section
    return None


def _plan_from_llm(question):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    sections = ", ".join(DOCS_INCLUDE_DIRS)
    prompt = f"""Classify this Fabrix.ai documentation question and return JSON only.

Question: {question}

Allowed doc_section values (or null): {sections}

Return JSON with keys:
- intent: bot_lookup | cfxql | comparison | narrative | negative
- type_filter: bot | narrative | null
- doc_section: one allowed section or null
- top_k: integer 5-10
- category_hint: lookup | multi_part | negative | null
"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    data = json.loads(raw)
    doc_section = data.get("doc_section")
    if doc_section and doc_section not in DOCS_INCLUDE_DIRS:
        doc_section = None
    return RetrievalPlan(
        intent=data.get("intent", "narrative"),
        type_filter=data.get("type_filter"),
        doc_section=doc_section,
        top_k=int(data.get("top_k", 5)),
        category_hint=data.get("category_hint"),
    )


def plan_query(question):
    q = question.lower()

    if bot_name_hints(question):
        multi = "and" in q
        top_k = 8 if multi else 5
        return RetrievalPlan(
            intent="bot_lookup",
            type_filter="bot",
            doc_section=None,
            top_k=top_k,
            category_hint="multi_part" if multi else "lookup",
        )

    if _contains_keyword(q, CFXQL_KEYWORDS):
        return RetrievalPlan(
            intent="cfxql",
            type_filter=None,
            doc_section=None,
            top_k=10,
            category_hint=None,
        )

    if _contains_keyword(q, COMPARISON_KEYWORDS):
        return RetrievalPlan(
            intent="comparison",
            type_filter=None,
            doc_section=None,
            top_k=10,
            category_hint=None,
        )

    doc_section = _match_section(q)
    if doc_section:
        return RetrievalPlan(
            intent="narrative",
            type_filter="narrative",
            doc_section=doc_section,
            top_k=5,
            category_hint=None,
        )

    if _contains_keyword(q, NEGATIVE_KEYWORDS):
        return RetrievalPlan(
            intent="negative",
            type_filter=None,
            doc_section=None,
            top_k=5,
            category_hint="negative",
        )

    logger.warning("plan_query: LLM fallback used")
    return _plan_from_llm(question)


def _chunk_summaries(chunks):
    lines = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        ref = meta.get("bot_name") if meta.get("type") == "bot" else meta.get("source", "?")
        preview = chunk["text"][:100].replace("\n", " ")
        lines.append(f"[{i}] {ref}: {preview}")
    return "\n".join(lines) if lines else "(no chunks)"


def is_answer_sufficient(question, answer, chunks):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = f"""Judge whether the answer fully addresses the question using only the retrieved excerpts.

Question: {question}

Answer: {answer}

Retrieved excerpts:
{_chunk_summaries(chunks)}

Return JSON only:
{{"sufficient": true/false, "retry_query": "optional reformulated search query if not sufficient"}}
If sufficient, omit retry_query or set it to null.
"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    return json.loads(raw)


def _chunk_url(meta):
    if meta.get("type") == "bot" and meta.get("source"):
        return public_doc_url(f"{BOTS_REL_PREFIX}/{meta['source']}")
    return chunk_metadata_to_url(meta)


def _sources_from_chunks(chunks):
    sources = []
    for chunk in chunks:
        meta = chunk["metadata"]
        sources.append({
            "title": meta.get("bot_name") or meta.get("source", ""),
            "url": _chunk_url(meta),
            "excerpt": chunk["text"][:200],
        })
    return sources


def answer(question, client=None):
    plan = plan_query(question)
    logger.info(
        "plan_query: intent=%s type_filter=%s doc_section=%s top_k=%s category_hint=%s",
        plan.intent,
        plan.type_filter,
        plan.doc_section,
        plan.top_k,
        plan.category_hint,
    )

    if client is None:
        client = QdrantClient(path=QDRANT_DIR)
    chunks, ans = run_pipeline(
        question,
        client,
        top_k=plan.top_k,
        filter_dict=filter_dict_from_plan(plan),
        category=plan.category_hint,
    )

    judge = is_answer_sufficient(question, ans, chunks)
    sufficient = bool(judge.get("sufficient"))
    retry_query = judge.get("retry_query")

    if not sufficient and retry_query:
        retry_filter = {"type": plan.type_filter} if plan.type_filter else None
        chunks, ans = run_pipeline(
            retry_query,
            client,
            top_k=plan.top_k + 5,
            filter_dict=retry_filter,
            category=plan.category_hint,
        )
        sufficient = True

    return AgentResponse(
        answer=ans,
        sources=_sources_from_chunks(chunks),
        sufficient=sufficient,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python3 src/agent.py \"your question\"")
        sys.exit(1)
    result = answer(" ".join(sys.argv[1:]))
    print(result.answer)
    for source in result.sources:
        print(f"  [{source['title']}]({source['url']})")
