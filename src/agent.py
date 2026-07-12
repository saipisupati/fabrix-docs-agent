"""
agent.py — scope + multi-facet KB retrieve + ops critique + unified best-answer UX (1B).

CLI: python3 src/agent.py "your question"
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field

from openai import OpenAI
from qdrant_client import QdrantClient

sys.path.insert(0, os.path.dirname(__file__))

from config import LLM_MODEL, QDRANT_DIR
from doc_urls import BOTS_REL_PREFIX, chunk_metadata_to_url, public_doc_url
from kb.store import retrieve_kb
from query_qdrant import (
    bot_name_hints,
    generate,
    parse_bot_param_table,
    prune_lookup_chunks,
    retrieve,
)

logger = logging.getLogger(__name__)

GAP_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "kb", "gap_log.jsonl")


INFERENCE_DISCLOSURE = (
    "*Includes inferred Fabrix guidance not stated verbatim in the docs.*"
)

TRAILER_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```\s*$",
    re.IGNORECASE | re.DOTALL,
)
TRAILER_LOOSE_RE = re.compile(
    r"\n\s*(\{\s*\"examples\".*\})\s*$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class AgentResponse:
    answer: str
    sources: list[dict]
    sufficient: bool
    examples: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    scope: str = "in_scope"  # in_scope | related | out_of_scope
    used_inference: bool = False
    inferred_summary: str = ""
    timing: dict = field(default_factory=dict)  # scope_ms, retrieve_ms, generate_ms, critique_ms, total_ms, llm_calls


def _looks_like_abstention(answer: str) -> bool:
    text = (answer or "").lower()
    return any(
        phrase in text
        for phrase in (
            "couldn't find",
            "could not find",
            "not in the documentation",
            "not in the fabrix",
            "outside the scope",
            "out of scope",
        )
    )


def _parse_llm_json(raw: str) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def _is_vague_ops_question(question: str) -> bool:
    """Broad how/what-is ops questions that need multi-facet retrieve even if scoped in_scope."""
    q = (question or "").lower().strip()
    if len(q) < 12:
        return False
    if "parameter" in q or "parameters" in q:
        return False
    vague_markers = (
        "how do", "how does", "how would", "how can", "how should",
        "what are the", "what is the most", "building block", "important",
        "overview", "architect", "end to end", "end-to-end", "chain",
        "integrate", "set up", "setup", "wire",
        "explain", "differ", "difference", "compare", "versus", " vs ",
        "instead of", "when should", "when would", "numbered step",
        "list operator", "operators available", "walk me through", "add ",
    )
    return any(m in q for m in vague_markers)


def _is_synthesis_question(question: str) -> bool:
    """
    Compare/recommend asks that need inference labeling even when scoped in_scope.
    Generic: not tied to a single eval question.
    """
    q = (question or "").lower().strip()
    if len(q) < 12:
        return False
    compare = any(
        m in q
        for m in (
            "should i", "should we", " vs ", " versus ", " or a ", " or an ",
            "start with", "which to", "which should", "prefer", "better to",
            "toolset or", "persona or",
        )
    )
    if not compare:
        return False
    fabrix_obj = any(
        m in q
        for m in (
            "fabrix", "rda", "toolset", "persona", "agentic", "bot",
            "pipeline", "dataset", "pstream", "cfxql", "dashboard",
            "datasource", "integration",
        )
    )
    return fabrix_obj


def _strip_format_noise(question: str) -> str:
    """Drop presentation constraints that pollute retrieval queries."""
    q = question or ""
    q = re.sub(
        r"\b(?:in\s+)?exactly\s+\d+\s+numbered\s+steps?\b",
        " ",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(
        r"\bwith\s+(?:a\s+)?blank\s+lines?\s+between\s+(?:each\s+)?steps?\b",
        " ",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(r"\bnumbered\s+steps?\b", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\s{2,}", " ", q).strip(" ,.;:")
    return q or question


def plan_scope_and_facets(question: str) -> dict:
    """One LLM call: scope classification + optional multi-facet retrieval plan."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = f"""Classify and plan retrieval for a Fabrix.ai / RDA Fabric / CFX documentation assistant.
Return JSON only.

Question: {question}

Return:
{{
  "scope": "in_scope" | "related" | "out_of_scope",
  "search_query": "short retrieval query for the docs/KB (Fabrix topic only)",
  "reason": "one short sentence",
  "needs_facets": true or false,
  "facets": ["up to 4 from: pipelines, bots, cfxql, datasources, datasets, pstreams, dashboards, ai_fabric, install, integrations, extensions"],
  "search_queries": ["2 to 4 short retrieval queries covering different facets when needs_facets"],
  "primary_objects": ["concrete Fabrix objects likely relevant, e.g. kubernetes-inventory, @c:timed-loop, persistent stream"]
}}

Scope rules:
- in_scope: fact or procedure clearly covered in Fabrix/RDA/CFX docs (bot params, CFXQL, named pipeline/guide, named extensions like agentic_ai, install/config of RDA features, etc.)
- related: synthesis / composition / how-would-I / compare Fabrix objects / broad “building blocks for automation” — multi-doc reasoning
- out_of_scope: ONLY billing/refunds/subscriptions, contractual support SLAs, private VPN/jump-host crypto, private mTLS/ingress cert rotation not in public docs, enterprise quotes/pricing/cost models/list prices, unrelated products, cooking/sports, personal advice
Never mark questions about Fabrix extensions, bots, pipelines, CFXQL, datasets, pstreams, dashboards, or AI Fabric as out_of_scope.
Prefer related over in_scope for combining pieces, comparing objects (pstream vs dataset), broad automation overviews, or designing workflows.
search_query must strip presentation constraints (exact step counts, blank-line formatting).

Facet rules:
- needs_facets=true for related questions, comparisons, broad how-to/ops, dashboards, monitoring, multi-intent.
- needs_facets=false for simple single-doc lookups (bot params, one named fact).
- When needs_facets: prefer classic RDA diversity (bots, CFXQL, pipelines, datasets, pstreams, dashboards, integrations);
  include ai_fabric only for agents/AI/toolsets/personas or broad automation.
- Dashboard questions: include datasets and/or pstreams, not only collectors.
- For out_of_scope: empty facets and search_queries.
"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    data = _parse_llm_json(response.choices[0].message.content or "")
    if not data:
        data = {
            "scope": "related",
            "search_query": question,
            "reason": "parse_error",
            "needs_facets": True,
        }
    scope = data.get("scope") or "related"
    if scope not in ("in_scope", "related", "out_of_scope"):
        scope = "related"
    search_query = (data.get("search_query") or question).strip() or question
    facets = [str(f) for f in (data.get("facets") or []) if str(f).strip()][:4]
    queries = [str(q).strip() for q in (data.get("search_queries") or []) if str(q).strip()][:4]
    objects = [str(o).strip() for o in (data.get("primary_objects") or []) if str(o).strip()][:8]
    needs_facets = bool(data.get("needs_facets"))
    if scope == "related":
        needs_facets = True
    if scope == "out_of_scope":
        needs_facets = False
        facets, queries, objects = [], [], []
    if needs_facets and not queries:
        queries = [search_query or question]
    return {
        "scope": scope,
        "search_query": search_query,
        "reason": data.get("reason", ""),
        "needs_facets": needs_facets,
        "facets": facets,
        "search_queries": queries,
        "primary_objects": objects,
    }


def plan_facets(question: str) -> dict:
    """LLM: which Fabrix facets to retrieve (thin-retrieve escalation only)."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = f"""Plan retrieval facets for a Fabrix.ai / RDA Fabric docs assistant.
Return JSON only.

Question: {question}

Return:
{{
  "facets": ["up to 4 from: pipelines, bots, cfxql, datasources, datasets, pstreams, dashboards, ai_fabric, install, integrations, extensions"],
  "search_queries": ["2 to 4 short retrieval queries covering different facets"],
  "primary_objects": ["concrete Fabrix objects likely relevant, e.g. kubernetes-inventory, @c:timed-loop, persistent stream"]
}}

Rules:
- Prefer diversity across classic RDA (bots, CFXQL, pipelines, datasets, pstreams, dashboards, integrations)
  and Agentic (ai_fabric) only when the question involves agents/AI/toolsets/personas OR is broad automation.
- For dashboard questions include datasets and/or pstreams, not only collectors.
- For broad automation questions include bots and pipelines, not only ai_fabric.
- search_queries must be Fabrix topics only (no step-count / formatting noise).
"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    data = _parse_llm_json(response.choices[0].message.content or "")
    facets = [str(f) for f in (data.get("facets") or []) if str(f).strip()][:4]
    queries = [str(q).strip() for q in (data.get("search_queries") or []) if str(q).strip()][:4]
    objects = [str(o).strip() for o in (data.get("primary_objects") or []) if str(o).strip()][:8]
    if not queries:
        queries = [question]
    return {"facets": facets, "search_queries": queries, "primary_objects": objects}


def classify_scope(question: str) -> dict:
    """LLM scope only (compat wrapper over merged planner)."""
    planned = plan_scope_and_facets(question)
    return {
        "scope": planned["scope"],
        "search_query": planned["search_query"],
        "reason": planned.get("reason", ""),
    }


def _draft_looks_clean(
    answer: str,
    used_inference: bool,
    question: str = "",
) -> bool:
    """Cheap local gate: skip LLM critique when draft already looks demo-safe."""
    if not answer or _looks_like_abstention(answer):
        return False
    low = answer.lower()
    if "```json" in low or re.search(r"(?m)^```\s*$", answer):
        return False
    if used_inference and INFERENCE_DISCLOSURE.lower() not in low and "next (inferred)" not in low:
        return False
    nums = [int(m.group(1)) for m in re.finditer(r"(?m)^(\d+)\.\s+", answer)]
    if len(nums) >= 3 and all(n == 1 for n in nums):
        return False
    if len(nums) >= 2 and nums.count(1) >= 2 and max(nums) > 1:
        return False
    if re.search(r"(?m)^\d+\.\s+\S[^\n]*[ \t]+\d+\.\s+", answer):
        return False
    if question and _off_family_bot_tokens(question, answer):
        return False
    return True


def _datasource_ask_missing_sink(question: str, answer: str) -> bool:
    """True when a wire/add-datasource ask lacks stream/dataset/pipeline/dashboard."""
    q = (question or "").lower()
    if not any(
        w in q
        for w in (
            "datasource", "wire", "end-to-end", "walk me through",
            "add datadog", "add nagios", "add splunk", "add zabbix",
        )
    ) and not (
        "add " in q and any(p in q for p in ("datadog", "nagios", "splunk", "zabbix", "prometheus"))
    ):
        return False
    a = (answer or "").lower()
    return not any(w in a for w in ("stream", "pstream", "dataset", "pipeline", "dashboard"))


def _entry_section_key(entry: dict) -> str:
    src = (entry.get("source") or entry.get("title") or "").lower()
    for key in (
        "ai_fabric", "pipeline", "bot", "cfxql", "datasource", "dataset",
        "pstream", "dashboard", "install", "extension", "integration",
    ):
        if key in src:
            return key
    return "other"


def merge_kb_entries_diverse(entry_lists: list[list[dict]], limit: int = 12) -> list[dict]:
    """Dedupe by id/source/title and prefer section diversity."""
    seen: set[str] = set()
    pooled: list[dict] = []
    for lst in entry_lists:
        for e in lst:
            key = str(e.get("id") or e.get("source") or e.get("title") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            pooled.append(e)
    pooled.sort(key=lambda e: float(e.get("score") or 0), reverse=True)

    by_sec: dict[str, list[dict]] = {}
    for e in pooled:
        by_sec.setdefault(_entry_section_key(e), []).append(e)
    merged: list[dict] = []
    while len(merged) < limit and any(by_sec.values()):
        for sec in list(by_sec.keys()):
            if not by_sec[sec]:
                continue
            merged.append(by_sec[sec].pop(0))
            if len(merged) >= limit:
                break
        by_sec = {k: v for k, v in by_sec.items() if v}
    return merged


def retrieve_multi_facet(
    question: str,
    client: QdrantClient,
    search_queries: list[str],
) -> tuple[list[dict], list[dict]]:
    """Retrieve KB (+ chunks if thin) per query, merge with diversity."""
    kb_lists: list[list[dict]] = []
    chunk_lists: list[list[dict]] = []
    for q in search_queries[:4]:
        hits = retrieve_kb(q, top_k=6)
        kb_lists.append(hits)
        if len(hits) < 2:
            chunk_lists.append(retrieve(q, client, top_k=4, filter_dict=None))
    kb_entries = merge_kb_entries_diverse(kb_lists, limit=12)
    chunks: list[dict] = []
    seen_txt: set[str] = set()
    for lst in chunk_lists:
        for c in lst:
            t = (c.get("text") or "")[:120]
            if t in seen_txt:
                continue
            seen_txt.add(t)
            chunks.append(c)
    if len(kb_entries) < 3 and not chunks:
        chunks = retrieve(question, client, top_k=6, filter_dict=None)
    elif len(kb_entries) < 3:
        extra = retrieve(question, client, top_k=6, filter_dict=None)
        for c in extra:
            t = (c.get("text") or "")[:120]
            if t not in seen_txt:
                chunks.append(c)
                seen_txt.add(t)
    return kb_entries, chunks[:8]


def critique_ops_answer(question: str, answer: str, facets: list[str], primary_objects: list[str]) -> dict:
    """One LLM critique pass for related/ops answers."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = f"""Critique this Fabrix docs assistant draft. Return JSON only.

Question: {question}
Planned facets: {facets}
Primary objects to prefer if in docs: {primary_objects}

Draft answer:
{answer[:3500]}

Return:
{{
  "missing_layers": ["fabrix layers the answer should cover but skipped"],
  "wrong_emphasis": "short note if answer latched onto wrong facet",
  "overclaim": true or false,
  "fix_needed": true or false,
  "revision_notes": "concrete fixes if fix_needed else empty"
}}

Checklist:
- Broad automation: should mention bots/pipelines/CFXQL and/or Agentic stacks appropriately — not Agentic-only unless question is AI-agent specific.
- Dashboard questions: should mention datasets or pstreams, not only Edge Collector.
- Comparisons (e.g. pstream vs dataset, vCenter vs vROps): cover both sides from docs; inferred wiring under Next (inferred).
- Multi-intent questions: separate documented facts per intent; put undocumented handoffs under Next (inferred).
- Product fidelity: named bots (@prefix: / *prefix:) must belong to products named in the question (e.g. ServiceNow+Slack must not cite BMC/Remedy bots). Flag off-product bots as wrong_emphasis + fix_needed.
- Single-product datasource asks: stay on that product; include credentials/auth + named bots + stream/dataset; flag unrelated integrations as wrong_emphasis; prefer v2 bot family when both v1 and v2 appear.
- Multi-intent: separate documented facts per intent; cross-wiring under Next (inferred).
- Host OS / jump-box details only under Next (inferred) or gaps — not as documented steps.
- Capability honesty: do not assert product can change pipelines / has an audit UI / etc. unless an excerpt supports it; otherwise Next (inferred) or gaps. Overclaim=true if invented.
- Format-only constraints (exact N steps) are presentation — do not abstain if CFXQL/operators/topic are in excerpts.
- Undocumented handoffs only under Next (inferred); no invented auto-formatting.
- No nested 1. 1. 2. numbering or "1) intro then 1. 2."; no leaked ```json fences.
- Do not abstain when excerpts contain the Fabrix topic; format constraints (step count) are presentation only.
"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    data = _parse_llm_json(response.choices[0].message.content or "")
    return {
        "missing_layers": data.get("missing_layers") or [],
        "wrong_emphasis": str(data.get("wrong_emphasis") or ""),
        "overclaim": bool(data.get("overclaim")),
        "fix_needed": bool(data.get("fix_needed")),
        "revision_notes": str(data.get("revision_notes") or "").strip(),
    }


def _chunk_url(meta: dict) -> str:
    if meta.get("type") == "bot" and meta.get("source"):
        return public_doc_url(f"{BOTS_REL_PREFIX}/{meta['source']}")
    return chunk_metadata_to_url(meta)


def _sources_from_kb(entries: list[dict]) -> list[dict]:
    sources = []
    seen = set()
    for e in entries:
        url = e.get("url") or ""
        source = e.get("source") or ""
        if not url and source:
            if source.endswith(".md") and "/" not in source and not source.startswith("Bots"):
                url = public_doc_url(f"{BOTS_REL_PREFIX}/{source}")
            else:
                url = public_doc_url(source)
        key = (e.get("title") or source, url)
        if key in seen or (not url and not source):
            continue
        seen.add(key)
        sources.append({
            "title": e.get("title") or source or "docs",
            "url": url,
            "excerpt": (e.get("text") or "")[:200],
        })
    return sources


def _sources_from_chunks(chunks: list[dict]) -> list[dict]:
    sources = []
    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        bot_name = meta.get("bot_name") or ""
        title = bot_name if bot_name and bot_name != "n/a" else (meta.get("source") or "")
        sources.append({
            "title": title,
            "url": _chunk_url(meta),
            "excerpt": chunk.get("text", "")[:200],
        })
    return sources


def _format_kb_context(entries: list[dict]) -> str:
    parts = []
    for i, e in enumerate(entries, 1):
        ex = e.get("example") or ""
        block = f"[{i}] ({e.get('kind')}) {e.get('title')}\nSource: {e.get('source')}\n{e.get('text')}"
        if ex:
            block += f"\nExample: {ex[:400]}"
        parts.append(block)
    return "\n\n---\n\n".join(parts) if parts else "(no KB entries)"


def _format_chunk_context(chunks: list[dict], start_index: int = 1) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start_index):
        meta = chunk.get("metadata") or {}
        source = meta.get("bot_name") if meta.get("type") == "bot" else meta.get("source")
        parts.append(f"[{i}] Source: {source}\n{chunk.get('text', '')}")
    return "\n\n---\n\n".join(parts) if parts else ""


def build_kb_prompt(
    question: str,
    kb_entries: list[dict],
    chunks: list[dict],
    scope: str,
    primary_objects: list[str] | None = None,
    revision_notes: str = "",
) -> str:
    kb_ctx = _format_kb_context(kb_entries)
    chunk_ctx = _format_chunk_context(chunks, start_index=len(kb_entries) + 1)
    docs = kb_ctx
    if chunk_ctx:
        docs += "\n\n---\n\nADDITIONAL DOC CHUNKS:\n" + chunk_ctx

    objects_line = ""
    if primary_objects:
        objects_line = (
            "- Prefer naming these Fabrix objects when they appear in the excerpts: "
            + ", ".join(primary_objects[:8])
            + ".\n"
        )

    allowed_fams = _integration_family_hits(question)
    fidelity_line = ""
    if allowed_fams:
        fidelity_line = (
            "- Product fidelity: only name bots whose prefix/family matches products in the "
            f"question ({', '.join(allowed_fams)}). Never cite sibling-product bots "
            "(e.g. BMC/Remedy bots on a ServiceNow question, Zabbix on a Prometheus question).\n"
        )

    mental_model = """
Fabrix mental model (use when relevant; do not keyword-stuff):
- Classic RDA automation stack: bots → CFXQL → pipelines → datasets / persistent streams (pstreams) → dashboards / integrations.
- Agentic stack: Toolsets / Personas / Prompt Templates — lead with this only for agents/AI Fabric questions.
- Broad “automation / how Fabrix works” questions: briefly cover classic RDA building blocks; add Agentic only if excerpts support it — do not collapse to Agentic-only.
- Dashboard data path: prefer datasets / pstreams / pipelines from excerpts; do not answer with only Edge Collector unless that is all the excerpts support (then say so in Gaps).
"""

    path_first = """
- For "how would I / chain / integrate / wire / add X as a datasource" questions, use this shape:
  **Documented Fabrix path**   ← bold header only, NOT a numbered step
  1. credentials / auth for the named product (API token, username/password, read-only user — from excerpts)
  2. concrete bots from that product family (name them exactly as in excerpts)
  3. stream or dataset handoff
  4. optional dashboard / downstream step if in excerpts
  **Next (inferred):** undocumented handoffs (network access, jump-box/OS, security groups) — not numbered
  **Prerequisites** once at the end if needed (not numbered).
- Multi-intent questions (compare A vs B, then wire to C): cover each documented intent, then put cross-product handoffs under **Next (inferred):**.
- If the user asks for N numbered steps / blank lines, still answer the Fabrix topic; use continuous 1..N when content allows — do not abstain for formatting.
- If excerpts include both a v1 and v2 bot family for the same product (e.g. prometheus and prometheusv2),
  lead with the v2 family and mention v1 only as an alternate/legacy option — do not mix unmarked.
- Stay on the named integration(s); do not drag in unrelated products unless asked.
- Capability honesty: only claim what excerpts support (e.g. Copilot changing pipelines, audit trails).
  If not documented, use **Next (inferred):** or gaps — do not invent UI/capabilities.
- Continuous numbers 1. 2. 3. — never restart at 1.; no blank lines between steps unless the user asked for blank lines.
- Do NOT invent automation such as "automatically formatted for dashboards" unless an excerpt says that.
- Trailer examples: concrete excerpt lines only from the same product family.
"""

    if scope == "related":
        infer_guidance = f"""
- Write ONE coherent best answer that combines documented facts with limited Fabrix
  technical synthesis implied by the docs.
- Cite documented claims with [1], [2], … Do NOT put [n] on inferred reasoning.
- Set used_inference=true and fill inferred_summary when you synthesize beyond verbatim docs.
{objects_line}{fidelity_line}{path_first}
"""
    else:
        infer_guidance = f"""
- Write ONE coherent best answer from the excerpts; cite with [1], [2], …
- If you add connective technical reasoning beyond a single excerpt, set used_inference=true
  and fill inferred_summary. Otherwise used_inference=false and inferred_summary="".
- Do NOT put [n] on inferred reasoning.
{objects_line}{fidelity_line}{path_first}
"""

    revision_block = ""
    if revision_notes:
        revision_block = f"\nREVISION REQUIRED (apply these fixes):\n{revision_notes}\n"

    return f"""You are the Fabrix.ai / RDA Fabric documentation assistant backed by a structured knowledge base.

Answer using the KB entries and doc excerpts below.
Scope for this question: {scope}
{mental_model}{revision_block}
Output format (strict):
1) A single unified answer in markdown — the best direct response for the user.
   - Do NOT use report headings like ## Examples, ## Gaps, or ## Inferred.
   - Do NOT invent product features, numeric limits, APIs, or UI flows absent from excerpts.
   - Do NOT leave raw ```json fences or trailer JSON visible in the answer body.
{infer_guidance}
2) Then a JSON trailer in a fenced ```json code block with exactly this shape:
{{
  "examples": ["short doc example snippets ONLY from the same topic as the question"],
  "gaps": ["what the public docs do not cover relative to the question, including missing handoffs"],
  "used_inference": true or false,
  "inferred_summary": "one short sentence of inferred synthesis, or empty string"
}}

If the excerpts do not support an answer, the unified answer should be:
"I couldn't find that in the documentation."
and set used_inference=false.

KNOWLEDGE BASE / DOCUMENTATION:
{docs}

USER QUESTION:
{question}

ANSWER:"""


def _parse_legacy_sections(text: str) -> tuple[list[str], list[str]]:
    examples: list[str] = []
    gaps: list[str] = []
    ex_match = re.search(r"##\s*Examples\s*\n(.*?)(?=\n##\s|\Z)", text, re.IGNORECASE | re.DOTALL)
    gap_match = re.search(r"##\s*Gaps\s*\n(.*?)(?=\n##\s|\Z)", text, re.IGNORECASE | re.DOTALL)
    if ex_match:
        block = ex_match.group(1).strip()
        examples = [
            re.sub(r"^[-*]\s+", "", ln).strip()
            for ln in block.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ][:8]
    if gap_match:
        block = gap_match.group(1).strip()
        gaps = [
            re.sub(r"^[-*]\s+", "", ln).strip()
            for ln in block.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ][:8]
    return examples, gaps


def _strip_legacy_headings(text: str) -> str:
    """Remove old report-style sections from user-facing answer."""
    cleaned = re.sub(
        r"\n##\s*(Examples|Gaps|Inferred(?:\s*\([^)]*\))?)\s*\n.*?(?=\n##\s|\Z)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(r"^#+\s*Main answer\s*:?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^1\.\s*\*?\*?Main answer\*?\*?\s*:?\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def parse_unified_response(raw: str) -> dict:
    """Split model output into unified answer + trailer metadata."""
    text = (raw or "").strip()
    trailer: dict = {}
    body = text

    m = TRAILER_RE.search(text)
    if not m:
        m = TRAILER_LOOSE_RE.search(text)
    if m:
        body = text[: m.start()].strip()
        try:
            trailer = json.loads(m.group(1))
        except json.JSONDecodeError:
            trailer = {}

    examples = trailer.get("examples") if isinstance(trailer.get("examples"), list) else []
    gaps = trailer.get("gaps") if isinstance(trailer.get("gaps"), list) else []
    examples = [str(x).strip() for x in examples if str(x).strip()][:8]
    gaps = [str(x).strip() for x in gaps if str(x).strip()][:8]
    used_inference = bool(trailer.get("used_inference"))
    inferred_summary = str(trailer.get("inferred_summary") or "").strip()

    # Fallback: legacy ## Examples / ## Gaps / ## Inferred
    if not examples or not gaps:
        leg_ex, leg_gaps = _parse_legacy_sections(text)
        if not examples:
            examples = leg_ex
        if not gaps:
            gaps = leg_gaps
    if not used_inference and re.search(
        r"##\s*Inferred\s*\(not in documentation\)", text, re.IGNORECASE
    ):
        used_inference = True
        if not inferred_summary:
            inf = re.search(
                r"##\s*Inferred\s*\(not in documentation\)\s*\n(.*?)(?=\n##\s|\Z)",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if inf:
                inferred_summary = " ".join(inf.group(1).split())[:300]

    body = _strip_legacy_headings(body)
    # Drop any leftover fence if model put JSON without match
    body = re.sub(r"\n```(?:json)?\s*\{[\s\S]*\}\s*```\s*$", "", body).strip()

    if used_inference and INFERENCE_DISCLOSURE not in body:
        body = f"{body}\n\n{INFERENCE_DISCLOSURE}".strip()

    body = polish_answer_text(body)

    return {
        "answer": body,
        "examples": examples,
        "gaps": gaps,
        "used_inference": used_inference,
        "inferred_summary": inferred_summary,
    }


def polish_answer_text(text: str) -> str:
    """Collapse excess blank lines; strip leaked JSON fences; renumber broken lists."""
    if not text:
        return text
    cleaned = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leaked trailer / bare json fences from user-facing body
    cleaned = re.sub(r"\n```(?:json)?\s*\{[\s\S]*?\}\s*```", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n```(?:json)?\s*$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"\n```\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(
        r"\n\{\s*\"examples\"[\s\S]*\}\s*$",
        "",
        cleaned,
    )
    # Normalize "1)" markers to "1."
    cleaned = re.sub(r"(?m)^(\d+)\)\s+", r"\1. ", cleaned)
    # Split same-line nested restarts: "1. **Header** 1. step 2. more"
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = re.sub(
            r"(?m)^(\d+\.\s+[^\n]*?)[ \t]+(\d+\.\s+)",
            r"\1\n\2",
            cleaned,
        )

    # Demote numbered section labels to bold headers (not steps)
    header_pat = re.compile(
        r"(?i)^(\d+)\.\s+\*{0,2}("
        r"(?:Documented\s+)?Fabrix\s+[Pp]ath|"
        r"Next\s*\(\s*inferred\s*\)|"
        r"Prerequisites"
        r")\*{0,2}\s*:?\s*$"
    )
    demoted: list[str] = []
    for ln in cleaned.split("\n"):
        hm = header_pat.match(ln.strip())
        if not hm:
            demoted.append(ln)
            continue
        label = hm.group(2).strip()
        low = label.lower()
        if "path" in low:
            demoted.append("**Documented Fabrix path**")
        elif "next" in low:
            demoted.append("**Next (inferred):**")
        else:
            demoted.append("**Prerequisites**")
    cleaned = "\n".join(demoted)

    lines = cleaned.split("\n")
    i = 0
    out: list[str] = []
    while i < len(lines):
        if not re.match(r"^\d+\.\s+", lines[i]):
            out.append(lines[i])
            i += 1
            continue

        run_idxs: list[int] = []
        j = i
        while j < len(lines):
            if re.match(r"^\d+\.\s+", lines[j]):
                run_idxs.append(j)
                j += 1
                continue
            if lines[j].strip() == "":
                k = j + 1
                while k < len(lines) and lines[k].strip() == "":
                    k += 1
                if k < len(lines) and re.match(r"^\d+\.\s+", lines[k]):
                    j = k
                    continue
            break

        nums = [int(re.match(r"^(\d+)\.", lines[idx]).group(1)) for idx in run_idxs]
        # Renumber if all 1. OR if numbering restarts mid-run (e.g. 1,1,2 or 1,2,1)
        # or after demoting a header left steps starting at 2.
        restart = False
        if len(run_idxs) >= 2:
            seen_gt1 = False
            for n in nums:
                if n == 1 and seen_gt1:
                    restart = True
                    break
                if n > 1:
                    seen_gt1 = True
        expected = list(range(1, len(run_idxs) + 1))
        off_sequence = len(run_idxs) >= 1 and nums != expected
        needs_renumber = (
            (len(run_idxs) >= 3 and all(n == 1 for n in nums))
            or restart
            or (len(run_idxs) >= 2 and nums[0] == 1 and nums.count(1) >= 2)
            or off_sequence
        )
        if needs_renumber:
            n = 1
            cursor = i
            while cursor < j:
                if cursor in run_idxs:
                    out.append(re.sub(r"^\d+\.", f"{n}.", lines[cursor], count=1))
                    n += 1
                else:
                    if lines[cursor].strip() != "":
                        out.append(lines[cursor])
                cursor += 1
            i = j
            continue

        while i < j:
            out.append(lines[i])
            i += 1

    return "\n".join(out).strip()


def _log_gap(question: str, scope: str, gaps: list[str], abstained: bool) -> None:
    try:
        os.makedirs(os.path.dirname(GAP_LOG_PATH), exist_ok=True)
        row = {
            "question": question,
            "scope": scope,
            "gaps": gaps,
            "abstained": abstained,
        }
        with open(GAP_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("gap log failed: %s", e)


# Integration product families (aliases → canonical). Soft facets are separate.
INTEGRATION_FAMILIES: list[tuple[str, tuple[str, ...]]] = [
    ("prometheus", ("prometheus", "prometheusv2", "prometheus_v2")),
    ("zabbix", ("zabbix",)),
    ("splunk", ("splunk",)),
    ("servicenow", ("servicenow", "service now", "sn ticketing")),
    ("bmc", ("bmc-remedy", "bmc remedy", "bmc_remedy")),
    ("kubernetes", ("kubernetes", "k8s", "kubectl", "kubernetes-inventory")),
    ("vmware", ("vmware", "vcenter", "vrealize", "vrops")),
    ("cisco_ucs", ("ucsm", "ucs-manager", "ucs manager", "cisco ucs")),
    ("pagerduty", ("pagerduty", "pager duty")),
    ("datadog", ("datadog",)),
    ("nagios", ("nagios", "nagios xi", "nagiosxi")),
    ("slack", ("slack",)),
    ("aws", ("aws", "amazon web services")),
    ("linux", ("linux-inventory", "linux-os", "linux os")),
]

SOFT_FACET_ALIASES: list[tuple[str, ...]] = [
    ("cfxql",),
    ("pstream", "persistent stream", "persistent-stream"),
    ("dashboard",),
    ("pipeline",),
    ("dataset",),
    ("toolset", "persona"),
]

AGENTIC_MARKERS: tuple[str, ...] = (
    "fabio", "copilot", "ai fabric", "ai_fabric", "agentic",
)

BOT_TOKEN_RE = re.compile(r"[@*]([a-zA-Z0-9][a-zA-Z0-9_-]{1,60})\s*:")


def _integration_family_hits(question: str) -> list[str]:
    """Canonical integration families explicitly named in the question."""
    q = (question or "").lower()
    hits: list[str] = []
    for canonical, aliases in INTEGRATION_FAMILIES:
        # Bare "linux"/"ubuntu" host mentions are red herrings, not linux-inventory
        if canonical == "linux":
            if any(a in q for a in ("linux-inventory", "linux-os", "linux os")):
                hits.append(canonical)
            continue
        if canonical == "servicenow":
            if "servicenow" in q or "service now" in q or "sn ticketing" in q:
                hits.append(canonical)
            elif re.search(r"\bsn\b", q):
                hits.append(canonical)
            continue
        if canonical == "bmc":
            # Do not treat bare "remedy" alone; require BMC markers
            if any(a in q for a in ("bmc-remedy", "bmc remedy", "bmc_remedy")) or (
                "bmc" in q and "remedy" in q
            ):
                hits.append(canonical)
            continue
        if any(a in q for a in aliases):
            hits.append(canonical)
    return hits


def _single_product_mode(question: str) -> tuple[bool, str | None]:
    """True when exactly one integration family is named."""
    hits = _integration_family_hits(question)
    if len(hits) == 1:
        return True, hits[0]
    return False, None


def _is_agentic_question(question: str) -> bool:
    """Agentic/Copilot ask with no named integration product."""
    q = (question or "").lower()
    if _integration_family_hits(question):
        return False
    return any(m in q for m in AGENTIC_MARKERS)


def _blocked_sibling_terms(question: str) -> list[str]:
    """Aliases of families not named in the question (when ≥1 family is named)."""
    allowed = set(_integration_family_hits(question))
    if not allowed:
        return []
    blocked: list[str] = []
    for canonical, aliases in INTEGRATION_FAMILIES:
        if canonical in allowed:
            continue
        blocked.extend(aliases)
        blocked.append(canonical)
    if "linux" not in allowed:
        blocked.extend(["linux-inventory", "linux-os", "linux os"])
    return sorted(set(blocked), key=len, reverse=True)


def _family_match_terms(canonical: str) -> list[str]:
    for name, aliases in INTEGRATION_FAMILIES:
        if name == canonical:
            return list(aliases) + [canonical]
    return [canonical]


def _allowed_family_terms(question: str) -> list[str]:
    terms: list[str] = []
    for fam in _integration_family_hits(question):
        terms.extend(_family_match_terms(fam))
    return sorted(set(terms), key=len, reverse=True)


def _blob_has_blocked(blob: str, blocked: list[str]) -> bool:
    low = (blob or "").lower()
    return any(b in low for b in blocked)


def _blob_family_hits(blob: str) -> list[str]:
    """Which registry families appear in a text blob."""
    low = (blob or "").lower()
    hits: list[str] = []
    for canonical, aliases in INTEGRATION_FAMILIES:
        terms = list(aliases) + [canonical]
        if any(t in low for t in terms):
            hits.append(canonical)
    return hits


def _family_for_bot_prefix(prefix: str) -> str | None:
    """Map a bot name prefix (e.g. bmc-remedy, servicenow) to a registry family."""
    p = (prefix or "").lower().strip()
    if not p:
        return None
    best: tuple[int, str] | None = None
    for canonical, aliases in INTEGRATION_FAMILIES:
        for term in list(aliases) + [canonical]:
            t = term.lower()
            if t in p or p in t.replace(" ", "-") or p in t.replace(" ", "_"):
                score = len(t)
                if best is None or score > best[0]:
                    best = (score, canonical)
    return best[1] if best else None


def _bot_tokens_in_text(text: str) -> list[str]:
    return [m.group(1).lower() for m in BOT_TOKEN_RE.finditer(text or "")]


def _off_family_bot_tokens(question: str, answer: str) -> list[str]:
    """
    Bot prefixes in the answer that map to a known family outside the question's
    allowlist. Unknown prefixes are left alone.
    """
    allowed = set(_integration_family_hits(question))
    if not allowed:
        return []
    bad: list[str] = []
    seen: set[str] = set()
    for prefix in _bot_tokens_in_text(answer):
        if prefix in seen:
            continue
        seen.add(prefix)
        fam = _family_for_bot_prefix(prefix)
        if fam and fam not in allowed:
            bad.append(prefix)
    return bad


def _question_topic_keys(question: str) -> list[str]:
    """Topic keys used to keep examples/sources on-subject."""
    q = (question or "").lower()
    keys: list[str] = []
    for canonical, aliases in INTEGRATION_FAMILIES:
        if canonical == "linux":
            if any(a in q for a in ("linux-inventory", "linux-os", "linux os")) or (
                "linux" in q and not _integration_family_hits(question)
            ):
                keys.extend(list(aliases) + ["linux"])
            continue
        if canonical in _integration_family_hits(question):
            keys.extend(list(aliases) + [canonical])
    for group in SOFT_FACET_ALIASES:
        if any(a in q for a in group):
            keys.extend(group)
    for m in AGENTIC_MARKERS:
        if m in q:
            keys.append(m)

    stop = {
        "what", "when", "where", "which", "would", "could", "should", "after",
        "before", "about", "using", "with", "into", "from", "that", "this",
        "have", "does", "how", "build", "chain", "make", "friendly", "single",
        "following", "parameters", "take", "used", "for", "and", "the",
        "ubuntu", "jump", "host", "helps", "matter", "matters", "also", "runs",
        "wire", "fabrix", "fabric",
    }
    for w in re.findall(r"[a-z0-9][a-z0-9_-]{3,}", q):
        if w not in stop and w not in keys:
            keys.append(w)
    return keys


def _text_matches_topic(text: str, keys: list[str]) -> bool:
    if not keys:
        return True
    t = (text or "").lower()
    return any(k in t for k in keys)


def _filter_examples_to_topic(
    examples: list[str],
    question: str,
    kb_entries: list[dict] | None = None,
) -> list[str]:
    """Drop cross-topic example snippets (sibling integrations when families named)."""
    keys = _question_topic_keys(question)
    if not keys or not examples:
        return examples

    blocked = _blocked_sibling_terms(question)
    kept: list[str] = []
    for ex in examples:
        low = (ex or "").lower()
        if blocked and _blob_has_blocked(low, blocked):
            # keep if it also names an allowed family
            allowed_terms = _allowed_family_terms(question)
            if allowed_terms and any(t in low for t in allowed_terms):
                kept.append(ex)
            continue
        if _text_matches_topic(ex, keys):
            kept.append(ex)
    if kept:
        return kept[:8]

    if kb_entries:
        for e in kb_entries:
            blob = f"{e.get('title') or ''} {e.get('source') or ''} {e.get('example') or ''}"
            if blocked and _blob_has_blocked(blob, blocked):
                allowed_terms = _allowed_family_terms(question)
                if not (allowed_terms and any(t in blob.lower() for t in allowed_terms)):
                    continue
            if not _text_matches_topic(blob, keys):
                continue
            if e.get("example"):
                kept.append(str(e["example"])[:400])
            if len(kept) >= 3:
                break
    return kept[:8]


def _filter_sources_to_topic(sources: list[dict], question: str) -> list[dict]:
    """
    Topic-filter sources when ≥1 integration family is named, or for agentic asks.
    Drop blobs that only match off-allowlist families. Fail open if empty.
    """
    if not sources:
        return sources

    allowed = _integration_family_hits(question)
    if not allowed:
        if not _is_agentic_question(question):
            return sources
        keys = _question_topic_keys(question)
        kept_a: list[dict] = []
        for s in sources:
            blob = f"{s.get('title') or ''} {s.get('url') or ''} {s.get('excerpt') or ''}"
            low = blob.lower()
            fams = _blob_family_hits(low)
            if fams:
                continue
            if _text_matches_topic(blob, keys):
                kept_a.append(s)
        if kept_a:
            return kept_a[:12]
        logger.info("agentic source filter emptied list; fail-open keeping original sources")
        return sources

    allowed_set = set(allowed)
    allowed_terms = _allowed_family_terms(question)
    kept: list[dict] = []
    for s in sources:
        blob = f"{s.get('title') or ''} {s.get('url') or ''} {s.get('excerpt') or ''}"
        low = blob.lower()
        fams = _blob_family_hits(low)
        if fams and not any(f in allowed_set for f in fams):
            continue
        if fams and any(f in allowed_set for f in fams):
            kept.append(s)
            continue
        # Neutral (no registry family): keep if soft-topic match or no strong conflict
        if any(t in low for t in allowed_terms) or _text_matches_topic(
            blob, _question_topic_keys(question)
        ):
            kept.append(s)
    if kept:
        return kept[:12]
    logger.info("source filter emptied list; fail-open keeping original sources")
    return sources


def _rank_entries_for_product(entries: list[dict], question: str) -> list[dict]:
    """Soft-rank KB entries so allowed-family (or agentic topic) hits come first."""
    if not entries:
        return entries
    allowed = _integration_family_hits(question)
    if allowed:
        terms = _allowed_family_terms(question)
        matched: list[dict] = []
        other: list[dict] = []
        for e in entries:
            blob = f"{e.get('title') or ''} {e.get('source') or ''} {e.get('text') or ''}".lower()
            fams = _blob_family_hits(blob)
            if fams and not any(f in allowed for f in fams):
                other.append(e)
                continue
            if any(t in blob for t in terms):
                matched.append(e)
            else:
                other.append(e)
        return matched + other

    if _is_agentic_question(question):
        keys = _question_topic_keys(question)
        matched = []
        other = []
        for e in entries:
            blob = f"{e.get('title') or ''} {e.get('source') or ''} {e.get('text') or ''}".lower()
            if _blob_family_hits(blob):
                other.append(e)
                continue
            if _text_matches_topic(blob, keys):
                matched.append(e)
            else:
                other.append(e)
        return matched + other
    return entries


def _filter_kb_entries_to_families(entries: list[dict], question: str) -> list[dict]:
    """Drop KB entries that only match off-allowlist families. Fail open."""
    if not entries:
        return entries
    allowed = set(_integration_family_hits(question))
    if not allowed:
        if not _is_agentic_question(question):
            return entries
        keys = _question_topic_keys(question)
        kept_a = []
        for e in entries:
            blob = f"{e.get('title') or ''} {e.get('source') or ''} {e.get('text') or ''}"
            if _blob_family_hits(blob.lower()):
                continue
            if _text_matches_topic(blob, keys):
                kept_a.append(e)
        return kept_a or entries

    kept: list[dict] = []
    for e in entries:
        blob = f"{e.get('title') or ''} {e.get('source') or ''} {e.get('text') or ''}".lower()
        fams = _blob_family_hits(blob)
        if fams and not any(f in allowed for f in fams):
            continue
        kept.append(e)
    return kept or entries


def _kb_examples_for_topic(question: str, kb_entries: list[dict], limit: int = 3) -> list[str]:
    keys = _question_topic_keys(question)
    blocked = _blocked_sibling_terms(question)
    out: list[str] = []
    for e in kb_entries:
        blob = f"{e.get('title') or ''} {e.get('source') or ''} {e.get('example') or ''}"
        if blocked and _blob_has_blocked(blob, blocked):
            allowed_terms = _allowed_family_terms(question)
            if not (allowed_terms and any(t in blob.lower() for t in allowed_terms)):
                continue
        if keys and not _text_matches_topic(blob, keys):
            continue
        if e.get("example"):
            out.append(str(e["example"])[:400])
        if len(out) >= limit:
            break
    return out


def _lookup_fast_path(question: str, client: QdrantClient) -> AgentResponse | None:
    hints = bot_name_hints(question)
    if not hints:
        return None
    chunks = retrieve(question, client, top_k=5, filter_dict={"type": "bot"})
    chunks = prune_lookup_chunks(question, chunks, "lookup")
    if not chunks or chunks[0]["metadata"].get("type") != "bot":
        return None
    bot_name = (chunks[0]["metadata"].get("bot_name") or "").lower()
    if not all(h in bot_name for h in hints):
        return None
    rows = parse_bot_param_table(chunks[0].get("text", ""))
    if not rows:
        return None
    required_rows = [r for r in rows if r["required"]]
    optional_rows = [r for r in rows if not r["required"]]
    lines = ["The bot takes the following parameters:"]
    for r in required_rows + optional_rows:
        req = " *" if r["required"] else ""
        desc = r["description"] or ""
        if r["default"]:
            desc = (desc + f" (default {r['default']})").strip()
        lines.append(f"- **{r['name']}**{req}: {desc}".strip())
    example = None
    text = chunks[0].get("text", "")
    if "Example" in text:
        example = text.split("Example", 1)[-1][:300].strip()
    examples = [example] if example else []
    return AgentResponse(
        answer="\n".join(lines).strip(),
        sources=_sources_from_chunks(chunks[:1]),
        sufficient=True,
        examples=examples,
        gaps=[],
        scope="in_scope",
        used_inference=False,
    )


def answer(question: str, client: QdrantClient | None = None) -> AgentResponse:
    if client is None:
        client = QdrantClient(path=QDRANT_DIR)

    t0 = time.perf_counter()
    timing = {
        "scope_ms": 0.0,
        "retrieve_ms": 0.0,
        "generate_ms": 0.0,
        "critique_ms": 0.0,
        "total_ms": 0.0,
        "llm_calls": 0,
    }

    def _ms(since: float) -> float:
        return round((time.perf_counter() - since) * 1000, 1)

    # Structural bot-param fast path
    fast = _lookup_fast_path(question, client)
    if fast is not None:
        timing["total_ms"] = _ms(t0)
        timing["llm_calls"] = 0
        fast.timing = timing
        return fast

    t_scope = time.perf_counter()
    planned = plan_scope_and_facets(question)
    timing["scope_ms"] = _ms(t_scope)
    timing["llm_calls"] += 1

    scope = planned["scope"]
    search_q = _strip_format_noise(planned.get("search_query") or question)
    search_q = _strip_format_noise(search_q)
    if not search_q.strip():
        search_q = _strip_format_noise(question) or question

    # Safety: do not out-of-scope clear Fabrix product questions (unless known traps)
    if scope == "out_of_scope":
        qlow = (question or "").lower()
        trap = any(
            t in qlow
            for t in (
                "sla", "refund", "subscription", "billing", "invoice",
                "vpn", "jump host", "encryption key", "mtls", "mTLS".lower(),
                "private ingress", "enterprise quote", "list price", "cost model",
                "discount", "cake", "world series",
            )
        )
        fabrixish = any(
            m in qlow
            for m in (
                "fabrix", "rda", "cfx", "cfxql", "agentic", "extension",
                "pipeline", "pstream", "dataset", "dashboard", "bot",
            )
        )
        if fabrixish and not trap:
            scope = "related" if _is_vague_ops_question(question) else "in_scope"
            logger.info("scope latch: out_of_scope→%s for Fabrix product question", scope)

    # Broad ops overviews need related/inference treatment
    if scope == "in_scope" and _is_vague_ops_question(question):
        scope = "related"
        logger.info("scope latch: in_scope→related for vague ops question")

    # Compare/recommend asks need related + inference labeling
    if scope == "in_scope" and _is_synthesis_question(question):
        scope = "related"
        logger.info("scope latch: in_scope→related for synthesis/compare question")

    # Force commercial / private-infra traps out of scope even if classifier misfires
    qlow_trap = (question or "").lower()
    force_oos = any(
        t in qlow_trap
        for t in (
            "enterprise quote", "list price", "cost model", "discount",
            "contractual", "p1 support sla", "refund", "subscription",
            "private vpn", "jump host", "encryption key", "mtls",
            "private ingress",
        )
    )
    if force_oos and scope != "out_of_scope":
        logger.info("scope latch: %s→out_of_scope for commercial/private trap", scope)
        scope = "out_of_scope"

    logger.info("scope=%s reason=%s search_query=%s", scope, planned.get("reason"), search_q)

    if scope == "out_of_scope":
        msg = (
            "I couldn't find that in the Fabrix / RDA documentation. "
            "This question looks outside the public docs scope."
        )
        _log_gap(question, scope, ["Question classified as out of scope for public Fabrix docs"], True)
        timing["total_ms"] = _ms(t0)
        return AgentResponse(
            answer=msg,
            sources=[],
            sufficient=True,
            examples=[],
            gaps=["Out of scope for public Fabrix/CFX documentation"],
            scope=scope,
            used_inference=False,
            timing=timing,
        )

    use_facets = (
        bool(planned.get("needs_facets"))
        or scope == "related"
        or _is_vague_ops_question(question)
        or _is_synthesis_question(question)
    )
    facet_plan: dict = {
        "facets": list(planned.get("facets") or []),
        "search_queries": list(planned.get("search_queries") or [search_q]),
        "primary_objects": list(planned.get("primary_objects") or []),
    }

    # Product bias: prepend product queries when ≥1 integration family is named
    fam_hits = _integration_family_hits(question)
    if fam_hits:
        queries0 = list(facet_plan["search_queries"] or [])
        for fam in reversed(fam_hits):
            prod_q = f"{fam} Fabrix datasource bots integration"
            if prod_q not in queries0:
                queries0 = [prod_q] + queries0
        facet_plan["search_queries"] = queries0
        objs = list(facet_plan.get("primary_objects") or [])
        for fam in fam_hits:
            for term in _family_match_terms(fam):
                if term not in objs:
                    objs.insert(0, term)
        facet_plan["primary_objects"] = objs[:8]
        if len(fam_hits) == 1:
            logger.info("single_product mode dominant=%s", fam_hits[0])
        else:
            logger.info("multi_product mode families=%s", fam_hits)
    elif _is_agentic_question(question):
        queries0 = list(facet_plan["search_queries"] or [])
        for seed in (
            "Fabio Copilot AI Fabric",
            "AI Fabric Copilot pipeline audit",
        ):
            if seed not in queries0:
                queries0 = [seed] + queries0
        facet_plan["search_queries"] = queries0
        logger.info("agentic topic mode retrieve bias")

    # CFXQL / operator asks: always prepend topic seeds (format stress often empties search_q)
    qlow_seed = (question or "").lower()
    if "cfxql" in qlow_seed or (
        "operator" in qlow_seed and ("full" in qlow_seed or "restricted" in qlow_seed)
    ):
        queries0 = list(facet_plan["search_queries"] or [])
        for seed in (
            "Full CFXQL operators operators list",
            "CFXQL Full vs Restricted operators",
            "CFXQL operators = AND OR",
        ):
            if seed not in queries0:
                queries0 = [seed] + queries0
        facet_plan["search_queries"] = queries0
        objs = list(facet_plan.get("primary_objects") or [])
        for term in ("cfxql", "Full CFXQL", "operator", "="):
            if term not in objs:
                objs.insert(0, term)
        facet_plan["primary_objects"] = objs[:8]
        logger.info("cfxql topic retrieve bias")

    t_ret = time.perf_counter()
    if use_facets:
        queries = facet_plan["search_queries"] or [search_q]
        queries = [_strip_format_noise(q) or q for q in queries]
        if search_q not in queries:
            queries = [search_q] + queries
        logger.info(
            "facet_plan facets=%s queries=%s objects=%s",
            facet_plan.get("facets"),
            queries,
            facet_plan.get("primary_objects"),
        )
        kb_entries, chunks = retrieve_multi_facet(question, client, queries)
    else:
        kb_entries = retrieve_kb(search_q, top_k=8)
        chunks = []
        if len(kb_entries) < 3:
            logger.info("KB thin (%s hits); falling back to doc chunks", len(kb_entries))
            chunks = retrieve(search_q, client, top_k=6, filter_dict=None)
        # Escalate thin in_scope lookups to multi-facet (format-stressed / mis-scoped compares)
        if len(kb_entries) < 2 and len(chunks) < 2:
            logger.info("in_scope retrieve thin; escalating to multi-facet")
            use_facets = True
            t_fac = time.perf_counter()
            facet_plan = plan_facets(_strip_format_noise(question) or question)
            timing["scope_ms"] += _ms(t_fac)
            timing["llm_calls"] += 1
            queries = facet_plan["search_queries"] or [search_q]
            queries = [_strip_format_noise(q) or q for q in queries]
            if search_q not in queries:
                queries = [search_q] + queries
            kb_entries, chunks = retrieve_multi_facet(question, client, queries)

    if not kb_entries and not chunks:
        kb_entries = retrieve_kb(question, top_k=8)
        chunks = retrieve(question, client, top_k=8, filter_dict=None)

    if not kb_entries and not chunks:
        topic_q = _strip_format_noise(question) or question
        kb_entries = retrieve_kb(topic_q, top_k=8)
        chunks = retrieve(topic_q, client, top_k=8, filter_dict=None)

    # Last-chance topic seeds when format-stressed queries still miss
    if not kb_entries and not chunks:
        qlow = (question or "").lower()
        seeds: list[str] = []
        if "pstream" in qlow or "persistent stream" in qlow:
            seeds.append("persistent streams pstreams RDA Fabric")
        if "dataset" in qlow:
            seeds.append("datasets RDA Fabric")
        if "servicenow" in qlow or " sn " in f" {qlow} ":
            seeds.append("ServiceNow integration Fabrix")
        if "cfxql" in qlow or "full cfxql" in qlow or "operator" in qlow:
            seeds.append("Full CFXQL operators operators list")
            seeds.append("CFXQL Full vs Restricted")
        if "datadog" in qlow:
            seeds.append("Datadog Fabrix bots datasource")
        if "nagios" in qlow:
            seeds.append("Nagios XI Fabrix bots datasource")
        for seed in seeds:
            kb_entries = retrieve_kb(seed, top_k=8)
            if kb_entries:
                break
            chunks = retrieve(seed, client, top_k=6, filter_dict=None)
            if chunks:
                break
    timing["retrieve_ms"] = _ms(t_ret)

    if not kb_entries and not chunks:
        _log_gap(question, scope, ["No KB or doc chunks retrieved"], True)
        timing["total_ms"] = _ms(t0)
        return AgentResponse(
            answer="I couldn't find that in the documentation.",
            sources=[],
            sufficient=False,
            examples=[],
            gaps=["No matching knowledge-base or documentation entries were retrieved"],
            scope=scope,
            used_inference=False,
            timing=timing,
        )

    primary_objects = list(facet_plan.get("primary_objects") or [])
    kb_entries = _filter_kb_entries_to_families(kb_entries, question)
    kb_entries = _rank_entries_for_product(kb_entries, question)
    prompt = build_kb_prompt(
        question, kb_entries, chunks, scope, primary_objects=primary_objects
    )
    t_gen = time.perf_counter()
    raw_answer = generate(prompt) or ""
    timing["llm_calls"] += 1
    parsed = parse_unified_response(raw_answer)
    answer_text = parsed["answer"]
    examples = parsed["examples"]
    gaps = parsed["gaps"]
    used_inference = parsed["used_inference"]
    inferred_summary = parsed["inferred_summary"]

    abstained = _looks_like_abstention(answer_text)

    # Retry when abstained with context, or synthesis/related missing inference label
    need_retry = (kb_entries or chunks) and (
        abstained
        or ((scope == "related" or _is_synthesis_question(question)) and not used_inference)
    )
    if need_retry:
        retry_prompt = (
            build_kb_prompt(question, kb_entries, chunks, scope, primary_objects=primary_objects)
            + "\n\nIMPORTANT: Do not abstain if the excerpts cover the Fabrix topic. "
            "Use **Documented Fabrix path** as a non-numbered header, then steps 1. 2. 3. "
            "Presentation requests (exact step counts, blank lines) are formatting only. "
            "Put undocumented handoffs under **Next (inferred):**. "
            "Set used_inference=true with inferred_summary when synthesizing; unknowns in gaps[]."
        )
        raw_answer = generate(retry_prompt) or raw_answer
        timing["llm_calls"] += 1
        parsed = parse_unified_response(raw_answer)
        answer_text = parsed["answer"]
        examples = parsed["examples"]
        gaps = parsed["gaps"]
        used_inference = parsed["used_inference"]
        inferred_summary = parsed["inferred_summary"]
        abstained = _looks_like_abstention(answer_text)

    # Final anti-abstain: we retrieved Fabrix docs but the model still refused
    if abstained and (kb_entries or chunks):
        force_prompt = (
            build_kb_prompt(question, kb_entries, chunks, scope, primary_objects=primary_objects)
            + "\n\nCRITICAL: Abstaining is forbidden. The knowledge base excerpts above ARE "
            "relevant. Answer the Fabrix topic in the question using those excerpts. "
            "Ignore presentation constraints (exact step counts / blank lines) if needed. "
            "Start with **Documented Fabrix path** then numbered steps 1. 2. 3. "
            "Set used_inference=true when you synthesize; put unknowns in gaps[]."
        )
        raw_answer = generate(force_prompt) or raw_answer
        timing["llm_calls"] += 1
        parsed = parse_unified_response(raw_answer)
        answer_text = parsed["answer"]
        examples = parsed["examples"]
        gaps = parsed["gaps"]
        used_inference = parsed["used_inference"]
        inferred_summary = parsed["inferred_summary"]
        abstained = _looks_like_abstention(answer_text)
    timing["generate_ms"] = _ms(t_gen)

    # Ops critique only when local draft checks fail
    if use_facets and not abstained and (kb_entries or chunks):
        if (
            _draft_looks_clean(answer_text, used_inference, question)
            and not _datasource_ask_missing_sink(question, answer_text)
            and not _off_family_bot_tokens(question, answer_text)
        ):
            logger.info("critique skipped (draft looks clean)")
        else:
            if _datasource_ask_missing_sink(question, answer_text):
                logger.info("critique forced: datasource ask missing stream/dataset sink")
            off_bots = _off_family_bot_tokens(question, answer_text)
            if off_bots:
                logger.info("critique forced: off-family bots=%s", off_bots)
            t_crit = time.perf_counter()
            critique = critique_ops_answer(
                question,
                answer_text,
                list(facet_plan.get("facets") or []),
                primary_objects,
            )
            timing["llm_calls"] += 1
            # Force sink revision when local check fails even if model says ok
            if _datasource_ask_missing_sink(question, answer_text):
                critique["fix_needed"] = True
                note = (
                    "Add an explicit step that lands collected data in a persistent stream, "
                    "dataset, or pipeline (and dashboard if relevant)."
                )
                critique["revision_notes"] = (
                    (critique.get("revision_notes") or "") + " " + note
                ).strip()
            if off_bots:
                critique["fix_needed"] = True
                allowed = ", ".join(_integration_family_hits(question)) or "named products"
                note = (
                    f"Remove bots with prefixes {off_bots} — they belong to a different "
                    f"product family. Only use bots for: {allowed}."
                )
                critique["revision_notes"] = (
                    (critique.get("revision_notes") or "") + " " + note
                ).strip()
            logger.info(
                "critique fix_needed=%s overclaim=%s notes=%s",
                critique.get("fix_needed"),
                critique.get("overclaim"),
                (critique.get("revision_notes") or "")[:160],
            )
            if critique.get("fix_needed") and critique.get("revision_notes"):
                rev_prompt = build_kb_prompt(
                    question,
                    kb_entries,
                    chunks,
                    scope,
                    primary_objects=primary_objects,
                    revision_notes=critique["revision_notes"],
                )
                raw_answer = generate(rev_prompt) or raw_answer
                timing["llm_calls"] += 1
                parsed = parse_unified_response(raw_answer)
                answer_text = parsed["answer"]
                examples = parsed["examples"] or examples
                gaps = parsed["gaps"] or gaps
                used_inference = parsed["used_inference"] or used_inference
                inferred_summary = parsed["inferred_summary"] or inferred_summary
                abstained = _looks_like_abstention(answer_text)
            timing["critique_ms"] = _ms(t_crit)

    # Local bot-fidelity revision even when critique path was skipped (e.g. in_scope)
    off_bots_final = _off_family_bot_tokens(question, answer_text)
    if off_bots_final and not abstained and (kb_entries or chunks):
        allowed = ", ".join(_integration_family_hits(question)) or "named products"
        note = (
            f"Remove bots with prefixes {off_bots_final} — wrong product family. "
            f"Only use bots for: {allowed}."
        )
        logger.info("bot fidelity revision: off-family bots=%s", off_bots_final)
        rev_prompt = build_kb_prompt(
            question,
            kb_entries,
            chunks,
            scope,
            primary_objects=primary_objects,
            revision_notes=note,
        )
        t_bot = time.perf_counter()
        raw_answer = generate(rev_prompt) or raw_answer
        timing["llm_calls"] += 1
        timing["critique_ms"] = timing.get("critique_ms", 0.0) + _ms(t_bot)
        parsed = parse_unified_response(raw_answer)
        answer_text = parsed["answer"]
        examples = parsed["examples"] or examples
        gaps = parsed["gaps"] or gaps
        used_inference = parsed["used_inference"] or used_inference
        inferred_summary = parsed["inferred_summary"] or inferred_summary
        abstained = _looks_like_abstention(answer_text)

    answer_text = polish_answer_text(answer_text)
    grounded = bool(kb_entries or chunks) and not abstained

    if grounded:
        examples = _filter_examples_to_topic(examples, question, kb_entries)
        if not examples:
            examples = _kb_examples_for_topic(question, kb_entries, limit=3)
    else:
        examples = []
        used_inference = False
        inferred_summary = ""
        answer_text = answer_text.replace(INFERENCE_DISCLOSURE, "").strip()

    sources = []
    if grounded:
        sources = _sources_from_kb(kb_entries) if kb_entries else []
        if chunks:
            sources.extend(_sources_from_chunks(chunks))
        seen = set()
        deduped = []
        for s in sources:
            u = s.get("url") or s.get("title")
            if u in seen:
                continue
            seen.add(u)
            deduped.append(s)
        sources = _filter_sources_to_topic(deduped[:12], question)

    if abstained and not gaps:
        gaps = ["No direct answer found in the retrieved documentation"]

    if used_inference and grounded and INFERENCE_DISCLOSURE not in answer_text:
        answer_text = f"{answer_text}\n\n{INFERENCE_DISCLOSURE}".strip()

    answer_text = polish_answer_text(answer_text)
    _log_gap(question, scope, gaps, abstained)
    timing["total_ms"] = _ms(t0)
    logger.info(
        "timing total_ms=%s llm_calls=%s scope=%s retrieve=%s generate=%s critique=%s",
        timing["total_ms"],
        timing["llm_calls"],
        timing["scope_ms"],
        timing["retrieve_ms"],
        timing["generate_ms"],
        timing["critique_ms"],
    )

    return AgentResponse(
        answer=answer_text,
        sources=sources,
        sufficient=not abstained,
        examples=examples,
        gaps=gaps,
        scope=scope,
        used_inference=used_inference and grounded,
        inferred_summary=inferred_summary if (used_inference and grounded) else "",
        timing=timing,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python3 src/agent.py \"your question\"")
        sys.exit(1)
    result = answer(" ".join(sys.argv[1:]))
    print(result.answer)
    if result.used_inference:
        print(f"\n[used_inference] {result.inferred_summary or '(yes)'}")
    if result.examples:
        print("\nExamples:")
        for ex in result.examples:
            print(f"  - {ex[:200]}")
    if result.gaps:
        print("\nGaps:")
        for g in result.gaps:
            print(f"  - {g}")
    if result.sources:
        print("\nSources:")
        for source in result.sources:
            print(f"  [{source['title']}]({source['url']})")
    else:
        print("\nSources: (none)")
    if result.timing:
        t = result.timing
        print(
            f"\n[timing] total={t.get('total_ms')}ms llm_calls={t.get('llm_calls')} "
            f"scope={t.get('scope_ms')}ms retrieve={t.get('retrieve_ms')}ms "
            f"generate={t.get('generate_ms')}ms critique={t.get('critique_ms')}ms"
        )
