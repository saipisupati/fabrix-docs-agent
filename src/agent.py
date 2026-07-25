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
from live_docs import live_install_kb_entries
from page_expand import expand_context, pages_to_kb_entries
from query_qdrant import (
    generate,
    prune_lookup_chunks,
    retrieve,
)
from bot_lookup import (
    best_chunk_lookup,
    bot_family_hints,
    bot_operation_hints,
    catalog_source_dict,
    extract_example_snippet,
    format_param_answer,
    is_bot_param_lookup,
    kb_source_dict,
    lookup_bot_params_from_catalog,
    lookup_bot_params_from_kb,
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


def _is_exhaustive_ask(question: str) -> bool:
    """Asks for complete/exact catalogs that docs often only partially cover."""
    q = (question or "").lower()
    return any(
        m in q
        for m in (
            "exact complete", "complete grammar", "every operator", "all edge cases",
            "full grammar", "exhaustive", "every single",
        )
    )


def _is_capability_overclaim_ask(question: str) -> bool:
    """Asks the product to perform an end-to-end action that docs rarely promise."""
    q = (question or "").lower()
    actor = any(m in q for m in ("fabio", "copilot", "agentic"))
    action = any(
        m in q
        for m in (
            "rewrite", "for me end-to-end", "do it for me", "change production",
            "in production for me",
            "auto-remediate", "auto remediate", "autoremediate",
            "automatically remediate", "automatic remediation",
            "no human approval", "without human approval", "without a human",
            "with no human",
        )
    )
    return actor and action


def _agentic_honesty_fallback(question: str, existing_gaps: list[str] | None = None) -> tuple[str, list[str]]:
    """
    Deterministic honesty answer when the model abstains on Fabio/Copilot/Agentic
    capability asks despite retrieved docs (critique can re-introduce abstention).
    Generic family rule — not per-question.
    """
    text = (
        "Fabio Copilot is part of Fabrix Agentic AI. Documented building blocks include "
        "Personas, Toolsets, and Copilot workflows under ai_fabric. The public docs do "
        "not claim end-to-end auto-remediation of production outages with no human "
        "approval. Treat that as unsupported unless a cited excerpt says otherwise."
    )
    gaps = list(existing_gaps or [])
    if not gaps:
        gaps = [
            "End-to-end auto-remediation with no human approval is not documented in "
            "public Fabrix docs"
        ]
    return text, gaps


def _is_full_script_automation_ask(question: str) -> bool:
    """Asks for a turnkey script/playbook chaining an end-to-end automation."""
    q = (question or "").lower()
    wants_artifact = any(
        w in q
        for w in (
            "bash script", "shell script", "write me a script", "write a script",
            "write me a bash", "playbook", "automate", "automation script",
        )
    )
    end_to_end = any(
        w in q
        for w in ("end-to-end", "end to end", "e2e", "full script", "complete script")
    )
    return wants_artifact and (end_to_end or "automat" in q)


# Live-incident / host-playbook asks: distinct from day-2 how-to and wiring.
_INCIDENT_SYMPTOM_CUES: tuple[str, ...] = (
    "oom",
    "out of memory",
    "disk is full",
    "disk full",
    "cpu spike",
    "production is down",
    "prod is down",
    "on fire",
    "live incident",
)
_INCIDENT_URGENCY_CUES: tuple[str, ...] = (
    "right now",
    "urgent",
    "this live",
    "live incident",
)
_INCIDENT_REMEDIATION_CUES: tuple[str, ...] = (
    "playbook",
    "remediation",
    "remediate",
    "exact fix",
    "remediation steps",
    "remediation playbook",
    "walk me through fixing",
    "walk me through the remediation",
    "what's the exact fix",
    "what is the exact fix",
    "fix right now",
)
# Strong procedure language — general mem_limit / worker config pages do not count.
_INCIDENT_PROCEDURE_MARKERS: tuple[str, ...] = (
    "playbook",
    "runbook",
    "remediation guide",
    "remediation procedure",
    "incident response procedure",
    "incident response playbook",
    "troubleshoot oom",
    "troubleshooting oom",
    "disk full remediation",
    "cpu spike remediation",
)


def _is_live_incident_ask(question: str) -> bool:
    """
    Live-incident / "what do I do right now" remediation asks.

    Requires remediation/playbook framing PLUS symptom or urgency cues.
    Must not false-trigger on ordinary day-2 how-to (scale workers, rollback,
    scheduled-pipeline debug, ServiceNow incident field updates).
    """
    q = (question or "").lower()
    if not q.strip():
        return False
    # Ticketing "incident" how-tos are not live host remediation.
    if any(
        w in q
        for w in (
            "servicenow",
            "service now",
            "snowv2",
            "snow ",
            "pagerduty",
            "opsgenie",
            "jira",
        )
    ) and not any(s in q for s in _INCIDENT_SYMPTOM_CUES):
        return False
    has_remediation = any(c in q for c in _INCIDENT_REMEDIATION_CUES)
    if not has_remediation:
        return False
    has_symptom = any(c in q for c in _INCIDENT_SYMPTOM_CUES)
    has_urgency = any(c in q for c in _INCIDENT_URGENCY_CUES) or (
        "incident" in q and any(c in q for c in ("live", "right now", "urgent", "on fire"))
    )
    return has_symptom or has_urgency


def _live_incident_has_documented_procedure(
    question: str,
    kb_entries: list[dict] | None,
    chunks: list[dict] | None,
) -> bool:
    """
    True only when retrieved excerpts look like a named runbook/playbook for the
    scenario — not loosely related memory/worker/config pages.
    """
    blob = _retrieved_context_blob(kb_entries, chunks)
    if not blob.strip():
        return False
    if not any(m in blob for m in _INCIDENT_PROCEDURE_MARKERS):
        return False
    q = (question or "").lower()
    symptoms = [s for s in _INCIDENT_SYMPTOM_CUES if s in q]
    # If the ask names a symptom, require that symptom (or close token) in excerpts.
    if symptoms and not any(s in blob for s in symptoms):
        # Allow common alternate spellings for OOM
        if any(s in ("oom", "out of memory") for s in symptoms):
            if not any(t in blob for t in ("oom", "out of memory", "memory")):
                return False
        else:
            return False
    return True


def _live_incident_honesty(
    question: str,
    kb_entries: list[dict] | None = None,
    chunks: list[dict] | None = None,
    existing_gaps: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Refuse invented live-incident playbooks; point at related docs + on-call."""
    titles: list[str] = []
    for e in (kb_entries or [])[:4]:
        t = (e.get("title") or "").strip()
        if t and t.lower() not in {x.lower() for x in titles}:
            titles.append(t)
    for c in (chunks or [])[:4]:
        t = (c.get("title") or "").strip()
        if t and t.lower() not in {x.lower() for x in titles}:
            titles.append(t)
    if titles:
        related = (
            " Here's what IS documented that may be related: "
            + "; ".join(titles[:3])
            + "."
        )
    else:
        related = ""
    text = (
        "I don't see a documented playbook for this specific live-incident scenario "
        "in the Fabrix / RDA public docs."
        f"{related} "
        "Related pages (if any) are general reference, not a step-by-step incident "
        "remediation procedure. For a live incident, verify with your on-call process "
        "/ support runbook rather than treating this chat as an executable playbook."
    )
    gaps = list(existing_gaps or [])
    gap = (
        "No documented live-incident playbook/runbook for this scenario in public docs"
    )
    if gap.lower() not in " ".join(g.lower() for g in gaps):
        gaps.insert(0, gap)
    return text, gaps[:8]


def _answer_has_turnkey_script_risk(answer: str) -> bool:
    """Detect invented ready-to-run scripts / credential placeholders."""
    text = answer or ""
    low = text.lower()
    if re.search(r"(?m)^#!/bin/(?:ba)?sh\b", text):
        return True
    if "$password" in low or "${password}" in low:
        return True
    if re.search(r"(?i)\b(?:password|passwd)\s*=", text):
        return True
    if "admin:changeme" in low:
        return True
    return False


_FABRIX_COMPARE_SELF = frozenset(
    {
        "fabrix", "rda", "rdaf", "cloudfabrix", "cfx", "cfxql", "oia", "cfxoia",
        "edge", "studio", "worker", "pipeline", "pipelines", "dataset", "datasets",
        "pstream", "pstreams", "bot", "bots", "dashboard", "dashboards", "generic",
        "etl", "tool", "architecture", "platform", "fabric", "how", "what", "does",
        "terms", "tooling", "stack", "system", "product", "products",
    }
)


def _asks_version_specificity(question: str) -> bool:
    """Question asks for a concrete version number, not just whether a topic exists."""
    q = (question or "").lower()
    return bool(
        re.search(r"\b(?:what|which)\s+version\b", q)
        or re.search(r"\bversion\s+(?:does|is|of|do)\b", q)
        or "if so, what version" in q
        or "what k8s version" in q
        or "what kubernetes version" in q
    )


def _external_comparison_targets(question: str) -> list[str]:
    """
    Named third-party / competitor targets when the user asks to bake off Fabrix/RDA
    against an external tool. Does NOT fire on Fabrix-internal compares
    (e.g. Splunk vs Elastic for ingestion, ServiceNow CMDB vs Ticketing, pstream vs dataset).
    """
    q = question or ""
    qlow = q.lower()
    # Internal product-vs-product framed inside Fabrix — not a rival bake-off.
    if re.search(
        r"\bcompare\b.+\b(?:vs\.?|versus)\b.+\b(?:in|for|within|on|with)\s+fabrix\b",
        qlow,
    ):
        return []
    if re.search(r"\b(?:vs\.?|versus)\b.+\bmodules?\b.+\bfabrix\b", qlow):
        return []
    # Must be Fabrix/RDA as the subject of the comparison, or "generic … like X".
    fabrix_subject = bool(
        re.search(r"\b(?:fabrix\.?ai|fabrix|rda\s*fabric|rdaf|cloudfabrix)\b", qlow)
    )
    generic_like = bool(
        re.search(
            r"generic\s+\w+\s+(?:tool|platform|system)\s+like\s+[A-Z]",
            q,
        )
    )
    # Fabrix only as install/destination context before a separate A-vs-B compare
    # (e.g. "wire BMC into Fabrix, and also compare bots to ServiceNow") is NOT a rival bake-off.
    fabrix_as_destination_only = bool(
        re.search(
            r"\b(?:into|in|for|with|on)\s+(?:fabrix\.?ai|fabrix|rda\s*fabric|rdaf)\b"
            r".{0,100}\b(?:compar(?:e|es|ed|ison)|versus|\bvs\.?\b)\b",
            qlow,
        )
    ) and not bool(
        re.search(
            r"\b(?:how\s+does|how\s+do)\s+(?:fabrix\.?ai|fabrix|rda\s*fabric|rdaf)\b",
            qlow,
        )
        or re.search(
            r"\b(?:fabrix\.?ai|fabrix|rda\s*fabric|rdaf)\s+"
            r"(?:compar(?:e|es|ed|ison)|versus|\bvs\.?\b)\b",
            qlow,
        )
    )
    if fabrix_as_destination_only and not generic_like:
        return []
    compare_to_fabrix = bool(
        re.search(
            r"(?:how\s+does|how\s+do)\s+(?:fabrix\.?ai|fabrix|rda\s*fabric|rdaf)\b"
            r".{0,60}(?:compar(?:e|es|ed|ison)|versus|\bvs\.?\b|differ)",
            qlow,
        )
        or re.search(
            r"\b(?:fabrix\.?ai|fabrix|rda\s*fabric|rdaf)\s+"
            r"(?:compar(?:e|es|ed|ison)|versus|\bvs\.?\b)\b.{0,40}"
            r"(?:to|with|against|like)\b",
            qlow,
        )
        or re.search(
            r"\b(?:fabrix\.?ai|fabrix|rda\s*fabric|rdaf)\s+(?:vs\.?|versus)\s+",
            qlow,
        )
        or re.search(
            r"\bcompar(?:e|es|ed|ison)\s+(?:fabrix\.?ai|fabrix|rda\s*fabric|rdaf)\b"
            r".{0,40}(?:to|with|against|like|vs\.?|versus)\b",
            qlow,
        )
        or re.search(
            r"\b(?:fabrix\.?ai|fabrix|rda\s*fabric|rdaf)\b.{0,20}"
            r"(?:better|worse)\s+than\b",
            qlow,
        )
    )
    if not (generic_like or (fabrix_subject and compare_to_fabrix)):
        return []

    targets: list[str] = []
    for m in re.finditer(
        r"(?:like|to|vs\.?|versus|than|with|against|from)\s+"
        r"(?:a\s+|an\s+|the\s+)?"
        r"(?:generic\s+\w+\s+(?:tool|platform|system)\s+like\s+)?"
        r"([A-Z][A-Za-z0-9+._-]{2,})",
        q,
    ):
        name = m.group(1)
        if name.lower() in _FABRIX_COMPARE_SELF:
            continue
        targets.append(name)
    seen: set[str] = set()
    out: list[str] = []
    for t in targets:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def _version_detail_in_excerpts(question: str, blob: str) -> bool:
    """True if excerpts appear to state a concrete version for the asked topic."""
    if not blob:
        return False
    low = blob.lower()
    q = (question or "").lower()
    topic_keys: list[str] = []
    if any(k in q for k in ("kubernetes", "k8s")):
        topic_keys.extend(["kubernetes", "k8s"])
    # Generic: noun before "version" / after "version of"
    for m in re.finditer(r"\bversion\s+of\s+([a-z0-9._-]+)", q):
        topic_keys.append(m.group(1))
    for m in re.finditer(r"\b([a-z0-9._-]{3,})\s+version\b", q):
        topic_keys.append(m.group(1))
    topic_keys = [k for k in topic_keys if k not in ("what", "which", "the", "a", "an")]
    if not topic_keys:
        # Fall back: any explicit "runs on X version N" style claim
        return bool(
            re.search(r"\bversion\s*(?:is|=|:)?\s*v?\d+\.\d+", low)
            and re.search(r"\b(?:runs on|requires|based on)\b.{0,40}\bversion\b", low)
        )
    for key in topic_keys:
        # version near topic, or topic + semver-ish
        if re.search(
            rf"\b{re.escape(key)}\b.{{0,60}}\bversion\b.{{0,20}}v?\d+\.\d+",
            low,
        ):
            return True
        if re.search(
            rf"\bversion\b.{{0,40}}\b{re.escape(key)}\b.{{0,20}}v?\d+\.\d+",
            low,
        ):
            return True
        if re.search(rf"\b{re.escape(key)}\s+v?\d+\.\d+", low):
            return True
    return False


def _specificity_detail_missing(
    question: str,
    kb_entries: list[dict] | None,
    chunks: list[dict] | None,
) -> dict | None:
    """
    Topic may be documented while a requested specific (version / rival comparison)
    is not. Returns a gap descriptor when that specific is absent from excerpts.
    """
    blob = _retrieved_context_blob(kb_entries, chunks)
    if _asks_version_specificity(question) and not _version_detail_in_excerpts(question, blob):
        return {"kind": "version", "targets": [], "detail": "specific version number"}
    targets = _external_comparison_targets(question)
    if targets:
        missing = [t for t in targets if not _term_grounded_in_blob(t, blob)]
        # Even if the name appears once, require the docs to discuss it — a stray
        # mention without the name is still a missing comparison; if grounded but
        # question asks compare, still treat as missing unless comparison language exists.
        if missing:
            return {
                "kind": "comparison",
                "targets": missing,
                "detail": f"direct comparison to {', '.join(missing)}",
            }
        # Name appears in corpus: only treat as missing if no compare language near it
        low = blob.lower()
        for t in targets:
            tl = t.lower()
            if not re.search(
                rf"(?:compar|versus|vs\.?|unlike|differ|alternative to).{{0,40}}{re.escape(tl)}"
                rf"|{re.escape(tl)}.{{0,40}}(?:compar|versus|vs\.?|unlike|differ)",
                low,
            ):
                return {
                    "kind": "comparison",
                    "targets": targets,
                    "detail": f"direct comparison to {', '.join(targets)}",
                }
    return None


def _answer_acknowledges_specificity_gap(answer: str) -> bool:
    low = (answer or "").lower()
    return any(
        m in low
        for m in (
            "not specified",
            "not stated",
            "not documented",
            "doesn't specify",
            "does not specify",
            "do not specify",
            "don't specify",
            "isn't in the public docs",
            "is not in the public docs",
            "aren't in the public docs",
            "are not in the public docs",
            "no direct comparison",
            "direct comparison isn't",
            "direct comparison is not",
            "comparison isn't",
            "comparison is not",
            "don't compare",
            "do not compare",
            "will not invent",
            "won't invent",
            "public docs do not",
            "public docs don't",
        )
    )


def _answer_invents_specificity(answer: str, gap: dict) -> bool:
    """Detect invented versions or rival bake-off frameworks for a specificity gap."""
    low = (answer or "").lower()
    kind = (gap or {}).get("kind")
    if kind == "version":
        if re.search(r"\bkubernetes version (?:is|=)\s*v?\d+", low):
            return True
        if re.search(r"\bruns on (?:k8s|kubernetes) version\b", low):
            return True
        if re.search(r"\bversion 1\.\d+", low):
            return True
        if re.search(r"\b(?:kubernetes|k8s)\s+v?\d+\.\d+", low) and not _answer_acknowledges_specificity_gap(
            answer
        ):
            return True
        return False
    if kind == "comparison":
        for t in gap.get("targets") or []:
            tl = (t or "").lower()
            if not tl or tl not in low:
                continue
            if any(
                p in low
                for p in (
                    f"unlike {tl}",
                    f"whereas {tl}",
                    f"{tl} uses",
                    f"{tl} primarily",
                    f"{tl} focuses",
                    f"while {tl}",
                    f"compared to {tl}",
                    f"in contrast to {tl}",
                    f"in contrast, {tl}",
                    f"tools like {tl}",
                    f"tool like {tl}",
                    f"for {tl}, which",
                    f"outlined for {tl}",
                )
            ):
                return True
            if re.search(
                rf"\b{re.escape(tl)}\b.{{0,90}}\b(?:uses|focuses|primarily|designed|dags?|orchestrat|batch processing|task scheduler)\b",
                low,
            ):
                return True
            if re.search(
                rf"(?:unlike|whereas|contrasts?\s+with|compared\s+to|like)\s+.{{0,50}}\b{re.escape(tl)}\b",
                low,
            ):
                return True
            if re.search(
                rf"traditional\b.{{0,60}}\blike\s+{re.escape(tl)}\b",
                low,
            ):
                return True
            if ("in contrast" in low or "key differences" in low or "differ significantly" in low):
                return True
            if low.count(tl) >= 2 and re.search(
                rf"\b{re.escape(tl)}\b.{{0,80}}\b(?:focus|task|dag|batch|schedul|orchestr)\b",
                low,
            ):
                return True
        return False
    return False


def _specificity_honesty_fallback(
    question: str,
    gap: dict,
    existing_gaps: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Topic-documented ≠ detail-documented: state topic, refuse invented specifics."""
    kind = (gap or {}).get("kind")
    targets = gap.get("targets") or []
    if kind == "version":
        text = (
            "Fabrix public docs cover related topics (for example Kubernetes integration, "
            "inventory bots, and/or K8s deployment paths when those pages are retrieved). "
            "That does **not** mean every specific detail is documented. "
            "The public docs do **not specify** the exact version number you asked for "
            "(e.g. which Kubernetes version Fabrix runs on or requires). "
            "Use the documented integration/deployment material for what *is* stated; "
            "treat the missing version as unspecified rather than inventing one."
        )
        gap_line = "Requested specific version number is not specified in the public docs"
    else:
        named = ", ".join(targets) if targets else "that third-party tool"
        text = (
            "Public Fabrix docs describe RDA Fabric's own architecture (classic stack: "
            "bots → CFXQL → pipelines → datasets/pstreams → dashboards/integrations, "
            "plus Agentic pieces when relevant). "
            f"They do **not** include a direct architectural comparison to {named}. "
            "I will describe Fabrix's documented approach only and will **not** invent a "
            "rival bake-off framework (unlike/whereas claims, DAG vs pipeline tables, etc.). "
            "A direct comparison isn't in the public docs."
        )
        gap_line = f"Direct comparison to {named} is not documented in public Fabrix docs"
    gaps = list(existing_gaps or [])
    if gap_line.lower() not in " ".join(g.lower() for g in gaps):
        gaps = [gap_line] + gaps
    return text, gaps


def _full_script_honesty_fallback(
    question: str,
    existing_gaps: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Refuse turnkey scripts; point users at documented steps + manual verification."""
    text = (
        "**Documented Fabrix path** (not a turnkey script)\n"
        "I will not invent a ready-to-run bash/playbook that chains an end-to-end "
        "upgrade/automation with placeholder versions, paths, or credentials.\n\n"
        "Use the individual documented commands/steps from the RDAF upgrade / "
        "installation guides in order (CLI install/upgrade, backup/registry as "
        "documented, infra/platform/worker/app status checks). Exact values "
        "(version tags, hostnames, paths) must come from your environment and "
        "release notes — verify each step manually before production use.\n\n"
        "**Next (inferred):** Operator runbooks and site-specific sequencing may "
        "differ; treat any automation you build as requiring human verification."
    )
    gaps = list(existing_gaps or [])
    gap = (
        "Public docs list commands/procedures but do not provide a verified "
        "end-to-end turnkey upgrade script"
    )
    if gap not in gaps:
        gaps.insert(0, gap)
    return text, gaps[:8]


def _is_platform_install_ask(question: str) -> bool:
    """
    Platform / Studio / RDAF install, upgrade, prerequisites, VM or hardware sizing.
    Excludes asks that name a datasource product family (those stay on integration docs).
    """
    if _integration_family_hits(question):
        return False
    q = _normalize_question_typos(question or "").lower()
    # VM = virtual machine (sizing / host requirements)
    vmish = bool(
        re.search(r"\bvm\b", q)
        or "virtual machine" in q
        or "virtual machines" in q
    )
    installish = vmish or any(
        w in q
        for w in (
            "install", "installation", "deploy", "deployment", "prerequisite",
            "prerequisites", "hardware", "sizing",
            "system requirement", "system requirements",
            "software do i need", "software requirements",
            "requirements if i want", "requirements for",
            "upgrade", "upgrading", "upgrade the", "upgrade rdaf",
            # "update the platform" is day-2 ops language (ChatGPT-style), not only "upgrade"
            "update fabrix", "update the platform", "update rdaf",
            "update the fabrix", "updating the platform", "platform update",
        )
    )
    if not installish:
        return False
    return any(
        w in q
        for w in (
            "fabrix", "rdaf", "rda fabric", "rda studio", "platform",
            "agents", "worker", "on my own", "own vm", "own vms", "site",
        )
    )


def _is_platform_upgrade_ask(question: str) -> bool:
    """
    RDAF / platform upgrade or update workflow (CLI, backup, registry).

    Matches "upgrade" and day-2 "update … platform/fabrix/rdaf" phrasing.
    Excludes credential-only "update image repository" style asks when they
    do not also name platform upgrade/update.
    """
    if not _is_platform_install_ask(question):
        return False
    q = _normalize_question_typos(question or "").lower()
    if any(w in q for w in ("upgrade", "upgrading")):
        return True
    # "update fabrix / platform / rdaf" — not "update credentials" alone
    if "update" in q or "updating" in q:
        if any(
            w in q
            for w in (
                "platform", "fabrix", "rdaf", "rda fabric", "site",
                "deployment", "infra", "worker",
            )
        ):
            # Credential-rotation page alone is not a full platform upgrade ask
            if "credential" in q and "platform" not in q and "fabrix" not in q:
                return False
            return True
    return False


# Prefer the same docs ChatGPT opens: https://docs.fabrix.ai/installation_guides/
_INSTALL_GUIDE_PATH_MARKERS: tuple[str, ...] = (
    "installation_guides/",
    "docs.fabrix.ai/installation_guides",
    "rda_studio_installation",
    "rda studio installation",
)

_UPGRADE_DOC_MARKERS: tuple[str, ...] = (
    "rdaf_cli",
    "rdaf_k8s",
    "rdafcli",
    "rdaf_start_stop",
    "start_stop_ops",
    "oia_upgrades",
    "staging_infra_upgrade",
    "rdaf backup",
    "rdaf platform upgrade",
    "rdaf infra upgrade",
    "rdaf worker upgrade",
    "registry fetch",
)

# Strong CLI / lifecycle pages to pin first on upgrade asks
_UPGRADE_DOC_PRIMARY: tuple[str, ...] = (
    "rdaf_cli",
    "rdaf_k8s_cli",
    "rdaf_start_stop",
    "start_stop_ops",
    "oia_upgrades",
)

# Pages that steal upgrade retrieve (Studio laptop sizing / registry-cred rotation)
_UPGRADE_DOC_DEMOTE: tuple[str, ...] = (
    "update_image_repository",
    "update_docker_regsitry",  # typo in real doc filename
    "docker1.cloudfabrix",
    "rda_studio_installation",
)

_INSTALL_DOC_MARKERS: tuple[str, ...] = _INSTALL_GUIDE_PATH_MARKERS + (
    "rda_studio",
    "rdaf_cli",
    "rdaf_k8s",
    "deployment",
    "fsm_preupgrade",
    "fsm_installation",
) + _UPGRADE_DOC_MARKERS

_INSTALL_CONTENT_MARKERS: tuple[str, ...] = (
    "docker", "docker-compose", "docker compose", "cpu", "memory", "ram",
    "disk", "gib", "registry", "cloudfabrix.io", "rda studio", "pip3",
    "python 3", "ubuntu", "rhel", "ports 4222", "9443", "virtual machine",
)

_INTEGRATION_PREREQ_NOISE: tuple[str, ...] = (
    "servicenow", "service now", "qualys", "crowdstrike", "logrhythm",
    "zabbix", "pagerduty", "ms team", "microsoft teams", "splunk",
    "datadog", "prometheus", "bmc remedy",
)


def _entry_source_blob(entry: dict) -> str:
    return (
        f"{entry.get('title') or ''} {entry.get('source') or ''} "
        f"{entry.get('url') or ''} {entry.get('text') or ''}"
    ).lower()


def _looks_like_install_guide_path(blob: str) -> bool:
    """True when the hit is clearly under installation_guides (preferred index)."""
    low = (blob or "").lower()
    return any(m in low for m in _INSTALL_GUIDE_PATH_MARKERS)


def _looks_like_install_doc(blob: str) -> bool:
    low = (blob or "").lower()
    if _looks_like_install_guide_path(low):
        return True
    if any(m in low for m in _INSTALL_DOC_MARKERS):
        return True
    return sum(1 for m in _INSTALL_CONTENT_MARKERS if m in low) >= 2


def _looks_like_integration_prereq_noise(blob: str) -> bool:
    """Datasource 'Prerequisites' chunks that steal install/platform retrieve."""
    low = (blob or "").lower()
    if _looks_like_install_doc(low):
        return False
    if "datasource_integrations" in low or "bots/" in low:
        return any(n in low for n in _INTEGRATION_PREREQ_NOISE)
    # Untitled "**__2. Prerequisites__**" integration snippets
    if "prerequisite" in low and any(n in low for n in _INTEGRATION_PREREQ_NOISE):
        return True
    return False


def _filter_kb_entries_for_install_ask(entries: list[dict], question: str) -> list[dict]:
    if not entries or not _is_platform_install_ask(question):
        return entries
    kept = []
    for e in entries:
        blob = _entry_source_blob(e)
        if _looks_like_integration_prereq_noise(blob):
            continue
        kept.append(e)
    return kept or entries


def _looks_like_upgrade_doc(blob: str) -> bool:
    low = (blob or "").lower()
    if _looks_like_upgrade_demote(low):
        return False
    return any(m in low for m in _UPGRADE_DOC_MARKERS)


def _looks_like_upgrade_primary(blob: str) -> bool:
    low = (blob or "").lower()
    return any(m in low for m in _UPGRADE_DOC_PRIMARY)


def _looks_like_upgrade_demote(blob: str) -> bool:
    """Credential-rotation / Studio-install pages that should not lead upgrade answers."""
    low = (blob or "").lower()
    # Bare installation_guides index (Studio prereqs) — demote on upgrade unless CLI markers present
    if any(m in low for m in _UPGRADE_DOC_PRIMARY):
        return False
    if "update_image_repository" in low or "docker1.cloudfabrix" in low:
        return True
    if "rda_studio_installation" in low and "rdaf_cli" not in low:
        return True
    # Live/index Studio prereq pages often titled "RDA Studio Installation Guide"
    if "rda studio installation" in low and "rdaf" not in low.replace("rda studio", ""):
        return True
    if any(m in low for m in _UPGRADE_DOC_DEMOTE):
        return True
    return False


def _rank_entries_for_install_ask(entries: list[dict], question: str) -> list[dict]:
    """Rank: primary CLI upgrade docs, other upgrade docs, guides, then demoted/other."""
    if not entries or not _is_platform_install_ask(question):
        return entries
    upgrade_ask = _is_platform_upgrade_ask(question)
    primary, upgrade, guide, installish, demoted, other = [], [], [], [], [], []
    for e in entries:
        blob = _entry_source_blob(e)
        if upgrade_ask and _looks_like_upgrade_demote(blob):
            demoted.append(e)
        elif upgrade_ask and _looks_like_upgrade_primary(blob):
            primary.append(e)
        elif upgrade_ask and _looks_like_upgrade_doc(blob):
            upgrade.append(e)
        elif _looks_like_install_guide_path(blob):
            guide.append(e)
        elif _looks_like_install_doc(blob):
            installish.append(e)
        else:
            other.append(e)
    if upgrade_ask:
        return primary + upgrade + guide + installish + other + demoted
    return guide + installish + other + demoted


def _install_answer_missing_hardware(answer: str) -> bool:
    """Platform install answers should surface Docker/CPU/RAM-style prerequisites."""
    low = (answer or "").lower()
    return not any(
        w in low
        for w in (
            "docker", "cpu", "memory", "ram", "disk", "python", "gib",
            "compose", "registry",
        )
    )


def _upgrade_answer_missing_cli_path(answer: str) -> bool:
    """Upgrade answers should mention RDAF CLI / backup / registry workflow."""
    low = (answer or "").lower()
    has_cli = any(w in low for w in ("rdaf", "cli", "rdafk8s", "pip install"))
    has_ops = any(
        w in low
        for w in (
            "backup", "registry", "status", "infra", "platform upgrade",
            "worker upgrade", "docker2", "registry fetch",
        )
    )
    return not (has_cli and has_ops)


def _upgrade_answer_stuck_on_cred_or_studio(answer: str) -> bool:
    """True when answer leads with docker1 creds / Studio 8GB and skips CLI upgrade path."""
    low = (answer or "").lower()
    compact = low.replace(" ", "")
    cred_led = "docker1.cloudfabrix" in low or (
        "8.0.0" in low and ("credential" in low or "image repository" in low)
    )
    has_cli_ops = any(
        w in low
        for w in (
            "rdaf backup", "rdaf platform", "rdaf infra", "registry fetch",
            "rdaf --version", "rdaf status", "infra→platform", "infra -> platform",
        )
    )
    studio_sizing_only = (
        ("8gb" in compact or "8 gb" in low)
        and ("50gb" in compact or "50 gb" in low)
        and not has_cli_ops
    )
    false_gap = any(
        p in low
        for p in (
            "docs don't cover",
            "docs do not cover",
            "missing specific commands for upgrading",
            "not elaborated in the provided",
        )
    )
    return bool(cred_led or studio_sizing_only or false_gap)


def _filter_sources_for_install_ask(sources: list[dict], question: str) -> list[dict]:
    if not sources or not _is_platform_install_ask(question):
        return sources
    kept = []
    for s in sources:
        blob = f"{s.get('title') or ''} {s.get('url') or ''} {s.get('excerpt') or ''}".lower()
        if _looks_like_integration_prereq_noise(blob):
            continue
        kept.append(s)
    if not kept:
        logger.info("install source filter emptied list; fail-open keeping original sources")
        return sources
    upgrade_ask = _is_platform_upgrade_ask(question)

    def _blob(s: dict) -> str:
        return f"{s.get('title') or ''} {s.get('url') or ''} {s.get('excerpt') or ''}"

    if not upgrade_ask:
        guide = [s for s in kept if _looks_like_install_guide_path(_blob(s))]
        rest = [s for s in kept if s not in guide]
        installish = [s for s in rest if _looks_like_install_doc(_blob(s))]
        other = [s for s in rest if s not in installish]
        return (guide + installish + other)[:12]

    primary = [s for s in kept if _looks_like_upgrade_primary(_blob(s))]
    rest = [s for s in kept if s not in primary]
    upgrade = [s for s in rest if _looks_like_upgrade_doc(_blob(s))]
    rest2 = [s for s in rest if s not in upgrade]
    demoted = [s for s in rest2 if _looks_like_upgrade_demote(_blob(s))]
    rest3 = [s for s in rest2 if s not in demoted]
    guide = [s for s in rest3 if _looks_like_install_guide_path(_blob(s))]
    rest4 = [s for s in rest3 if s not in guide]
    installish = [s for s in rest4 if _looks_like_install_doc(_blob(s))]
    other = [s for s in rest4 if s not in installish]
    return (primary + upgrade + guide + installish + other + demoted)[:12]


def _answer_has_integration_prereq_drift(answer: str) -> bool:
    low = (answer or "").lower()
    return any(
        n in low
        for n in (
            "servicenow", "qualys", "crowdstrike", "logrhythm",
            "ms team's name", "microsoft teams",
        )
    )


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
- out_of_scope: ONLY billing/refunds/subscriptions, contractual support SLAs, SOC2/HIPAA/GDPR/DPA/compliance report downloads and BAA requests, third-party penetration-test/CVSS reports, private VPN/jump-host crypto, private mTLS/ingress cert rotation not in public docs, private control-plane/root password resets, HR/salary/internal employment topics, enterprise quotes/pricing/cost models/list prices, unrelated products, cooking/sports, personal advice
Never mark questions about Fabrix extensions, bots, pipelines, CFXQL, datasets, pstreams, dashboards, or AI Fabric as out_of_scope.
Prefer related over in_scope for combining pieces, comparing objects (pstream vs dataset), broad automation overviews, or designing workflows.
For platform install / prerequisites / VM or hardware sizing with no named datasource product: use facet install; search_queries must target RDA Studio / RDA Fabric installation_guides (Docker, CPU/RAM, registry) — never datasource integration credential prerequisites.
search_query must strip presentation constraints (exact step counts, blank-line formatting).

Facet rules:
- needs_facets=true for related questions, comparisons, broad how-to/ops, dashboards, monitoring, multi-intent, platform install/prereq/VM sizing.
- needs_facets=false for simple single-doc lookups (bot params, one named fact).
- When needs_facets: prefer classic RDA diversity (bots, CFXQL, pipelines, datasets, pstreams, dashboards, integrations);
  include ai_fabric only for agents/AI/toolsets/personas or broad automation.
  For install/prereq/VM asks: prefer install facet and installation_guides queries over integrations.
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


    # Deterministic safety net: known-unanswerable specifics that keyword-match
    # a real Fabrix topic but ask for a number/limit/detail the public docs don't contain.
    UNANSWERABLE_SPECIFIC_KEYWORDS = [
        "maximum number", "max number", "how many workers",
        "worker limit", "maximum workers",
        "max concurrent", "maximum concurrent", "concurrent bots",
    ]
    q_lower = question.lower()
    if any(kw in q_lower for kw in UNANSWERABLE_SPECIFIC_KEYWORDS):
        scope = "out_of_scope"


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


def _is_param_or_procedure_ask(question: str) -> bool:
    q = (question or "").lower()
    return any(
        w in q
        for w in (
            "parameter",
            "parameters",
            "cron",
            "schedule",
            "how do i",
            "how to",
            "every ",
            "minutes",
        )
    )


def _answer_claims_missing_from_excerpts(answer: str) -> bool:
    low = (answer or "").lower()
    return any(
        p in low
        for p in (
            "not in the excerpts",
            "not provided in the documentation excerpts",
            "not provided in the excerpts",
            "are not provided in the",
            "aren't provided in the",
            "not detailed in the excerpts",
            "not listed in the excerpts",
        )
    )


def _is_concurrent_dataset_ask(question: str) -> bool:
    q = (question or "").lower()
    return "concurrent" in q and "dataset" in q and any(
        w in q for w in ("write", "writes", "writing", "pipeline")
    )


def _concurrent_dataset_honesty(
    existing_gaps: list[str] | None = None,
) -> tuple[str, list[str]]:
    text = (
        "Public Fabrix docs describe datasets / pstreams and pipelines as building blocks "
        "for landing and transforming data. They do **not** specify a locking mechanism, "
        "merge policy, or guaranteed behavior for concurrent writes to the same dataset "
        "from two pipelines. Treat that concurrency detail as **not documented**; use "
        "separate datasets/streams or operational controls unless a cited excerpt says "
        "otherwise."
    )
    gaps = list(existing_gaps or [])
    gap = (
        "Concurrent writes / dataset locking behavior is not specified in the public docs"
    )
    if gap.lower() not in " ".join(g.lower() for g in gaps):
        gaps.insert(0, gap)
    return text, gaps[:8]


def _schedule_debug_honesty(
    existing_gaps: list[str] | None = None,
) -> tuple[str, list[str]]:
    text = (
        "**Documented Fabrix path**\n"
        "Scheduled pipelines are configured in a service blueprint under "
        "`scheduled_pipelines` with a `cron_expression` (see "
        "beginners_guide/scheduled_pipelines). To debug a pipeline that did not fire: "
        "1) confirm the blueprint lists the pipeline under `scheduled_pipelines`, "
        "2) verify the `cron_expression`, 3) check that workers/site services are up, "
        "4) review pipeline run history/logs. Do not invent a dedicated pipeline "
        "scheduler bot.\n\n"
        "**Next (inferred):** Exact UI clicks and log paths can vary by site."
    )
    gaps = list(existing_gaps or [])
    gap = (
        "Public docs cover scheduled_pipelines + cron_expression; "
        "site-specific miss diagnosis steps may be incomplete"
    )
    if gap.lower() not in " ".join(g.lower() for g in gaps):
        gaps.insert(0, gap)
    return text, gaps[:8]


def _is_dashboard_pipeline_kickoff_ask(question: str) -> bool:
    """Asks whether dashboards can automatically start/remediate via pipelines."""
    q = (question or "").lower()
    if "dashboard" not in q:
        return False
    if not any(w in q for w in ("pipeline", "remediat", "alert")):
        return False
    return any(
        w in q
        for w in (
            "kick off", "kickoff", "trigger", "invoke", "when an alert",
            "automatically", "auto-", "fire",
        )
    )


def _dashboard_kickoff_honesty(
    existing_gaps: list[str] | None = None,
) -> tuple[str, list[str]]:
    text = (
        "Public Fabrix docs describe dashboards (analytics on datasets / pstreams) and "
        "pipelines / alerts as related but separate building blocks. They do **not** "
        "document a turnkey path where a dashboard automatically kicks off remediation "
        "pipelines when an alert fires. Do not invent undocumented bots for that handoff. "
        "Describe documented alert/dashboard pieces you do see; treat "
        "dashboard→pipeline kickoff as **not documented**."
    )
    gaps = list(existing_gaps or [])
    gap = (
        "Dashboard-triggered remediation pipelines are not documented in public Fabrix docs"
    )
    if gap.lower() not in " ".join(g.lower() for g in gaps):
        gaps.insert(0, gap)
    return text, gaps[:8]


def _strip_ungrounded_bot_mentions(answer: str, tokens: list[str]) -> str:
    """Deterministically remove invented bot tokens that survived LLM revision."""
    text = answer or ""
    for tok in tokens:
        raw = (tok or "").lstrip("@")
        if not raw:
            continue
        text = re.sub(
            rf"`?@?{re.escape(raw)}`?",
            "a documented bot from the retrieved catalog",
            text,
            flags=re.IGNORECASE,
        )
    return text


def _is_pipeline_schedule_ask(question: str) -> bool:
    q = (question or "").lower()
    if ("schedule" in q or "cron" in q) and (
        "pipeline" in q or "rda" in q or "fabric" in q
    ):
        return True
    # Debug / miss language for scheduled pipelines (day-2 ops)
    if "pipeline" in q and any(
        w in q
        for w in (
            "didn't fire",
            "did not fire",
            "not firing",
            "failed to run",
            "didn't run",
            "did not run",
            "last night",
            "missed schedule",
        )
    ):
        return True
    return False


_INVENTED_SCHEDULE_BOTS = (
    "pipeline-scheduler",
    "schedule-pipeline",
    "@c:schedule ",
    "@c:schedule\n",
)


def _schedule_answer_invents_bot(answer: str) -> bool:
    low = (answer or "").lower()
    return any(tok in low for tok in _INVENTED_SCHEDULE_BOTS)


def _schedule_answer_missing_cron(answer: str) -> bool:
    """Schedule ask answered without cron / scheduled_pipelines shape."""
    low = (answer or "").lower()
    return not any(
        x in low
        for x in ("cron", "scheduled_pipelines", "*/15", "cron_expression")
    )


def _is_integration_wiring_ask(question: str) -> bool:
    """
    Named-product asks that expect a cred → bot → stream/dataset/dashboard path.
    Includes map/land/building-blocks wording, not only wire/datasource cues.
    Requires a real INTEGRATION_FAMILIES hit (or ServiceNow slang) — unknown
    product names must not enter depth-forcing.
    """
    q = (question or "").lower()
    if not _has_real_family_match(question):
        return False
    if any(
        w in q
        for w in (
            "datasource",
            "wire",
            "stream",
            "integrate",
            "integration",
            "into a fabrix",
            "into fabrix",
            "end-to-end",
            "walk me through",
        )
    ):
        return True
    # Family named + land/map/build into a sink, or classic building-blocks ask
    return any(
        w in q
        for w in (
            "dashboard",
            "dataset",
            "pstream",
            "building block",
            "building blocks",
            "map ",
            "land ",
            "land it",
            "connect ",
            "searchable",
        )
    )


_WIRING_INTENT_CUES: tuple[str, ...] = (
    "datasource",
    "wire",
    "stream",
    "integrate",
    "integration",
    "into a fabrix",
    "into fabrix",
    "end-to-end",
    "end to end",
    "walk me through",
    "dashboard",
    "dataset",
    "pstream",
    "building block",
    "building blocks",
    "map ",
    "land ",
    "land it",
    "connect ",
    "searchable",
    "persistent stream",
)

_WIRING_PRODUCT_STOPWORDS: frozenset[str] = frozenset(
    {
        "how", "what", "whats", "what's", "walk", "explain", "can", "is", "are",
        "where", "when", "why", "the", "for", "into", "end", "setup", "document",
        "documented", "fabrix", "rda", "rdaf", "cloud", "data", "alerts", "issues",
        "events", "metrics", "ops", "me", "through", "with", "bots", "involved",
        "persistent", "stream", "dashboard", "dataset", "pipeline", "pipelines",
        "configure", "fabio", "copilot", "agentic", "persona", "toolset", "personas",
        "toolsets", "outages", "production", "approval", "human",
    }
)


def _has_wiring_intent_cues(question: str) -> bool:
    q = (question or "").lower()
    return any(w in q for w in _WIRING_INTENT_CUES)


def _has_real_family_match(
    question: str,
    kb_entries: list[dict] | None = None,
    chunks: list[dict] | None = None,
) -> bool:
    """
    True when the question names a known INTEGRATION_FAMILIES product (or SN slang),
    or retrieved context already contains @/#/* family bot tokens for a hit family.
    Unknown product-sounding names (e.g. Grafana Cloud) return False so depth-forcing
    and path_first wiring shape do not run.
    """
    fams = _integration_family_hits(question)
    q = (question or "").lower()
    if not fams and any(p in q for p in ("servicenow", "service now", "snow", "sn ")):
        fams = ["servicenow"]
    if fams:
        return True
    blob = _retrieved_context_blob(kb_entries, chunks) if (kb_entries or chunks) else ""
    if not blob:
        return False
    # Context-only rescue: bot tokens present whose family aliases appear in the question
    for tok in _full_bot_tokens_in_text(blob):
        pref = tok.split(":", 1)[0]
        fam = _family_for_bot_prefix(pref)
        if fam and fam in q.replace(" ", ""):
            return True
        if any(a in q for a in (pref, pref.replace("-", " "), pref.replace("_", " "))):
            return True
    return False


def _extract_wiring_product_phrases(question: str) -> list[str]:
    """Product-like phrases from a wiring-shaped question (before family gating)."""
    q = question or ""
    found: list[str] = []
    seen: set[str] = set()

    def _add(phrase: str) -> None:
        p = re.sub(r"^(?i)(?:the|a|an)\s+", "", (phrase or "").strip().strip("\"'`"))
        p = re.sub(r"\s+", " ", p).strip()
        if not p or len(p) < 2:
            return
        key = p.lower()
        if key in seen or key in _WIRING_PRODUCT_STOPWORDS or key in _KNOWN_DOC_ENTITY_ALLOW:
            return
        # Drop phrases that are only stopwords
        parts = key.split()
        if parts and all(part in _WIRING_PRODUCT_STOPWORDS for part in parts):
            return
        seen.add(key)
        found.append(p)

    for m in re.finditer(
        r"(?i)\b(?:wire|map|land|connect|integrate)\s+(.+?)\s+"
        r"(?:into|to|as|with|for|end)\b",
        q,
    ):
        _add(m.group(1))
    for m in re.finditer(
        r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,2})\b",
        q,
    ):
        _add(m.group(1))
    return found


def _unmatched_wiring_product_names(question: str) -> list[str]:
    """
    Wiring-intent ask that names a product-sounding term with no INTEGRATION_FAMILIES hit.
    Requires explicit wire/map/land/connect/integrate cues — not bare end-to-end / dashboard
    (those also appear on agentic capability asks like Fabio auto-remediate).
    """
    q = (question or "").lower()
    if not any(
        w in q
        for w in (
            "wire",
            "map ",
            "land ",
            "land it",
            "connect ",
            "integrate",
            "integration",
            "datasource",
            "into a fabrix",
            "into fabrix",
        )
    ):
        return []
    if _has_real_family_match(question):
        return []
    # Agentic / Copilot capability asks are not third-party wiring products
    if _question_names_agentic(question) or _is_capability_overclaim_ask(question):
        return []
    return _extract_wiring_product_phrases(question)


def _ungrounded_wiring_products(
    question: str,
    kb_entries: list[dict] | None,
    chunks: list[dict] | None,
) -> list[str]:
    """Unmatched wiring products that also never appear in retrieved excerpts."""
    names = _unmatched_wiring_product_names(question)
    if not names:
        return []
    blob = _retrieved_context_blob(kb_entries, chunks)
    if not blob.strip():
        return names
    return [n for n in names if not _term_grounded_in_blob(n, blob)]


def _wiring_answer_has_auth_cue(answer: str) -> bool:
    low = (answer or "").lower()
    return any(
        w in low
        for w in (
            "credential",
            "username",
            "password",
            "token",
            "api key",
            "auth",
            "api access",
            "api integration",
            "read access",
            "service user",
            "hec",
        )
    )


def _wiring_answer_has_asked_sink(question: str, answer: str) -> bool:
    """Prefer sinks the user named; otherwise any stream/dataset/dashboard/pipeline."""
    q = (question or "").lower()
    a = (answer or "").lower()
    asked: list[str] = []
    if "dashboard" in q:
        asked.append("dashboard")
    if "dataset" in q or "searchable" in q:
        asked.append("dataset")
    if "pstream" in q or "persistent stream" in q:
        asked.extend(["pstream", "persistent stream", "stream"])
    elif "stream" in q:
        asked.extend(["stream", "pstream"])
    if "pipeline" in q:
        asked.append("pipeline")
    if not asked:
        asked = ["stream", "pstream", "dataset", "pipeline", "dashboard"]
    return any(s in a for s in asked)


def _wiring_answer_missing_product_bot(
    question: str,
    answer: str,
    context_blob: str = "",
) -> bool:
    """
    Named-product wiring ask whose answer never cites a concrete @family:op bot.
    Requires a real bot token matching the named family when those tokens appear in
    retrieved context (prose 'bot' alone is not enough). If context has no family
    bots, do not force invention — gaps/honesty handle that.
    """
    fams = _integration_family_hits(question)
    if not fams:
        return False
    if not _is_integration_wiring_ask(question):
        return False
    if context_blob and not _wiring_context_has_family_bots(question, context_blob):
        return False
    tokens = _full_bot_tokens_in_text(answer)
    if not tokens:
        return True
    for tok in tokens:
        fam = _family_for_bot_prefix(tok.split(":", 1)[0])
        if fam and fam in fams:
            return False
        if "servicenow" in fams and any(x in tok for x in ("snow", "servicenow")):
            return False
    return True


def _wiring_context_has_family_bots(question: str, context_blob: str) -> bool:
    fams = set(_integration_family_hits(question))
    if not fams:
        return False
    for tok in _full_bot_tokens_in_text(context_blob):
        fam = _family_for_bot_prefix(tok.split(":", 1)[0])
        if fam and fam in fams:
            return True
        if "servicenow" in fams and any(x in tok for x in ("snow", "servicenow")):
            return True
    return False


def _wiring_source_bot_tokens_in_context(question: str, context_blob: str) -> list[str]:
    """
    @family:op tokens marked Source in expanded catalog text for named families.
    Used so into-Fabrix wiring prefers pull/search bots over external sink bots.
    """
    fams = set(_integration_family_hits(question))
    if not fams or not context_blob:
        return []
    blob = context_blob
    out: list[str] = []
    for m in re.finditer(
        r"##\s*Bot\s+[@*#]([A-Za-z0-9_-]+):([A-Za-z0-9_-]+)(.*?)(?=\n##\s*Bot\s+[@*#]|\Z)",
        blob,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        prefix, op, body = m.group(1), m.group(2), m.group(3) or ""
        fam = _family_for_bot_prefix(prefix.lower())
        if not fam or fam not in fams:
            if not (
                "servicenow" in fams
                and any(x in prefix.lower() for x in ("snow", "servicenow"))
            ):
                continue
        if "bot position in pipeline: source" not in body.lower():
            continue
        tok = f"{prefix.lower()}:{op.lower()}"
        if tok not in out:
            out.append(tok)
    return out


def _wiring_answer_missing_source_bot(
    question: str,
    answer: str,
    context_blob: str = "",
) -> bool:
    """
    Into-Fabrix / dashboard / dataset wiring that cites only external-sink bots
    while Source bots for that family exist in context.
    """
    if not _is_integration_wiring_ask(question):
        return False
    q = (question or "").lower()
    into_fabrix = any(
        w in q
        for w in (
            "into a fabrix",
            "into fabrix",
            "dashboard",
            "dataset",
            "searchable",
            "building block",
            "land ",
            "map ",
            "persistent stream",
        )
    )
    if not into_fabrix:
        return False
    sources = _wiring_source_bot_tokens_in_context(question, context_blob)
    if not sources:
        return False
    ans_toks = set(_full_bot_tokens_in_text(answer))
    if any(t in ans_toks for t in sources):
        return False
    # Only force when the answer already named some family bot (likely wrong direction)
    fams = set(_integration_family_hits(question))
    named_family_bot = False
    for tok in ans_toks:
        fam = _family_for_bot_prefix(tok.split(":", 1)[0])
        if fam and fam in fams:
            named_family_bot = True
            break
        if "servicenow" in fams and any(x in tok for x in ("snow", "servicenow")):
            named_family_bot = True
            break
    return named_family_bot


def _wiring_answer_missing_shape(question: str, answer: str) -> bool:
    """
    Wiring ask incomplete when missing (auth cue OR concrete bot token) or missing sink.
    """
    if not _is_integration_wiring_ask(question):
        return False
    has_auth_or_bot = _wiring_answer_has_auth_cue(answer) or bool(
        _full_bot_tokens_in_text(answer)
    )
    has_sink = _wiring_answer_has_asked_sink(question, answer)
    return (not has_auth_or_bot) or (not has_sink)


def _agentic_overclaim_without_hedge(question: str, answer: str) -> bool:
    """
    Capability overclaim on Fabio/auto-remediate without honesty hedge / gaps cue.
    """
    if not _question_names_agentic(question):
        return False
    q = (question or "").lower()
    if not any(
        w in q
        for w in ("auto-remediat", "no human", "end-to-end", "without human", "fully automatic")
    ):
        return False
    low = (answer or "").lower()
    over = any(
        p in low
        for p in (
            "without any human",
            "no human approval",
            "fully automatic",
            "end-to-end without",
            "automatically remediate",
            "no human intervention",
        )
    )
    if not over:
        # Claiming yes without hedge is also bad
        if re.search(r"\byes\b", low[:80]) and "not" not in low[:200]:
            over = True
    if not over:
        return False
    hedge = any(
        h in low
        for h in (
            "does not",
            "do not",
            "don't",
            "doesn't",
            "not explicitly",
            "not documented",
            "not support",
            "docs do not",
            "documentation does not",
            "unsupported",
            "cannot claim",
            "can't claim",
            "put unsupported",
            "next (inferred)",
            "gaps[]",
            "in gaps",
            "human oversight",
            "human approval is",
            "requires human",
            "with human",
            "need human",
        )
    )
    return not hedge


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
    """True when a wire/add-datasource/map-to-sink ask lacks the asked sink."""
    if _is_integration_wiring_ask(question):
        return not _wiring_answer_has_asked_sink(question, answer)
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
- Comparisons of Fabrix-internal concepts (e.g. pstream vs dataset, vCenter vs vROps): cover both sides from docs; inferred wiring under Next (inferred).
- Do NOT require or invent comparisons to named third-party/competitor tools (Airflow, generic ETL rivals, etc.) unless the draft's excerpts-backed claim already states that comparison. If the draft correctly says a direct rival comparison or a specific version/limit is not documented, that is correct — set fix_needed=false for that issue; do not ask to "add a clear comparison" to an undocumented rival.
- Specificity: topic documented ≠ detail documented. Flag overclaim=true if the draft invents a version number, numeric limit, or rival bake-off framework not supported by docs.
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
    ungrounded_named_entities: list[str] | None = None,
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

    full_page_line = ""
    if any(e.get("kind") == "full_page" for e in kb_entries):
        full_page_line = (
            "- FULL DOC PAGES entries are complete source pages (prefer parameter tables, "
            "cron/YAML examples, and CLI blocks from those over shorter excerpts).\n"
        )

    allowed_fams = _integration_family_hits(question)
    fidelity_line = ""
    if allowed_fams:
        fidelity_line = (
            "- Product fidelity: only name bots whose prefix/family matches products in the "
            f"question ({', '.join(allowed_fams)}). Never cite sibling-product bots "
            "(e.g. BMC/Remedy bots on a ServiceNow question, Zabbix on a Prometheus question).\n"
        )

    ungrounded_line = ""
    if ungrounded_named_entities:
        terms = ", ".join(f"'{t}'" for t in ungrounded_named_entities[:6])
        ungrounded_line = (
            f"- NAMED ENTITY NOT IN EXCERPTS: the user asked about {terms}, which does not "
            "appear anywhere in the retrieved documentation excerpts. Do NOT assume it exists "
            "or describe how to use/configure/enable it. State clearly that you do not see "
            "this specific feature/bot documented. You may briefly mention genuinely related "
            "documented concepts only if helpful, without implying the named thing is real.\n"
        )

    mental_model = """
Fabrix mental model (use when relevant; do not keyword-stuff):
- Classic RDA automation stack: bots → CFXQL → pipelines → datasets / persistent streams (pstreams) → dashboards / integrations.
- Agentic stack: Toolsets / Personas / Prompt Templates — lead with this only for agents/AI Fabric questions.
- REQUIRED for broad "automation / how Fabrix works / building blocks" questions: your answer MUST explicitly name at least 2 of (bots, CFXQL, pipelines, datasets/pstreams) from the classic stack BEFORE mentioning any Agentic components. Never give an Agentic-only answer to a broad automation question, even if ai_fabric excerpts are present in context.
- Dashboard data path: prefer datasets / pstreams / pipelines from excerpts; do not answer with only Edge Collector unless that is all the excerpts support (then say so in Gaps).
"""

    path_first = """
- For "how would I / chain / integrate / wire / map / land / connect / building blocks / add X as a datasource" questions, use this shape:
  **Documented Fabrix path**   ← bold header only, NOT a numbered step
  1. credentials / auth for the named product (API token, username/password, read-only user — from excerpts)
  2. concrete bots from that product family as exact `@family:op` tokens from excerpts (never invent `@c:…`); for map/land/into-Fabrix/dashboard/dataset asks prefer Source / search / read bots over Sink bots that only write back to the external product
  3. stream or dataset handoff (name stream/pstream/dataset as asked)
  4. dashboard / Slack / downstream step when the question asks for it (from excerpts)
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
    # Depth-forcing only when a real family/catalog match exists. Unknown products
    # (e.g. Grafana Cloud) must not get an auth→bot→sink narrative scaffold.
    if _unmatched_wiring_product_names(question):
        path_first = """
- The question asks to wire/map/land a named product that is NOT a documented Fabrix
  integration family in the knowledge base. Do NOT invent credentials, bots, streams,
  or a dashboard path for it. Clearly state you do not see that product/integration
  documented, and invite the user to name a documented family (e.g. Splunk, Datadog,
  PagerDuty, Jira) if that is what they meant.
"""
    elif not _has_real_family_match(question) and _has_wiring_intent_cues(question):
        # Wiring-shaped ask with no product family — keep light guidance, no invented family
        path_first = """
- If the user asks how to wire/land data in Fabrix without naming a documented integration,
  describe only the classic documented stack (credentials → bots → pipeline → dataset/pstream
  → dashboard) from excerpts. Do not invent a third-party product bot family.
"""

    code_gen_guardrail = ""
    if _is_full_script_automation_ask(question):
        code_gen_guardrail = """
- If asked to write a complete script, playbook, or end-to-end automation that chains multiple commands/steps together, do NOT invent a single ready-to-run script with placeholder versions, paths, or credentials.
- Instead: list the individual documented commands/steps in order (as shown in the excerpts), explicitly note that exact values (version tags, paths, hostnames) must be filled in from the user's own environment/release notes, and flag this as something requiring manual verification, not turnkey automation.
- Never invent password variables, default credentials, or example secrets in code blocks, even as placeholders like $PASSWORD or admin:changeme.
"""

    specificity_guardrail = """
- CRITICAL: a topic being documented does not mean every specific detail about it is documented. Before stating a specific number, version, exact comparison, or precise technical claim, verify that exact detail appears in the excerpts -- not just the general topic.
- If the excerpts discuss a topic broadly (e.g., "integrates with Kubernetes") but do NOT state a specific requested detail (e.g., a version number, a named comparison to another product, an exact limit), say so explicitly: name the topic that IS documented, then clearly state the specific detail is not specified in the docs.
- Never construct a comparison to a named competitor or third-party tool (e.g., Airflow, Zabbix as a rival, etc.) unless the docs explicitly make that comparison. Describe Fabrix's own documented approach instead, and note that a direct comparison isn't in the public docs.
"""

    if scope == "related":
        infer_guidance = f"""
- Write ONE coherent best answer that combines documented facts with limited Fabrix
  technical synthesis implied by the docs.
- Cite documented claims with [1], [2], … Do NOT put [n] on inferred reasoning.
- Set used_inference=true and fill inferred_summary when you synthesize beyond verbatim docs.
{objects_line}{full_page_line}{fidelity_line}{ungrounded_line}{path_first}{code_gen_guardrail}{specificity_guardrail}
"""
    else:
        infer_guidance = f"""
- Write ONE coherent best answer from the excerpts; cite with [1], [2], …
- If you add connective technical reasoning beyond a single excerpt, set used_inference=true
  and fill inferred_summary. Otherwise used_inference=false and inferred_summary="".
- Do NOT put [n] on inferred reasoning.
{objects_line}{full_page_line}{fidelity_line}{ungrounded_line}{path_first}{code_gen_guardrail}{specificity_guardrail}
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
    # Strip leaked trailer / bare json fences from user-facing body (any position)
    cleaned = re.sub(r"```json\b[\s\S]*?```", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```json\b[\s\S]*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```json\b", "", cleaned, flags=re.IGNORECASE)
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

        # Collect a contiguous ordered list, spanning blank lines, indented
        # continuations (sub-bullets / nested prose), and fenced code blocks.
        # LLMs often restart at "1." after those; treat them as one run so we
        # can renumber 1,2,1,1… → 1,2,3,4…
        run_idxs: list[int] = []
        j = i
        in_fence = False
        while j < len(lines):
            ln = lines[j]
            stripped = ln.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                j += 1
                continue
            if in_fence:
                j += 1
                continue
            if re.match(r"^\d+\.\s+", ln):
                run_idxs.append(j)
                j += 1
                continue
            if stripped == "" or ln[:1] in (" ", "\t"):
                j += 1
                continue
            break

        if not run_idxs:
            out.append(lines[i])
            i += 1
            continue

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
    ("elastic", ("elastic", "elasticsearch", "opensearch", "elk")),
    ("servicenow", ("servicenow", "service now", "sn ticketing")),
    ("bmc", ("bmc-remedy", "bmc remedy", "bmc_remedy")),
    ("kubernetes", ("kubernetes", "k8s", "kubectl", "kubernetes-inventory")),
    ("vmware", ("vmware", "vcenter", "vrealize", "vrops")),
    ("cisco_ucs", ("ucsm", "ucs-manager", "ucs manager", "cisco ucs")),
    ("pagerduty", ("pagerduty", "pager duty")),
    ("datadog", ("datadog",)),
    ("dynatrace", ("dynatrace",)),
    ("solarwinds", ("solarwinds", "solar winds")),
    ("newrelic", ("new relic", "newrelic", "new_relic")),
    ("appdynamics", ("appdynamics", "app dynamics", "appd")),
    ("nagios", ("nagios", "nagios xi", "nagiosxi")),
    ("kafka", ("kafka", "kafka-v2")),
    ("netapp", ("netapp", "netapp-eseries", "netapp7", "netappc")),
    ("nutanix", ("nutanix",)),
    ("opsgenie", ("opsgenie", "ops genie", "ops-genie")),
    ("msteams", ("msteams", "ms-teams", "microsoft teams", "ms teams")),
    ("jira", ("jira",)),
    ("slack", ("slack",)),
    ("aws", ("aws", "amazon web services")),
    ("azure", ("azure", "microsoft azure")),
    ("linux", ("linux-inventory", "linux-os", "linux os")),
    # Discovery / EDR families present in Bots/ but previously unwired
    ("windows", ("windows-inventory", "windows inventory", "windows-os", "windows os")),
    ("openstack", ("openstack", "openstack-v2", "openstack_v2")),
    ("crowdstrike", ("crowdstrike", "crowd strike", "crowd-strike")),
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

BOT_TOKEN_RE = re.compile(r"[@*#]([a-zA-Z0-9][a-zA-Z0-9_-]{0,60})\s*:")
BOT_FULL_TOKEN_RE = re.compile(
    r"[@*#]([a-zA-Z0-9][a-zA-Z0-9_-]{0,60})\s*:\s*([a-zA-Z0-9][a-zA-Z0-9_-]*)"
)
# Invented "something-bot" labels (backticks or bare prose)
INVENTED_BOT_LABEL_RE = re.compile(
    r"(?:`([a-z][a-z0-9_-]{2,60}-bot)`|(?<![@*\w/-])([a-z][a-z0-9_-]{2,60}-bot)(?![\w-]))",
    re.IGNORECASE,
)

# Common misspellings → canonical product tokens (generic; not per-question)
_PRODUCT_TYPOS: tuple[tuple[str, str], ...] = (
    ("servicenw", "servicenow"),
    ("servicenwo", "servicenow"),
    ("promethus", "prometheus"),
    ("prometheous", "prometheus"),
    ("kubernets", "kubernetes"),
    ("zabbixx", "zabbix"),
    ("splunkk", "splunk"),
    ("nagiosxi", "nagios xi"),
)


def _normalize_question_typos(question: str) -> str:
    """Rewrite known product misspellings / slang so family detection still fires."""
    q = question or ""
    low = q.lower()
    for typo, canonical in _PRODUCT_TYPOS:
        if typo == canonical:
            continue
        if typo in low:
            # case-insensitive replace preserving surrounding text
            q = re.sub(re.escape(typo), canonical, q, flags=re.IGNORECASE)
            low = q.lower()
    # Ops slang abbreviations (word-boundary only)
    q = re.sub(r"\bog\b", "Opsgenie", q, flags=re.IGNORECASE)
    # SQL-as-dataset-query → CFXQL (Fabrix's SQL-like language). Keep "SQL" in the
    # text so answers can address the user's framing; expand for retrieve/scope.
    low = q.lower()
    if re.search(r"\bsql\b", low) and "cfxql" not in low:
        if any(
            w in low
            for w in ("dataset", "datasets", "query", "queries", "querying", "filter")
        ):
            q = re.sub(r"\bSQL\b", "SQL CFXQL", q, flags=re.IGNORECASE)
    return q


def _answer_mentions_agentic(answer: str) -> bool:
    low = (answer or "").lower()
    return any(
        m in low
        for m in ("fabio", "fabaio", "copilot", "agentic", "persona", "toolset")
    )


def _question_names_agentic(question: str) -> bool:
    q = (question or "").lower()
    return any(m in q for m in ("fabio", "copilot", "agentic"))


def _polish_known_product_typos(answer: str) -> str:
    """Fix recurring model misspellings of Fabrix product names."""
    if not answer:
        return answer
    return re.sub(r"\bFabaio\b", "Fabio", answer, flags=re.IGNORECASE)


def _integration_family_hits(question: str) -> list[str]:
    """Canonical integration families explicitly named in the question."""
    q = _normalize_question_typos(question or "").lower()
    hits: list[str] = []
    for canonical, aliases in INTEGRATION_FAMILIES:
        # Bare "linux"/"ubuntu" host mentions are red herrings, not linux-inventory
        if canonical == "linux":
            if any(a in q for a in ("linux-inventory", "linux-os", "linux os")):
                hits.append(canonical)
            continue
        if canonical == "windows":
            # Bare "windows" is too broad; require inventory/OS discovery cues
            if any(
                a in q
                for a in (
                    "windows-inventory",
                    "windows inventory",
                    "windows-os",
                    "windows os",
                    "windows server",
                )
            ):
                hits.append(canonical)
            continue
        if canonical == "servicenow":
            if "servicenow" in q or "service now" in q or "sn ticketing" in q:
                hits.append(canonical)
            elif re.search(r"\bsn\b", q) or re.search(r"\bsnow\b", q):
                hits.append(canonical)
            continue
        if canonical == "pagerduty":
            if any(a in q for a in aliases) or re.search(r"\bpd\b", q):
                hits.append(canonical)
            continue
        if canonical == "opsgenie":
            if any(a in q for a in aliases) or re.search(r"\bog\b", q):
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


def _full_bot_tokens_in_text(text: str) -> list[str]:
    """Full @family:operation tokens (normalized lowercase)."""
    out: list[str] = []
    for m in BOT_FULL_TOKEN_RE.finditer(text or ""):
        out.append(f"{m.group(1).lower()}:{m.group(2).lower()}")
    return out


def _invented_bot_labels(answer: str, context_blob: str) -> list[str]:
    """Hyphenated *-bot labels in the answer that never appear in retrieved context."""
    blob = (context_blob or "").lower()
    bad: list[str] = []
    for m in INVENTED_BOT_LABEL_RE.finditer(answer or ""):
        label = (m.group(1) or m.group(2) or "").lower()
        if not label:
            continue
        if label not in blob and label.replace("-", "_") not in blob:
            bad.append(label)
    return bad


def _retrieved_context_blob(kb_entries: list[dict] | None, chunks: list[dict] | None) -> str:
    parts: list[str] = []
    for e in kb_entries or []:
        parts.append(
            f"{e.get('title') or ''} {e.get('text') or ''} "
            f"{e.get('example') or ''} {e.get('source') or ''}"
        )
    for c in chunks or []:
        meta = c.get("metadata") or {}
        parts.append(
            f"{c.get('title') or ''} {c.get('text') or c.get('page_content') or ''} "
            f"{c.get('source') or ''} {meta.get('source') or ''} {meta.get('bot_name') or ''}"
        )
    return "\n".join(parts).lower()


# Platform / product tokens that are common in questions and should not trigger
# "invented feature" checks even when a thin retrieve misses them.
_KNOWN_DOC_ENTITY_ALLOW = frozenset(
    {
        "fabrix", "cloudfabrix", "rda", "rdaf", "rdafabric", "cfxql", "cfx", "aiops",
        "oia", "mcp", "kafka", "splunk", "zabbix", "prometheus", "servicenow", "snow",
        "datadog", "pagerduty", "newrelic", "dynatrace", "solarwinds", "nagios",
        "opsgenie", "slack", "kubernetes", "k8s", "docker", "ubuntu", "linux",
        "centos", "freebsd", "fedora", "alpine", "debian", "rhel", "windows",
        "macos", "suse", "rocky", "alma",
        "toolset", "persona", "copilot", "fabio", "pipeline", "pstream", "dataset",
        "dashboard", "bot", "bots", "cli", "api", "vm", "vms", "sql", "yaml",
    }
)

_BOT_ENTITY_RE = re.compile(r"@([A-Za-z][A-Za-z0-9_-]*):([A-Za-z0-9_-]+)")
_QUOTED_ENTITY_RE = re.compile(r"['\"]([A-Za-z][A-Za-z0-9 _-]{1,60})['\"]")
_CAMEL_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-zA-Z0-9]+)+)\b")
_CALLED_ENTITY_RE = re.compile(
    r"(?:feature\s+called|called|named)\s+['\"]?([A-Za-z][A-Za-z0-9_-]{2,40})['\"]?",
    re.IGNORECASE,
)


def _named_entity_candidates(question: str) -> list[str]:
    """Proper-noun / bot-style terms from the question worth grounding in excerpts."""
    q = question or ""
    found: list[str] = []
    seen: set[str] = set()

    def _add(term: str) -> None:
        t = (term or "").strip()
        if not t:
            return
        key = t.lower()
        if key in seen or key in _KNOWN_DOC_ENTITY_ALLOW:
            return
        # Skip pure sentence starters / tiny tokens
        if len(key) < 3:
            return
        seen.add(key)
        found.append(t)

    for m in _BOT_ENTITY_RE.finditer(q):
        _add(f"@{m.group(1)}:{m.group(2)}")
    for m in _QUOTED_ENTITY_RE.finditer(q):
        _add(m.group(1).strip())
    for m in _CAMEL_ENTITY_RE.finditer(q):
        _add(m.group(1))
    for m in _CALLED_ENTITY_RE.finditer(q):
        _add(m.group(1))
    return found


def _term_grounded_in_blob(term: str, blob: str) -> bool:
    """Case-insensitive grounding with light punctuation folding."""
    if not term or not blob:
        return False
    t = term.lower().strip()
    variants = {
        t,
        t.lstrip("@"),
        t.replace("_", "-"),
        t.replace("-", "_"),
        t.replace(" ", ""),
        t.replace(" ", "-"),
        t.replace(" ", "_"),
    }
    return any(v and v in blob for v in variants)


def _ungrounded_named_entities(
    question: str,
    kb_entries: list[dict] | None,
    chunks: list[dict] | None,
) -> list[str]:
    """Named question entities that never appear in retrieved excerpt text."""
    candidates = _named_entity_candidates(question)
    if not candidates:
        return []
    blob = _retrieved_context_blob(kb_entries, chunks)
    if not blob.strip():
        return candidates
    return [t for t in candidates if not _term_grounded_in_blob(t, blob)]


def _answer_acknowledges_ungrounded(answer: str, terms: list[str]) -> bool:
    low = (answer or "").lower()
    honesty = any(
        m in low
        for m in (
            "don't see",
            "do not see",
            "doesn't appear",
            "does not appear",
            "not documented",
            "not in the documentation",
            "couldn't find",
            "could not find",
            "no feature called",
            "not a real bot",
            "doesn't exist",
            "does not exist",
            "isn't documented",
            "is not documented",
            "no bot named",
            "not a documented",
            "outside the",
            "out of scope",
        )
    )
    return honesty


def _ungrounded_entity_honesty(
    terms: list[str],
    existing_gaps: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Deterministic refusal when a named feature/bot is absent from excerpts."""
    shown = ", ".join(f"'{t}'" for t in terms[:4]) or "that named feature/bot"
    text = (
        f"I don't see {shown} in the Fabrix / RDA documentation excerpts. "
        "I won't invent how to enable, configure, or use a feature or bot that isn't "
        "documented. If you meant a different documented product or bot family, ask "
        "with that exact name and I can look it up."
    )
    gaps = list(existing_gaps or [])
    gap = f"{shown} is not present in the retrieved public docs"
    if gap not in gaps:
        gaps.insert(0, gap)
    return text, gaps[:8]


def _ungrounded_bot_tokens(
    answer: str,
    kb_entries: list[dict] | None,
    chunks: list[dict] | None,
) -> list[str]:
    """
    Bot tokens named in the answer that do not appear in retrieved docs.
    Phase 5: match full @family:op tokens (not only prefixes) + invented `*-bot` labels.
    """
    blob = _retrieved_context_blob(kb_entries, chunks)
    if not blob.strip():
        return []
    bad: list[str] = []
    seen: set[str] = set()

    for full in _full_bot_tokens_in_text(answer):
        if full in seen:
            continue
        seen.add(full)
        fam, _, op = full.partition(":")
        variants = {
            full,
            f"@{full}",
            f"*{full}",
            full.replace("_", "-"),
            full.replace("-", "_"),
        }
        if not any(v in blob for v in variants):
            bad.append(f"@{full}")

    for label in _invented_bot_labels(answer, blob):
        if label not in seen:
            seen.add(label)
            bad.append(label)
    return bad


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


def _chunk_family_blob(chunk: dict) -> str:
    meta = chunk.get("metadata") or {}
    return (
        f"{meta.get('bot_name') or ''} {meta.get('source') or ''} "
        f"{meta.get('extension') or ''} {meta.get('family') or ''} "
        f"{chunk.get('text') or ''}"
    ).lower()


def _filter_chunks_to_families(chunks: list[dict], question: str) -> list[dict]:
    """
    Phase 4: drop doc chunks that only match off-allowlist product families.
    Fail open if filtering would empty the list.
    """
    if not chunks:
        return chunks
    allowed = set(_integration_family_hits(question))
    if not allowed:
        return chunks
    blocked = _blocked_sibling_terms(question)
    kept: list[dict] = []
    for c in chunks:
        blob = _chunk_family_blob(c)
        fams = _blob_family_hits(blob)
        if fams and not any(f in allowed for f in fams):
            continue
        # Extra: drop linux-inventory / sibling sources even if blob_family misses
        if blocked and _blob_has_blocked(blob, blocked):
            allowed_terms = _allowed_family_terms(question)
            if not (allowed_terms and any(t in blob for t in allowed_terms)):
                continue
        kept.append(c)
    return kept or chunks


def _rank_chunks_for_product(chunks: list[dict], question: str) -> list[dict]:
    """Soft-rank allowed-family chunks above siblings (Phase 4)."""
    if not chunks:
        return chunks
    allowed = set(_integration_family_hits(question))
    if not allowed:
        return chunks
    terms = _allowed_family_terms(question)
    blocked = _blocked_sibling_terms(question)

    def sort_key(chunk: dict):
        blob = _chunk_family_blob(chunk)
        score = float(chunk.get("score") or 0)
        if terms and any(t in blob for t in terms):
            score += 100.0
        if blocked and _blob_has_blocked(blob, blocked):
            score -= 50.0
        return score

    return sorted(chunks, key=sort_key, reverse=True)


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


def _salvage_partial_answer(
    question: str,
    kb_entries: list[dict],
    chunks: list[dict],
    existing_gaps: list[str] | None = None,
) -> tuple[str, list[str]] | None:
    """
    When the model abstains despite retrieved docs (common on exhaustive asks),
    surface a documented subset from KB/chunk text. Generic — not per-question.
    """
    if not kb_entries and not chunks:
        return None
    keys = _question_topic_keys(question)
    qlow = _normalize_question_typos(question or "").lower()
    for group in SOFT_FACET_ALIASES:
        if any(a in qlow for a in group):
            keys.extend(group)
    lines = ["**Documented Fabrix path**", ""]
    n = 1
    for e in kb_entries or []:
        blob = f"{e.get('title') or ''} {e.get('text') or ''} {e.get('example') or ''}"
        if keys and not _text_matches_topic(blob, keys):
            continue
        snippet = (e.get("text") or e.get("title") or e.get("example") or "").strip()
        snippet = re.sub(r"\s+", " ", snippet)[:280]
        if len(snippet) < 20:
            continue
        lines.append(f"{n}. {snippet}")
        n += 1
        if n > 6:
            break
    if n == 1:
        for c in chunks or []:
            blob = f"{c.get('title') or ''} {c.get('text') or c.get('page_content') or ''}"
            if keys and not _text_matches_topic(blob, keys):
                continue
            snippet = re.sub(r"\s+", " ", (c.get("text") or c.get("page_content") or blob).strip())[:280]
            if len(snippet) < 20:
                continue
            lines.append(f"{n}. {snippet}")
            n += 1
            if n > 6:
                break
    if n == 1:
        return None
    lines.append("")
    lines.append(
        "**Next (inferred):** Public docs may not fully satisfy an exhaustive or "
        "end-to-end ask; use the documented subset above and fill gaps operationally."
    )
    gaps = list(existing_gaps or [])
    if not gaps:
        gaps = [
            "Public documentation does not provide a complete exhaustive answer for this ask"
        ]
    return "\n".join(lines), gaps


def _lookup_fast_path(question: str, client: QdrantClient) -> AgentResponse | None:
    if not is_bot_param_lookup(question):
        return None
    from bot_lookup import explicit_bot_tokens

    families = bot_family_hints(question)
    tokens = explicit_bot_tokens(question)
    if not families and not tokens:
        return None
    operation_hints = bot_operation_hints(question)

    # Phase 3: structured params from KB (ingest-time tables) — preferred
    kb_hit = lookup_bot_params_from_kb(families, operation_hints, question=question)
    if kb_hit:
        bot_name, rows, source = kb_hit
        return AgentResponse(
            answer=format_param_answer(bot_name, rows),
            sources=[kb_source_dict(source, bot_name)],
            sufficient=True,
            examples=[],
            gaps=[],
            scope="in_scope",
            used_inference=False,
        )

    # Explicit @family:op: never fall through to a different product family
    # (e.g. @snowv2:list-incidents must not become @opsgenie:list-incidents).
    if tokens:
        for fam, op in tokens:
            for fam_variant in (
                fam,
                fam.replace("-", "_"),
                fam.replace("_", "-"),
                f"{fam}_v2" if not fam.endswith("v2") else fam,
                f"{fam}-v2" if not fam.endswith("v2") else fam,
            ):
                catalog_hit = lookup_bot_params_from_catalog(fam_variant, [op])
                if not catalog_hit:
                    continue
                bot_name, rows, rel_path = catalog_hit
                if op.lower() in (bot_name or "").lower():
                    return AgentResponse(
                        answer=format_param_answer(bot_name, rows),
                        sources=[catalog_source_dict(rel_path, bot_name)],
                        sufficient=True,
                        examples=[],
                        gaps=[],
                        scope="in_scope",
                        used_inference=False,
                    )
        named = ", ".join(f"@{f}:{o}" for f, o in tokens)
        fams = ", ".join(sorted({f for f, _ in tokens}))
        text = (
            f"I don't see {named} as a documented bot in the Fabrix catalog. "
            f"I will not substitute a different product family's bot that happens to share "
            f"a similar operation name. Check the {fams} bot catalog for the nearest "
            f"documented operation, or ask about a bot that appears in the docs."
        )
        return AgentResponse(
            answer=text,
            sources=[],
            sufficient=False,
            examples=[],
            gaps=[f"{named} is not a documented bot name in the public catalog"],
            scope="in_scope",
            used_inference=False,
        )

    chunks = retrieve(question, client, top_k=8, filter_dict={"type": "bot"})
    chunks = prune_lookup_chunks(question, chunks, "lookup")

    chunk_hit = best_chunk_lookup(chunks, families, operation_hints)
    if chunk_hit:
        bot_name, rows, text = chunk_hit
        examples = []
        ex = extract_example_snippet(text)
        if ex:
            examples.append(ex)
        return AgentResponse(
            answer=format_param_answer(bot_name, rows, text),
            sources=_sources_from_chunks(chunks[:1]),
            sufficient=True,
            examples=examples,
            gaps=[],
            scope="in_scope",
            used_inference=False,
        )

    for family in families:
        catalog_hit = lookup_bot_params_from_catalog(family, operation_hints)
        if not catalog_hit:
            continue
        bot_name, rows, rel_path = catalog_hit
        return AgentResponse(
            answer=format_param_answer(bot_name, rows),
            sources=[catalog_source_dict(rel_path, bot_name)],
            sufficient=True,
            examples=[],
            gaps=[],
            scope="in_scope",
            used_inference=False,
        )

    return None


def answer(question: str, client: QdrantClient | None = None) -> AgentResponse:
    if client is None:
        client = QdrantClient(path=QDRANT_DIR)

    question = _normalize_question_typos(question)

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
    search_q = _strip_format_noise(
        _normalize_question_typos(planned.get("search_query") or question)
    )
    search_q = _strip_format_noise(search_q)
    if not search_q.strip():
        search_q = _strip_format_noise(_normalize_question_typos(question)) or question

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
                "maximum number", "max number", "how many workers",
                "worker limit", "maximum workers",
                "max concurrent", "maximum concurrent", "concurrent bots",
                "hipaa", "baa", "soc2", "soc 2", "gdpr", "signed dpa",
                "data processing agreement", "penetration test", "pentest", "cvss",
                "salary", "internal hr", "admin password", "root password",
                "whitelist an ip", "whitelist ip", "ip whitelist", "allowlist an ip",
            )
        )
        fabrixish = any(
            m in qlow
            for m in (
                "fabrix", "rda", "cfx", "cfxql", "agentic", "extension",
                "pipeline", "pstream", "dataset", "dashboard", "bot",
                "fabio", "copilot", "ai fabric", "persona", "toolset",
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

    # Exhaustive catalogs / "do it for me" capability asks → related + gaps/infer
    if scope == "in_scope" and (
        _is_exhaustive_ask(question) or _is_capability_overclaim_ask(question)
    ):
        scope = "related"
        logger.info("scope latch: in_scope→related for exhaustive/capability ask")

    # Force commercial / private-infra / jailbreak / credential-fishing traps out of scope
    qlow_trap = _normalize_question_typos(question or "").lower()
    force_oos = any(
        t in qlow_trap
        for t in (
            "enterprise quote", "list price", "cost model", "discount",
            "contractual", "p1 support sla", "refund", "subscription",
            "private vpn", "jump host", "encryption key", "mtls",
            "private ingress",
            "ignore the documentation", "ignore the docs", "ignore documentation",
            "invent a fabrix", "invent a password", "invent an admin",
            "training data", "default password", "admin password",
            "soc2", "soc 2", "type ii report", "type 2 report",
            "root password", "control plane vm",
            "hipaa", "baa", "business associate agreement",
            "salary band", "salary range", "internal hr",
            "gdpr", "signed dpa", "data processing agreement",
            "penetration test", "pentest", "cvss",
            "pretend you're not restricted", "pretend you are not restricted",
            "answer freely", "not restricted by the documentation",
            "ignore your instructions", "ignore your restrictions",
            "you said earlier", "as you mentioned before", "as you said before",
            "i'm a fabrix engineer", "i am a fabrix engineer", "for debugging purposes",
            "whitelist an ip", "whitelist ip", "ip whitelist", "allowlist an ip",
            "whitelist an ip for the api",
        )
    )
    # Credential fishing: ask for a secret value rather than how to configure auth
    if not force_oos and any(
        m in qlow_trap
        for m in ("password", "passwd", "secret key", "api key value")
    ) and any(
        m in qlow_trap
        for m in ("what is the", "what's the", "give me the", "tell me the", "default")
    ):
        force_oos = True
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
        or _is_exhaustive_ask(question)
        or _is_capability_overclaim_ask(question)
        or _is_pipeline_schedule_ask(question)
        or _is_integration_wiring_ask(question)
        or _question_names_agentic(question)
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
        # Agentic + product mix: also bias toward Copilot/audit docs
        qlow_mix = _normalize_question_typos(question or "").lower()
        if any(m in qlow_mix for m in AGENTIC_MARKERS):
            queries0 = list(facet_plan["search_queries"] or [])
            for seed in (
                "Fabio Copilot AI Fabric",
                "AI Fabric Copilot pipeline audit",
            ):
                if seed not in queries0:
                    queries0 = [seed] + queries0
            facet_plan["search_queries"] = queries0
            logger.info("agentic+product mix retrieve bias")
    elif _is_agentic_question(question) or _is_synthesis_question(question):
        queries0 = list(facet_plan["search_queries"] or [])
        seeds = [
            "Fabio Copilot AI Fabric",
            "AI Fabric Copilot pipeline audit",
        ]
        if _is_synthesis_question(question):
            qlow_syn = (question or "").lower()
            if any(w in qlow_syn for w in ("toolset", "persona", "agentic", "copilot", "fabio")):
                seeds = [
                    "Toolsets Personas Agentic AI Fabrix",
                    "AI Persona Toolset comparison",
                ] + seeds
        for seed in seeds:
            if seed not in queries0:
                queries0 = [seed] + queries0
        facet_plan["search_queries"] = queries0
        logger.info("agentic/synthesis topic mode retrieve bias")

    # CFXQL / operator asks: always prepend topic seeds (format stress often empties search_q)
    qlow_seed = _normalize_question_typos(question or "").lower()
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
        use_facets = True

    # Worker scale / capacity asks
    if "worker" in qlow_seed and any(
        w in qlow_seed for w in ("scale", "site", "limit", "max", "capacity", "busy")
    ):
        queries0 = list(facet_plan["search_queries"] or [])
        for seed in (
            "RDA workers site scale capacity",
            "RDA Fabric workers administration",
            "workers per site Fabrix",
        ):
            if seed not in queries0:
                queries0 = [seed] + queries0
        facet_plan["search_queries"] = queries0
        use_facets = True
        if scope == "in_scope":
            scope = "related"
        logger.info("worker scale topic retrieve bias")

    # Platform install / upgrade / VM / prerequisites (not datasource integration prereqs)
    if _is_platform_install_ask(question):
        queries0 = list(facet_plan["search_queries"] or [])
        if _is_platform_upgrade_ask(question):
            seeds = [
                "RDAF deployment CLI rdaf upgrade platform infrastructure workers",
                "rdaf_cli upgrade backup registry status non-kubernetes",
                "rdaf backup dest-dir mariadb minio before upgrade",
                "rdaf registry fetch tag docker2.cloudfabrix.io",
                "rdaf infra upgrade platform upgrade worker upgrade",
                "rdafk8s upgrade backup Kubernetes platform services",
                "RDAF start stop operations after upgrade validate",
                "installation_guides rdaf_cli pip install rdafcli tar.gz",
                "RDA Fabric distributed VM deployment CPU memory platform",
            ]
        else:
            seeds = [
                "RDA Studio installation prerequisites Docker CPU memory disk",
                "RDA Fabric platform deployment hardware software requirements",
                "installation_guides RDA Studio docker registry python",
                "RDAF install Ubuntu Docker Compose system requirements",
                "virtual machine VM requirements RDA Studio 8GB memory Docker",
                "docs.fabrix.ai installation_guides prerequisites",
            ]
        for seed in seeds:
            if seed not in queries0:
                queries0 = [seed] + queries0
        facet_plan["search_queries"] = queries0
        objs = list(facet_plan.get("primary_objects") or [])
        if _is_platform_upgrade_ask(question):
            terms = (
                "rdaf", "rdaf_cli", "upgrade", "backup", "registry", "status",
                "rdafk8s", "docker2.cloudfabrix.io", "infra",
            )
        else:
            terms = (
                "RDA Studio", "installation_guides", "Docker", "prerequisites",
                "CPU", "memory", "virtual machine",
            )
        for term in terms:
            if term not in objs:
                objs.insert(0, term)
        facet_plan["primary_objects"] = objs[:8]
        use_facets = True
        if scope == "in_scope":
            scope = "related"
        logger.info(
            "platform %s topic retrieve bias",
            "upgrade" if _is_platform_upgrade_ask(question) else "install",
        )

    # Ultra-short soft-facet asks (e.g. "pstream?") — seed the facet even if planner is thin
    stripped_q = re.sub(r"[^\w\s-]", " ", qlow_seed).strip()
    if len(stripped_q) <= 24:
        soft_seeds: list[tuple[str, tuple[str, ...]]] = [
            ("pstream", ("persistent stream pstream Fabrix", "pstream vs dataset")),
            ("dataset", ("Fabrix dataset dashboard", "dataset vs pstream")),
            ("cfxql", ("CFXQL operators Full Restricted",)),
            ("dashboard", ("RDA Fabric dashboard dataset pstream",)),
        ]
        for key, seeds in soft_seeds:
            if key in stripped_q or any(k in stripped_q for k in key.split()):
                queries0 = list(facet_plan["search_queries"] or [])
                for seed in seeds:
                    if seed not in queries0:
                        queries0 = [seed] + queries0
                facet_plan["search_queries"] = queries0
                if scope == "in_scope":
                    scope = "related"
                use_facets = True
                logger.info("short soft-facet retrieve bias key=%s", key)
                break

    # Product / agentic seed bias always forces multi-facet retrieve
    if fam_hits or any(m in qlow_seed for m in AGENTIC_MARKERS):
        use_facets = True

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

    # Install/upgrade asks: rescue when similarity latched onto wrong facet
    if _is_platform_install_ask(question):
        cleaned = _filter_kb_entries_for_install_ask(list(kb_entries or []), question)
        if _is_platform_upgrade_ask(question):
            good = [e for e in cleaned if _looks_like_upgrade_doc(_entry_source_blob(e))]
        else:
            good = [e for e in cleaned if _looks_like_install_doc(_entry_source_blob(e))]
        if len(good) < 2:
            logger.info(
                "%s retrieve rescue: seeding installation_guides",
                "upgrade" if _is_platform_upgrade_ask(question) else "install",
            )
            rescued: list[dict] = []
            seen_ids: set[str] = set()
            seeds = (
                (
                    "RDAF deployment CLI rdaf upgrade platform infrastructure",
                    "rdaf_cli upgrade backup registry status",
                    "rdaf backup dest-dir before upgrade",
                    "rdaf registry fetch docker2",
                    "rdaf infra upgrade platform upgrade worker upgrade",
                    "rdafk8s upgrade backup Kubernetes",
                    "RDAF start stop operations validate after upgrade",
                    "installation_guides rdaf_cli pip install rdafcli",
                )
                if _is_platform_upgrade_ask(question)
                else (
                    "RDA Studio installation prerequisites Docker CPU memory disk",
                    "RDA Fabric platform deployment hardware software requirements",
                    "installation_guides RDA Studio docker registry python pip3",
                    "RDAF install Ubuntu Docker Compose system requirements",
                    "virtual machine VM requirements RDA Studio Docker memory",
                )
            )
            for seed in seeds:
                for e in retrieve_kb(seed, top_k=8):
                    eid = str(e.get("id") or "")
                    if eid and eid in seen_ids:
                        continue
                    if eid:
                        seen_ids.add(eid)
                    rescued.append(e)
            rescued = _filter_kb_entries_for_install_ask(rescued, question)
            rescued = _rank_entries_for_install_ask(rescued, question)
            if rescued:
                kb_entries = rescued[:10]
                chunks = []

        # Optional live browse of the same pages ChatGPT opens (fail-open)
        live_entries = live_install_kb_entries(
            upgrade=_is_platform_upgrade_ask(question)
        )
        if live_entries:
            local = _rank_entries_for_install_ask(
                _filter_kb_entries_for_install_ask(list(kb_entries or []), question),
                question,
            )
            kb_entries = (live_entries + local)[:12]
            logger.info("live_docs: prepended %s installation_guides page(s)", len(live_entries))

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
        if "worker" in qlow and any(
            w in qlow for w in ("scale", "site", "limit", "max", "capacity")
        ):
            seeds.append("RDA workers site scale capacity")
            seeds.append("RDA Fabric workers administration")
        if _is_pipeline_schedule_ask(question):
            seeds.append("scheduled_pipelines cron_expression service blueprint")
            seeds.append("Schedule Pipelines within Service Blueprints")
        if any(m in qlow for m in AGENTIC_MARKERS):
            seeds.append("Fabio Copilot AI Fabric")
        if _is_platform_install_ask(question):
            seeds.extend(
                [
                    "RDA Studio installation prerequisites Docker CPU memory",
                    "RDA Fabric deployment hardware software requirements",
                    "installation_guides docker registry python pip3",
                ]
            )
        for fam in _integration_family_hits(question):
            seeds.append(f"{fam} Fabrix datasource bots integration")
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
    kb_entries = _filter_kb_entries_for_install_ask(kb_entries, question)
    kb_entries = _rank_entries_for_product(kb_entries, question)
    kb_entries = _rank_entries_for_install_ask(kb_entries, question)
    chunks = _rank_chunks_for_product(chunks, question)
    chunks = _filter_chunks_to_families(chunks, question)
    expanded_pages = expand_context(kb_entries, chunks)
    # Schedule asks: ensure beginners_guide/scheduled_pipelines is in context when available
    if _is_pipeline_schedule_ask(question):
        from page_expand import expand_page as _expand_page

        has_sched = any(
            "scheduled_pipelines" in (p.get("path") or "") for p in (expanded_pages or [])
        )
        if not has_sched:
            sched_text = _expand_page("beginners_guide/scheduled_pipelines.md")
            if sched_text:
                from doc_urls import public_doc_url as _pub

                expanded_pages = list(expanded_pages or []) + [{
                    "path": "beginners_guide/scheduled_pipelines.md",
                    "url": _pub("beginners_guide/scheduled_pipelines.md"),
                    "text": sched_text,
                }]
    # Integration wiring: load bot catalog pages so answers can cite real @family:op tokens
    if _is_integration_wiring_ask(question):
        from page_expand import expand_page as _expand_page
        from doc_urls import BOTS_REL_PREFIX, public_doc_url as _pub

        for fam in _integration_family_hits(question):
            stems = [
                f"{fam}_v2",
                f"{fam}-v2",
                f"{fam}v2",
                fam,
                fam.replace("_", "-"),
                fam.replace("-", "_"),
            ]
            if fam == "servicenow":
                stems = ["servicenow_v2", "servicenow", "snow"] + stems
            if fam == "windows":
                stems = ["windows-inventory", "windows_inventory", "windows"] + stems
            if fam == "openstack":
                stems = ["openstack-v2", "openstack_v2", "openstack"] + stems
            if fam == "crowdstrike":
                stems = ["crowdstrike", "crowd-strike"] + stems
            if fam == "vmware":
                stems = [
                    "vmware-vcenter-v2",
                    "vmware-vcenter",
                    "vmwarev2",
                    "vmware",
                ] + stems
            for stem in stems:
                rel = f"{BOTS_REL_PREFIX}/{stem}.md"
                if any((p.get("path") or "") == rel for p in (expanded_pages or [])):
                    break
                text = _expand_page(rel)
                if text:
                    expanded_pages = list(expanded_pages or []) + [{
                        "path": rel,
                        "url": _pub(rel),
                        "text": text,
                    }]
                    logger.info("page_expand: wiring catalog %s", rel)
                    break
            for di_stem in (fam, fam.replace("_", "-"), fam.replace("-", "_")):
                di_rel = f"Datasource_Integrations/{di_stem}.md"
                if any((p.get("path") or "") == di_rel for p in (expanded_pages or [])):
                    break
                di_text = _expand_page(di_rel)
                if di_text:
                    expanded_pages = list(expanded_pages or []) + [{
                        "path": di_rel,
                        "url": _pub(di_rel),
                        "text": di_text,
                    }]
                    logger.info("page_expand: wiring datasource %s", di_rel)
                    break
        for sink_rel in (
            "beginners_guide/data_at_rest.md",
            "beginners_guide/data_ingestion.md",
            "Datasets.md",
        ):
            if any((p.get("path") or "") == sink_rel for p in (expanded_pages or [])):
                continue
            sink_text = _expand_page(sink_rel)
            if sink_text:
                expanded_pages = list(expanded_pages or []) + [{
                    "path": sink_rel,
                    "url": _pub(sink_rel),
                    "text": sink_text,
                }]
                logger.info("page_expand: wiring sink page %s", sink_rel)
                break
    # Agentic / Fabio capability asks: load Copilot docs so we don't false-abstain
    if _question_names_agentic(question) or _is_capability_overclaim_ask(question):
        from page_expand import expand_page as _expand_page
        from doc_urls import public_doc_url as _pub

        for rel in (
            "ai_fabric/fabio_copilot.md",
            "ai_fabric/agentic_building_guide.md",
            "ai_fabric/how_to_build_agents.md",
            "ai_fabric/index.md",
        ):
            if any((p.get("path") or "") == rel for p in (expanded_pages or [])):
                continue
            text = _expand_page(rel)
            if text:
                expanded_pages = list(expanded_pages or []) + [{
                    "path": rel,
                    "url": _pub(rel),
                    "text": text,
                }]
                logger.info("page_expand: agentic page %s", rel)
                if len([p for p in expanded_pages if "ai_fabric" in (p.get("path") or "")]) >= 2:
                    break
    page_expanded = bool(expanded_pages)
    if expanded_pages:
        expanded_kb = pages_to_kb_entries(expanded_pages)
        kb_entries = (expanded_kb + list(kb_entries or []))[:14]

    ungrounded_entities = _ungrounded_named_entities(question, kb_entries, chunks)
    # Wiring ask naming an unknown product: treat as ungrounded BEFORE depth-forcing generate.
    wiring_miss = _ungrounded_wiring_products(question, kb_entries, chunks)
    for term in wiring_miss:
        if term not in ungrounded_entities:
            ungrounded_entities.append(term)
    named_entity_ungrounded = bool(ungrounded_entities)
    if named_entity_ungrounded:
        logger.info("named entities ungrounded in excerpts: %s", ungrounded_entities)

    if wiring_miss:
        # Honesty gate before path_first / critique depth-forcing can invent a narrative.
        logger.info(
            "ungrounded wiring product(s) — honesty before depth-force: %s",
            wiring_miss,
        )
        answer_text, gaps = _ungrounded_entity_honesty(wiring_miss, [])
        timing["generate_ms"] = 0
        timing["total_ms"] = _ms(t0)
        _log_gap(question, scope, gaps, True)
        sources = (_sources_from_kb(kb_entries) + _sources_from_chunks(chunks))[:8]
        return AgentResponse(
            answer=polish_answer_text(answer_text),
            sources=sources,
            sufficient=False,
            examples=[],
            gaps=gaps,
            scope=scope,
            used_inference=False,
            inferred_summary="",
            timing=timing,
        )

    # Live-incident playbook asks: do not invent a Documented Fabrix path from
    # loosely related memory/worker docs unless a real runbook is in excerpts.
    if _is_live_incident_ask(question) and not _live_incident_has_documented_procedure(
        question, kb_entries, chunks
    ):
        logger.info("live-incident ask without documented procedure — honesty gate")
        answer_text, gaps = _live_incident_honesty(
            question, kb_entries, chunks, []
        )
        timing["generate_ms"] = 0
        timing["total_ms"] = _ms(t0)
        _log_gap(question, scope, gaps, True)
        # Keep related sources for "what IS documented" transparency, but do not
        # claim a grounded executable playbook.
        sources = (_sources_from_kb(kb_entries) + _sources_from_chunks(chunks))[:8]
        return AgentResponse(
            answer=polish_answer_text(answer_text),
            sources=sources,
            sufficient=False,
            examples=[],
            gaps=gaps,
            scope=scope if (kb_entries or chunks) else "related",
            used_inference=False,
            inferred_summary="",
            timing=timing,
        )

    specificity_gap = _specificity_detail_missing(question, kb_entries, chunks)
    if specificity_gap:
        logger.info(
            "specificity gap kind=%s detail=%s targets=%s",
            specificity_gap.get("kind"),
            specificity_gap.get("detail"),
            specificity_gap.get("targets"),
        )

    def _prompt(revision_notes: str = "") -> str:
        return build_kb_prompt(
            question,
            kb_entries,
            chunks,
            scope,
            primary_objects=primary_objects,
            revision_notes=revision_notes,
            ungrounded_named_entities=ungrounded_entities or None,
        )

    prompt = _prompt()
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

    # Retry when abstained with context, or synthesis/related missing inference label.
    # Do not pressure the model to "answer anyway" when a named entity is missing.
    need_retry = (kb_entries or chunks) and not named_entity_ungrounded and (
        abstained
        or ((scope == "related" or _is_synthesis_question(question)) and not used_inference)
    )
    if need_retry:
        retry_extra = ""
        if specificity_gap:
            retry_extra = (
                f"The excerpts cover the general topic but NOT {specificity_gap['detail']}. "
                "Answer what IS documented about the topic, then clearly state that specific "
                "detail is not specified in the public docs. Do NOT invent versions, limits, "
                "or third-party comparisons. "
            )
        retry_prompt = (
            _prompt()
            + "\n\nIMPORTANT: Do not abstain if the excerpts cover the Fabrix topic. "
            + retry_extra
            + "Use **Documented Fabrix path** as a non-numbered header, then steps 1. 2. 3. "
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
    if abstained and (kb_entries or chunks) and not named_entity_ungrounded:
        force_extra = ""
        if specificity_gap:
            force_extra = (
                f"The question asks for {specificity_gap['detail']} which is NOT in the "
                "excerpts. Still answer the documented Fabrix topic that IS present, then "
                "explicitly say the specific detail is not specified in the public docs. "
                "Never invent a version number or a comparison to a named third-party tool. "
            )
        elif _is_exhaustive_ask(question) or _is_capability_overclaim_ask(question):
            force_extra = (
                "The question asks for more than the docs fully guarantee. "
                "Still answer with the documented Fabrix objects/operators/paths from the excerpts. "
                "For Fabio/Copilot/Agentic asks: name Fabio Copilot, Personas, and Toolsets from the "
                "excerpts; clearly state that end-to-end auto-remediation with no human approval is "
                "not documented; put that limit in gaps[]. "
                "Set used_inference=true and include Next (inferred) for handoffs. "
            )
        force_prompt = (
            _prompt()
            + "\n\nCRITICAL: Abstaining is forbidden. The knowledge base excerpts above ARE "
            "relevant. Answer the Fabrix topic in the question using those excerpts. "
            + force_extra
            + "Ignore presentation constraints (exact step counts / blank lines) if needed. "
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

    # Local salvage when model still abstains despite retrieved docs
    # Skip when a specificity gap: salvage dumps topic snippets and looks like overclaim.
    if (
        abstained
        and (kb_entries or chunks)
        and not named_entity_ungrounded
        and not specificity_gap
    ):
        salvaged = _salvage_partial_answer(question, kb_entries, chunks, gaps)
        if salvaged:
            answer_text, gaps = salvaged
            abstained = False
            used_inference = True
            inferred_summary = inferred_summary or (
                "Documented subset surfaced after model abstained on a partial-coverage ask"
            )
            logger.info("salvage_partial_answer applied")

    if named_entity_ungrounded and not _answer_acknowledges_ungrounded(
        answer_text, ungrounded_entities
    ):
        answer_text, gaps = _ungrounded_entity_honesty(ungrounded_entities, gaps)
        abstained = True
        used_inference = False
        inferred_summary = ""
        examples = []
        logger.info("named-entity honesty fallback applied for %s", ungrounded_entities)

    # Ops critique only when local draft checks fail
    if use_facets and not abstained and (kb_entries or chunks):
        ungrounded_bots = _ungrounded_bot_tokens(answer_text, kb_entries, chunks)
        wiring_ctx = _retrieved_context_blob(kb_entries, chunks)
        if (
            _draft_looks_clean(answer_text, used_inference, question)
            and not _datasource_ask_missing_sink(question, answer_text)
            and not _off_family_bot_tokens(question, answer_text)
            and not ungrounded_bots
            and not (
                _question_names_agentic(question) and not _answer_mentions_agentic(answer_text)
            )
            and not (
                _is_platform_install_ask(question)
                and (
                    _answer_has_integration_prereq_drift(answer_text)
                    or (
                        (
                            _upgrade_answer_missing_cli_path(answer_text)
                            or _upgrade_answer_stuck_on_cred_or_studio(answer_text)
                        )
                        if _is_platform_upgrade_ask(question)
                        else _install_answer_missing_hardware(answer_text)
                    )
                )
            )
            and not (
                page_expanded
                and _is_param_or_procedure_ask(question)
                and _answer_claims_missing_from_excerpts(answer_text)
            )
            and not (
                _is_pipeline_schedule_ask(question)
                and (
                    _schedule_answer_invents_bot(answer_text)
                    or _schedule_answer_missing_cron(answer_text)
                )
            )
            and not _agentic_overclaim_without_hedge(question, answer_text)
            and not _wiring_answer_missing_product_bot(question, answer_text, wiring_ctx)
            and not _wiring_answer_missing_shape(question, answer_text)
            and not _wiring_answer_missing_source_bot(question, answer_text, wiring_ctx)
        ):
            logger.info("critique skipped (draft looks clean)")
        else:
            if _datasource_ask_missing_sink(question, answer_text):
                logger.info("critique forced: datasource ask missing stream/dataset sink")
            if _wiring_answer_missing_shape(question, answer_text):
                logger.info("critique forced: wiring ask missing auth/bot/sink shape")
            if _wiring_answer_missing_product_bot(question, answer_text, wiring_ctx):
                logger.info("critique forced: wiring ask missing concrete @family:op bot")
            if _wiring_answer_missing_source_bot(question, answer_text, wiring_ctx):
                logger.info("critique forced: wiring ask missing Source bot for into-Fabrix path")
            off_bots = _off_family_bot_tokens(question, answer_text)
            if off_bots:
                logger.info("critique forced: off-family bots=%s", off_bots)
            if ungrounded_bots:
                logger.info("critique forced: ungrounded bots=%s", ungrounded_bots)
            if _is_platform_install_ask(question) and (
                _answer_has_integration_prereq_drift(answer_text)
                or (
                    (
                        _upgrade_answer_missing_cli_path(answer_text)
                        or _upgrade_answer_stuck_on_cred_or_studio(answer_text)
                    )
                    if _is_platform_upgrade_ask(question)
                    else _install_answer_missing_hardware(answer_text)
                )
            ):
                logger.info("critique forced: install/upgrade answer missing platform facet")
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
            if ungrounded_bots:
                critique["fix_needed"] = True
                note = (
                    f"Remove or replace bot tokens not present in the excerpts: {ungrounded_bots}. "
                    "Only name bots that appear in the retrieved documentation; "
                    "put missing exact bot names in gaps[]."
                )
                critique["revision_notes"] = (
                    (critique.get("revision_notes") or "") + " " + note
                ).strip()
            if _question_names_agentic(question) and not _answer_mentions_agentic(answer_text):
                critique["fix_needed"] = True
                note = (
                    "Name Fabio Copilot / Agentic AI (Persona/Toolset as documented) explicitly "
                    "in the answer; put unsupported auto-remediation / no-human-approval claims in gaps[]."
                )
                critique["revision_notes"] = (
                    (critique.get("revision_notes") or "") + " " + note
                ).strip()
            if (
                page_expanded
                and _is_param_or_procedure_ask(question)
                and _answer_claims_missing_from_excerpts(answer_text)
            ):
                critique["fix_needed"] = True
                note = (
                    "FULL DOC PAGES were loaded in context — parameter tables, cron schedules, "
                    "and procedure steps are available there. Do not claim they are missing from "
                    "excerpts; cite the full page content."
                )
                critique["revision_notes"] = (
                    (critique.get("revision_notes") or "") + " " + note
                ).strip()
            if _is_pipeline_schedule_ask(question) and (
                _schedule_answer_invents_bot(answer_text)
                or _schedule_answer_missing_cron(answer_text)
            ):
                critique["fix_needed"] = True
                note = (
                    "Do not invent bots like @c:pipeline-scheduler or @c:schedule-pipeline. "
                    "Schedule pipelines via service blueprint `scheduled_pipelines` with a "
                    "cron_expression (e.g. */15 * * * *) from beginners_guide/scheduled_pipelines. "
                    "Cite that YAML shape; put undocumented UI clicks in gaps[]."
                )
                critique["revision_notes"] = (
                    (critique.get("revision_notes") or "") + " " + note
                ).strip()
            if _agentic_overclaim_without_hedge(question, answer_text):
                critique["fix_needed"] = True
                note = (
                    "Do not claim Fabio Copilot auto-remediates production outages end-to-end "
                    "with no human approval unless excerpts say so. State what is documented "
                    "(Personas/Toolsets/Copilot), put unsupported auto-remediation claims in gaps[], "
                    "and keep a clear honesty hedge."
                )
                critique["revision_notes"] = (
                    (critique.get("revision_notes") or "") + " " + note
                ).strip()
            if (
                _wiring_answer_missing_shape(question, answer_text)
                or _wiring_answer_missing_product_bot(question, answer_text, wiring_ctx)
                or _wiring_answer_missing_source_bot(question, answer_text, wiring_ctx)
            ):
                critique["fix_needed"] = True
                source_hint = ""
                src_toks = _wiring_source_bot_tokens_in_context(question, wiring_ctx)
                if src_toks and _wiring_answer_missing_source_bot(
                    question, answer_text, wiring_ctx
                ):
                    sample = ", ".join(f"@{t}" for t in src_toks[:4])
                    source_hint = (
                        f" For into-Fabrix / dashboard / dataset asks, prefer Source bots "
                        f"from the catalog (e.g. {sample}) — do not lead with Sink bots that "
                        f"only write back into the external product."
                    )
                note = (
                    "REQUIRED wiring shape from excerpts only: "
                    "(1) credentials/auth cue for the named product, "
                    "(2) at least one concrete `@family:op` bot token from that product's "
                    "catalog page in context when present (never invent `@c:…` or prose-only "
                    "bot names), "
                    "(3) the sink the user asked for (stream/pstream/dataset/dashboard). "
                    "If an exact bot token is not in the excerpts, put it in gaps[] — do not invent."
                    + source_hint
                )
                critique["revision_notes"] = (
                    (critique.get("revision_notes") or "") + " " + note
                ).strip()
            if _is_platform_upgrade_ask(question) and (
                _answer_has_integration_prereq_drift(answer_text)
                or _upgrade_answer_missing_cli_path(answer_text)
                or _upgrade_answer_stuck_on_cred_or_studio(answer_text)
            ):
                critique["fix_needed"] = True
                note = (
                    "This is an RDAF platform upgrade/update question (including 'update platform' "
                    "+ VM wording). Do NOT lead with update_image_repository / docker1.cloudfabrix.io "
                    "credential rotation or RDA Studio 8GB/50GB laptop prereqs as the upgrade path. "
                    "Lead with installation_guides/rdaf_cli (rdafk8s if Kubernetes): "
                    "rdaf --version / rdaf status → rdaf backup → upgrade CLI via versioned "
                    "rdafcli-*.tar.gz pip install → rdaf registry fetch (docker2.cloudfabrix.io) → "
                    "rdaf infra/platform/app/worker upgrade --tag → validate (status / start-stop). "
                    "For VMs, cite deployment guide platform sizing (multi-VM roles), not Studio-only. "
                    "Do not claim the docs lack upgrade commands — they document them. "
                    "Put release-specific tag numbers and org-only Linux admin tips in gaps[]."
                )
                critique["revision_notes"] = (
                    (critique.get("revision_notes") or "") + " " + note
                ).strip()
            elif _is_platform_install_ask(question) and (
                _answer_has_integration_prereq_drift(answer_text)
                or _install_answer_missing_hardware(answer_text)
            ):
                critique["fix_needed"] = True
                note = (
                    "This is a platform install / VM (virtual machine) / hardware prerequisites "
                    "question. Remove ServiceNow/Qualys/Teams/Crowdstrike integration credential "
                    "prerequisites. Lead with RDA Studio installation_guides "
                    "(https://docs.fabrix.ai/installation_guides/): CPU, memory/RAM, disk, Docker, "
                    "Docker Compose, Python/pip, registry, ports. CLI/Kubernetes paths are secondary. "
                    "Put missing production sizing numbers in gaps[]."
                )
                critique["revision_notes"] = (
                    (critique.get("revision_notes") or "") + " " + note
                ).strip()
            if specificity_gap and (
                _answer_invents_specificity(answer_text, specificity_gap)
                or not _answer_acknowledges_specificity_gap(answer_text)
            ):
                critique["fix_needed"] = True
                critique["overclaim"] = True
                detail = specificity_gap.get("detail") or "the requested specific detail"
                note = (
                    f"Topic-documented ≠ detail-documented: {detail} is not in the excerpts. "
                    "Describe only what Fabrix docs state about the general topic, then clearly "
                    "say the specific detail is not specified / a direct third-party comparison "
                    "isn't in the public docs. Do NOT invent version numbers or rival bake-offs."
                )
                critique["revision_notes"] = (
                    (critique.get("revision_notes") or "") + " " + note
                ).strip()
            if _question_names_agentic(question) and not _answer_mentions_agentic(answer_text):
                logger.info("critique forced: agentic markers missing from answer")
            logger.info(
                "critique fix_needed=%s overclaim=%s notes=%s",
                critique.get("fix_needed"),
                critique.get("overclaim"),
                (critique.get("revision_notes") or "")[:160],
            )
            if critique.get("fix_needed") and critique.get("revision_notes"):
                rev_prompt = _prompt(revision_notes=critique["revision_notes"])
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

    # Local agentic naming revision (public: name Fabio/Copilot when asked)
    if (
        _question_names_agentic(question)
        and (kb_entries or chunks)
        and (abstained or not _answer_mentions_agentic(answer_text))
    ):
        note = (
            "Do not abstain. Name Fabio Copilot / Agentic AI (Persona/Toolset as "
            "documented) explicitly in the answer. If end-to-end auto-remediation with "
            "no human approval is not documented, say so clearly and put that limit in "
            "gaps[]."
        )
        logger.info("agentic naming revision")
        rev_prompt = _prompt(revision_notes=note)
        t_ag = time.perf_counter()
        raw_answer = generate(rev_prompt) or raw_answer
        timing["llm_calls"] += 1
        timing["critique_ms"] = timing.get("critique_ms", 0.0) + _ms(t_ag)
        parsed = parse_unified_response(raw_answer)
        answer_text = parsed["answer"]
        examples = parsed["examples"] or examples
        gaps = parsed["gaps"] or gaps
        used_inference = parsed["used_inference"] or used_inference
        inferred_summary = parsed["inferred_summary"] or inferred_summary
        abstained = _looks_like_abstention(answer_text)

    # Local bot-fidelity revision even when critique path was skipped (e.g. in_scope)
    off_bots_final = _off_family_bot_tokens(question, answer_text)
    ungrounded_final = (
        _ungrounded_bot_tokens(answer_text, kb_entries, chunks)
        if not abstained and (kb_entries or chunks)
        else []
    )
    if (off_bots_final or ungrounded_final) and not abstained and (kb_entries or chunks):
        notes = []
        if off_bots_final:
            allowed = ", ".join(_integration_family_hits(question)) or "named products"
            notes.append(
                f"Remove bots with prefixes {off_bots_final} — wrong product family. "
                f"Only use bots for: {allowed}."
            )
        if ungrounded_final:
            notes.append(
                f"Remove bot tokens not in the excerpts: {ungrounded_final}. "
                "Only name bots that appear in retrieved docs; missing names go in gaps[]."
            )
        note = " ".join(notes)
        logger.info(
            "bot fidelity revision: off-family=%s ungrounded=%s",
            off_bots_final,
            ungrounded_final,
        )
        rev_prompt = _prompt(revision_notes=note)
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

    # If LLM revision still invents ungrounded bots, strip them deterministically
    still_ungrounded = (
        _ungrounded_bot_tokens(answer_text, kb_entries, chunks)
        if not abstained and (kb_entries or chunks)
        else []
    )
    if still_ungrounded:
        answer_text = _strip_ungrounded_bot_mentions(answer_text, still_ungrounded)
        examples = [
            _strip_ungrounded_bot_mentions(ex, still_ungrounded) for ex in (examples or [])
        ]
        gap = (
            "Invented bot names were removed because they do not appear in retrieved excerpts: "
            + ", ".join(still_ungrounded)
        )
        if gap not in (gaps or []):
            gaps = [gap] + list(gaps or [])
        used_inference = True
        logger.info("stripped ungrounded bots after revision: %s", still_ungrounded)

    # Critique/revision can re-abstain on Fabio capability asks — deterministic honesty
    if (
        abstained
        and (kb_entries or chunks)
        and (
            _question_names_agentic(question) or _is_capability_overclaim_ask(question)
        )
    ):
        answer_text, gaps = _agentic_honesty_fallback(question, gaps)
        abstained = False
        used_inference = True
        inferred_summary = inferred_summary or (
            "Documented Agentic/Copilot objects surfaced after model abstained on a "
            "capability overclaim ask"
        )
        logger.info("agentic honesty fallback applied")

    if named_entity_ungrounded and not _answer_acknowledges_ungrounded(
        answer_text, ungrounded_entities
    ):
        answer_text, gaps = _ungrounded_entity_honesty(ungrounded_entities, gaps)
        abstained = True
        used_inference = False
        inferred_summary = ""
        examples = []
        logger.info("named-entity honesty fallback (post-revision) for %s", ungrounded_entities)

    if specificity_gap and (
        _answer_invents_specificity(answer_text, specificity_gap)
        or not _answer_acknowledges_specificity_gap(answer_text)
        or _looks_like_abstention(answer_text)
    ):
        answer_text, gaps = _specificity_honesty_fallback(question, specificity_gap, gaps)
        abstained = False
        used_inference = True
        inferred_summary = inferred_summary or (
            f"Documented topic separated from undocumented {specificity_gap.get('detail')}"
        )
        examples = []
        logger.info(
            "specificity honesty fallback applied kind=%s", specificity_gap.get("kind")
        )

    if _is_dashboard_pipeline_kickoff_ask(question):
        low_kick = (answer_text or "").lower()
        hedged = any(
            x in low_kick
            for x in (
                "not documented",
                "no documented",
                "isn't documented",
                "is not documented",
                "aren't documented",
                "are not documented",
                "one-way",
                "one way",
                "do not document",
                "don't document",
            )
        )
        if (not hedged) or ("dashboard-trigger" in low_kick) or _looks_like_abstention(
            answer_text
        ):
            answer_text, gaps = _dashboard_kickoff_honesty(gaps)
            abstained = False
            used_inference = True
            inferred_summary = inferred_summary or (
                "Dashboard→pipeline remediation kickoff is not documented"
            )
            examples = []
            logger.info("dashboard kickoff honesty fallback applied")

    if _is_pipeline_schedule_ask(question) and (
        _looks_like_abstention(answer_text)
        or _schedule_answer_missing_cron(answer_text)
    ):
        answer_text, gaps = _schedule_debug_honesty(gaps)
        abstained = False
        used_inference = True
        inferred_summary = inferred_summary or (
            "Documented scheduled_pipelines + cron_expression path for schedule debug"
        )
        examples = []
        logger.info("schedule debug honesty fallback applied")

    if _is_concurrent_dataset_ask(question):
        # Always replace — models invent locking/concurrency mechanisms with soft hedges.
        answer_text, gaps = _concurrent_dataset_honesty(gaps)
        abstained = False
        used_inference = True
        inferred_summary = inferred_summary or (
            "Concurrent dataset write behavior is not documented"
        )
        examples = []
        logger.info("concurrent dataset honesty fallback applied")

    if _is_full_script_automation_ask(question) and (
        _answer_has_turnkey_script_risk(answer_text)
        or _looks_like_abstention(answer_text)
    ):
        answer_text, gaps = _full_script_honesty_fallback(question, gaps)
        abstained = False
        used_inference = True
        inferred_summary = inferred_summary or (
            "Documented upgrade commands require environment-specific values; "
            "no turnkey script from public docs"
        )
        examples = []
        logger.info("full-script invention/abstain fallback applied")

    answer_text = polish_answer_text(answer_text)
    answer_text = _polish_known_product_typos(answer_text)
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
        sources = _filter_sources_for_install_ask(sources, question)

    if abstained and not gaps:
        gaps = ["No direct answer found in the retrieved documentation"]

    # Capability / exhaustive asks must surface gaps even when the model forgot
    if (
        not abstained
        and grounded
        and (_is_capability_overclaim_ask(question) or _is_exhaustive_ask(question))
        and not gaps
    ):
        gaps = [
            "Public documentation does not fully support this end-to-end or exhaustive ask"
        ]
        used_inference = True
        logger.info("gaps latch: capability/exhaustive ask missing gaps[]")

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
