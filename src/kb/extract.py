"""
kb/extract.py: rule-based KB extraction from public Fabrix markdown docs.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clean_markdown import clean_markdown, extract_bot_metadata
from config import (
    BOTS_DIR,
    CFXQL_FILE,
    DOCS_INCLUDE_DIRS,
    DOCS_ROOT,
    DOCS_ROOT_FILES,
)
from doc_urls import BOTS_REL_PREFIX, public_doc_url
from kb.schema import Entity, Fact, KnowledgeBase, Procedure, Relation, Topic
from query_qdrant import parse_bot_param_table

_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
_NUMBERED_STEP_RE = re.compile(r"^\s*(?:\d+[\.\)]\s+|[-*]\s+)(.+)$", re.MULTILINE)
_EXAMPLE_RE = re.compile(
    r"(?:Example(?:\s+Usage)?|For example)\s*:?\s*\n?(.*?)(?=\n#{1,4}\s|\n\n\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return s[:80] or "item"


def _first_paragraph(text: str, limit: int = 400) -> str:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    for p in parts:
        if p.startswith("#") or p.startswith("|") or p.startswith("```"):
            continue
        return p[:limit]
    return (text[:limit] if text else "").strip()


def _extract_example(text: str) -> str | None:
    m = _EXAMPLE_RE.search(text)
    if not m:
        # fenced code block as example
        code = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
        if code:
            return code.group(1).strip()[:500]
        return None
    ex = m.group(1).strip()
    return ex[:500] if ex else None


def _extract_steps(text: str) -> list[str]:
    steps = []
    for m in _NUMBERED_STEP_RE.finditer(text):
        step = m.group(1).strip()
        if 8 <= len(step) <= 300:
            steps.append(step)
        if len(steps) >= 15:
            break
    return steps


def _topic_for_section(section: str) -> Topic:
    names = {
        "beginners_guide": "Beginners Guide & Platform Concepts",
        "installation_guides": "Installation & Operations",
        "reference_guides": "Reference (CFXQL, Grok, Playground)",
        "Pipelines": "Example Pipelines",
        "Datasource_Integrations": "Datasource Integrations",
        "ai_fabric": "AI Fabric",
        "Extensions": "Extensions Catalog",
        "rda_releases": "RDA / OIA Releases",
    }
    return Topic(
        id=_slug(section),
        name=names.get(section, section.replace("_", " ").title()),
        section=section,
        summary=f"Documentation section covering {section.replace('_', ' ')}.",
        doc_paths=[],
    )


def _format_params_fact_text(bot_name: str, rows: list[dict]) -> str:
    """Embeddable summary of structured bot parameters for retrieve_kb."""
    parts = []
    for r in rows:
        name = r.get("name") or ""
        if not name:
            continue
        req = "required" if r.get("required") else "optional"
        typ = r.get("type") or ""
        default = r.get("default") or ""
        desc = (r.get("description") or "")[:120]
        bit = f"{name} ({req}"
        if typ:
            bit += f", {typ}"
        if default:
            bit += f", default {default}"
        bit += ")"
        if desc:
            bit += f": {desc}"
        parts.append(bit)
    joined = "; ".join(parts)
    return f"{bot_name} parameters: {joined}"[:4000]


def extract_bots(kb: KnowledgeBase) -> None:
    if not BOTS_DIR or not os.path.isdir(BOTS_DIR):
        return
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "h2")])
    for filename in sorted(f for f in os.listdir(BOTS_DIR) if f.endswith(".md")):
        filepath = os.path.join(BOTS_DIR, filename)
        with open(filepath, encoding="utf-8") as f:
            cleaned = clean_markdown(f.read())
        ext_slug = filename.removesuffix(".md")
        ext_id = f"extension-{_slug(ext_slug)}"
        kb.entities.append(
            Entity(
                id=ext_id,
                kind="extension",
                name=ext_slug,
                summary=f"Bot extension catalog page {filename}",
                section="Bots",
                source=filename,
                url=public_doc_url(f"{BOTS_REL_PREFIX}/{filename}"),
            )
        )
        for chunk in splitter.split_text(cleaned):
            h2 = chunk.metadata.get("h2", "")
            if not h2.startswith("Bot "):
                continue
            meta = extract_bot_metadata(h2)
            bot_name = meta["bot_name"]
            body = chunk.page_content.strip()
            summary = _first_paragraph(body, 350)
            example = _extract_example(body)
            param_rows = parse_bot_param_table(body)
            entity_id = f"bot-{_slug(bot_name)}"
            kb.entities.append(
                Entity(
                    id=entity_id,
                    kind="bot",
                    name=bot_name,
                    summary=summary or f"RDA bot {bot_name}",
                    section="Bots",
                    source=filename,
                    url=public_doc_url(f"{BOTS_REL_PREFIX}/{filename}"),
                    metadata={
                        "prefix": meta["prefix"],
                        "example": example or "",
                        "extension": ext_slug,
                        "parameters": param_rows,
                        "param_names": [r.get("name") for r in param_rows if r.get("name")],
                    },
                )
            )
            kb.relations.append(
                Relation(
                    id=f"rel-{entity_id}-{ext_id}",
                    from_id=entity_id,
                    to_id=ext_id,
                    relation="bot_in_extension",
                )
            )
            if param_rows:
                kb.facts.append(
                    Fact(
                        id=f"fact-params-{entity_id}",
                        text=_format_params_fact_text(bot_name, param_rows),
                        source=filename,
                        url=public_doc_url(f"{BOTS_REL_PREFIX}/{filename}"),
                        section="Bots",
                        entity_id=entity_id,
                        example=example,
                    )
                )


def extract_narrative_file(kb: KnowledgeBase, filepath: str, rel_source: str, section: str) -> None:
    with open(filepath, encoding="utf-8") as f:
        cleaned = clean_markdown(f.read())
    if not cleaned.strip():
        return

    url = public_doc_url(rel_source)
    topic = next((t for t in kb.topics if t.section == section), None)
    if topic is None:
        topic = _topic_for_section(section)
        kb.topics.append(topic)
    if rel_source not in topic.doc_paths:
        topic.doc_paths.append(rel_source)

    title = os.path.basename(rel_source).removesuffix(".md")
    # Prefer first H1/H2 as title
    hm = _HEADER_RE.search(cleaned)
    if hm:
        title = hm.group(2).strip()

    summary = _first_paragraph(cleaned, 400)
    entity_id = f"guide-{_slug(rel_source)}"
    kind = "guide"
    if section == "Pipelines":
        kind = "pipeline"
    elif section == "Datasource_Integrations":
        kind = "integration"
    elif section == "Extensions":
        kind = "extension"

    kb.entities.append(
        Entity(
            id=entity_id,
            kind=kind,
            name=title,
            summary=summary or title,
            section=section,
            source=rel_source,
            url=url,
        )
    )
    if summary:
        kb.facts.append(
            Fact(
                id=f"fact-{entity_id}",
                text=summary,
                source=rel_source,
                url=url,
                section=section,
                entity_id=entity_id,
                example=_extract_example(cleaned),
            )
        )

    # Header sections as additional facts (cap to keep KB embeddable)
    parts = re.split(r"(?=^#{2,4}\s)", cleaned, flags=re.MULTILINE)
    header_facts = 0
    for part in parts:
        if header_facts >= 3:
            break
        hm2 = _HEADER_RE.match(part.strip())
        if not hm2:
            continue
        header = hm2.group(2).strip()
        body = _first_paragraph(part, 300)
        if not body or body == summary:
            continue
        fid = f"fact-{_slug(rel_source)}-{_slug(header)}"
        kb.facts.append(
            Fact(
                id=fid,
                text=f"{header}: {body}",
                source=rel_source,
                url=url,
                section=section,
                entity_id=entity_id,
                example=_extract_example(part),
            )
        )
        header_facts += 1

    if section == "installation_guides" or "install" in rel_source.lower() or "upgrade" in rel_source.lower():
        steps = _extract_steps(cleaned)
        if len(steps) >= 2:
            kb.procedures.append(
                Procedure(
                    id=f"proc-{_slug(rel_source)}",
                    title=title,
                    steps=steps,
                    source=rel_source,
                    url=url,
                    section=section,
                )
            )


def extract_cfxql(kb: KnowledgeBase) -> None:
    path = CFXQL_FILE
    if not path or not os.path.isfile(path):
        return
    rel = os.path.relpath(path, DOCS_ROOT).replace("\\", "/") if DOCS_ROOT else "reference_guides/cfxql.md"
    extract_narrative_file(kb, path, rel, "reference_guides")
    # Explicit CFXQL facts
    with open(path, encoding="utf-8") as f:
        text = clean_markdown(f.read())
    url = public_doc_url(rel)
    if "Full CFXQL" in text or "full cfxql" in text.lower():
        kb.facts.append(
            Fact(
                id="fact-cfxql-full",
                text="Full CFXQL supports rich SQL-like operators and Result Format / GET clause for querying data.",
                source=rel,
                url=url,
                section="reference_guides",
                example=_extract_example(text),
            )
        )
    if "Restricted CFXQL" in text or "restricted cfxql" in text.lower():
        kb.facts.append(
            Fact(
                id="fact-cfxql-restricted",
                text="Restricted CFXQL is a simpler parameter style (mainly = and AND) used by API bots.",
                source=rel,
                url=url,
                section="reference_guides",
            )
        )


def build_knowledge_base() -> KnowledgeBase:
    kb = KnowledgeBase()
    for section in DOCS_INCLUDE_DIRS:
        kb.topics.append(_topic_for_section(section))

    extract_bots(kb)
    extract_cfxql(kb)

    if DOCS_ROOT and os.path.isdir(DOCS_ROOT):
        cfxql_rel = None
        if CFXQL_FILE and os.path.isfile(CFXQL_FILE):
            cfxql_rel = os.path.relpath(CFXQL_FILE, DOCS_ROOT).replace("\\", "/")

        for subdir in DOCS_INCLUDE_DIRS:
            dir_path = os.path.join(DOCS_ROOT, subdir)
            if not os.path.isdir(dir_path):
                continue
            for root, _, files in os.walk(dir_path):
                for filename in sorted(files):
                    if not filename.endswith(".md"):
                        continue
                    filepath = os.path.join(root, filename)
                    rel = os.path.relpath(filepath, DOCS_ROOT).replace("\\", "/")
                    if cfxql_rel and rel == cfxql_rel:
                        continue
                    extract_narrative_file(kb, filepath, rel, subdir)

        for filename in DOCS_ROOT_FILES:
            filepath = os.path.join(DOCS_ROOT, filename)
            if os.path.isfile(filepath):
                extract_narrative_file(kb, filepath, filename, filename.removesuffix(".md"))

    # Topic related links (same section co-occurrence is enough for v1)
    by_section: dict[str, list[str]] = {}
    for t in kb.topics:
        by_section.setdefault(t.section, []).append(t.id)
    for section, ids in by_section.items():
        for tid in ids:
            topic = next(t for t in kb.topics if t.id == tid)
            topic.related_topics = [x for x in ids if x != tid][:5]

    return kb
