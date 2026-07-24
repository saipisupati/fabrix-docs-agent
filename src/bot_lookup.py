"""
bot_lookup.py — deterministic bot parameter lookups (0 LLM calls).

Handles extension-family catalogs (kafka-v2, servicenow_v2, …) where one markdown
file contains many bots; picks the right ## Bot section by family + operation hints.
"""

from __future__ import annotations

import logging
import os
import re

from clean_markdown import extract_bot_metadata
from config import BOTS_DIR
from doc_urls import BOTS_REL_PREFIX, public_doc_url
from page_expand import expand_page
from query_qdrant import bot_name_hints, parse_bot_param_table

logger = logging.getLogger(__name__)

_kb_bot_cache: list[dict] | None = None


def _load_kb_bot_entities() -> list[dict]:
    """Cached list of bot entities that have structured parameters."""
    global _kb_bot_cache
    if _kb_bot_cache is not None:
        return _kb_bot_cache
    try:
        from kb.store import load_kb
    except ImportError:
        _kb_bot_cache = []
        return _kb_bot_cache
    kb = load_kb()
    if kb is None:
        _kb_bot_cache = []
        return _kb_bot_cache
    out: list[dict] = []
    for e in kb.entities:
        if e.kind != "bot":
            continue
        params = (e.metadata or {}).get("parameters") or []
        if not params:
            continue
        out.append({
            "name": e.name,
            "source": e.source,
            "url": e.url,
            "extension": (e.metadata or {}).get("extension") or "",
            "parameters": params,
            "example": (e.metadata or {}).get("example") or "",
        })
    _kb_bot_cache = out
    logger.info("bot_lookup: loaded %s bot entities with params from KB", len(out))
    return out


def clear_kb_bot_cache() -> None:
    global _kb_bot_cache
    _kb_bot_cache = None


def explicit_bot_tokens(question: str) -> list[tuple[str, str]]:
    """Return (family, operation) pairs from explicit @family:op tokens in the question."""
    return [
        (m.group(1).lower(), m.group(2).lower())
        for m in BOT_TOKEN_RE.finditer(question or "")
    ]


def _bot_matches_families(bot: dict, families: list[str]) -> bool:
    if not families:
        return False
    name = (bot.get("name") or "").lower()
    source = (bot.get("source") or "").lower().removesuffix(".md")
    extension = (bot.get("extension") or "").lower()
    blob = f"{name} {source} {extension}"
    for fam in families:
        fl = fam.lower()
        variants = {fl, fl.replace("-", "_"), fl.replace("_", "-")}
        if any(v and v in blob for v in variants):
            return True
    return False


def _score_kb_bot(bot: dict, families: list[str], operation_hints: list[str]) -> float:
    name = (bot.get("name") or "").lower()
    source = (bot.get("source") or "").lower().removesuffix(".md")
    extension = (bot.get("extension") or "").lower()
    score = 0.0
    family_hit = False
    for fam in families:
        fl = fam.lower()
        if fl in name or fl in source or fl in extension:
            score += 40.0
            family_hit = True
        if fl.replace("-", "_") in name or fl.replace("_", "-") in name:
            score += 40.0
            family_hit = True
    # Never let a same-op bot from another product family win.
    if families and not family_hit:
        return 0.0
    for op in operation_hints:
        ol = op.lower()
        if ol in name:
            score += 35.0 + len(ol) * 0.5
    if bot.get("parameters"):
        score += 20.0
    return score


def lookup_bot_params_from_kb(
    families: list[str],
    operation_hints: list[str] | None = None,
    question: str | None = None,
) -> tuple[str, list[dict], str] | None:
    """
    Look up structured params from KB entities (Phase 3 ingest).
    Returns (bot_name, rows, source_filename) or None.

    When the question names an explicit @family:op token, prefer an exact name match
    and never return a bot from a different family.
    """
    if not families and not explicit_bot_tokens(question or ""):
        return None
    ops = operation_hints or []
    bots = _load_kb_bot_entities()
    if not bots:
        return None

    tokens = explicit_bot_tokens(question or "")
    if tokens:
        # Exact @family:op match first
        for fam, op in tokens:
            want = f"@{fam}:{op}".lower()
            for bot in bots:
                name = (bot.get("name") or "").lower()
                if name == want or name.lstrip("@#*") == want.lstrip("@"):
                    rows = bot.get("parameters") or []
                    if not rows:
                        continue
                    source = bot.get("source") or ""
                    logger.info(
                        "bot_lookup: kb exact token bot=%s rows=%s",
                        bot.get("name"),
                        len(rows),
                    )
                    return bot.get("name") or "", rows, source
        # Constrain candidates to the token's family (no Opsgenie-for-snowv2)
        token_fams = list(dict.fromkeys(f for f, _ in tokens))
        bots = [b for b in bots if _bot_matches_families(b, token_fams)]
        if not bots:
            logger.info("bot_lookup: no KB bots for explicit families=%s", token_fams)
            return None
        families = token_fams
        # Require operation overlap when an explicit op was named; otherwise abstain
        # to the LLM rather than returning a random same-family sibling.
        ranked = sorted(bots, key=lambda b: _score_kb_bot(b, families, ops), reverse=True)
        best = ranked[0]
        best_score = _score_kb_bot(best, families, ops)
        best_name = (best.get("name") or "").lower()
        op_hit = any(op.lower() in best_name for op in ops if op)
        if best_score < 40 or (ops and not op_hit):
            logger.info(
                "bot_lookup: explicit token without same-family op match "
                "families=%s ops=%s best=%s score=%s",
                families,
                ops,
                best.get("name"),
                best_score,
            )
            return None
        rows = best.get("parameters") or []
        if not rows:
            return None
        logger.info(
            "bot_lookup: kb family=%s bot=%s rows=%s (token-constrained)",
            families[0],
            best.get("name"),
            len(rows),
        )
        return best.get("name") or "", rows, best.get("source") or ""

    if not families:
        return None
    ranked = sorted(bots, key=lambda b: _score_kb_bot(b, families, ops), reverse=True)
    best = ranked[0]
    if _score_kb_bot(best, families, ops) < 40:
        return None
    rows = best.get("parameters") or []
    if not rows:
        return None
    bot_name = best.get("name") or ""
    source = best.get("source") or ""
    logger.info(
        "bot_lookup: kb family=%s bot=%s rows=%s",
        families[0],
        bot_name,
        len(rows),
    )
    return bot_name, rows, source


def kb_source_dict(source_filename: str, bot_name: str, url: str = "") -> dict:
    rel = source_filename
    if not rel.startswith(f"{BOTS_REL_PREFIX}/"):
        rel = f"{BOTS_REL_PREFIX}/{source_filename}" if source_filename else ""
    return {
        "title": bot_name,
        "url": url or (public_doc_url(rel) if rel else ""),
        "excerpt": f"Structured bot parameters from KB ({source_filename})",
    }

BOT_TOKEN_RE = re.compile(r"[@#*]([a-zA-Z0-9][a-zA-Z0-9_-]{0,60})\s*:\s*([a-zA-Z0-9][a-zA-Z0-9_-]*)")
_PARAM_TABLE_HEADER_RE = re.compile(r"^\|\s*Parameter\s+Name\s*\|", re.IGNORECASE)

# Generic operation verbs → bot name suffix fragments (longest match wins in scoring).
OPERATION_SUFFIX_HINTS: tuple[tuple[str, ...], str] = (
    (("read-stream", "read stream"), "read-stream"),
    (("write-stream", "write stream"), "write-stream"),
    (("poll-topic", "poll topic"), "poll-topic"),
    (("list-topics-partitions",), "list-topics-partitions"),
    (("list-topics", "list topics"), "list-topics"),
    (("list-groups", "list groups"), "list-groups"),
    (("alter-topic", "create topic", "update topic"), "alter-topic"),
    (("consume", "consumer", "read", "receive"), "read-stream"),
    (("write", "produce", "sink"), "write-stream"),
    (("poll",), "poll"),
    (("list",), "list-"),
)

_V2_NORMALIZE_RE = re.compile(
    r"\b([a-z][a-z0-9_-]*)\s+v\s*2\b",
    re.IGNORECASE,
)

# Product families for retrieve boost (mirrors agent INTEGRATION_FAMILIES; no agent import).
PRODUCT_FAMILY_ALIASES: list[tuple[str, tuple[str, ...]]] = [
    ("prometheus", ("prometheus", "prometheusv2", "prometheus_v2")),
    ("zabbix", ("zabbix",)),
    ("splunk", ("splunk",)),
    ("elastic", ("elastic", "elasticsearch", "opensearch", "elk")),
    ("servicenow", ("servicenow", "service now", "sn ticketing")),
    ("bmc", ("bmc-remedy", "bmc remedy", "bmc_remedy")),
    ("kubernetes", ("kubernetes", "k8s", "kubectl", "kubernetes-inventory")),
    ("vmware", ("vmware", "vcenter", "vrealize", "vrops")),
    ("pagerduty", ("pagerduty", "pager duty")),
    ("datadog", ("datadog",)),
    ("dynatrace", ("dynatrace",)),
    ("solarwinds", ("solarwinds", "solar winds")),
    ("newrelic", ("new relic", "newrelic", "new_relic")),
    ("appdynamics", ("appdynamics", "app dynamics", "appd")),
    ("nagios", ("nagios", "nagios xi", "nagiosxi")),
    ("kafka", ("kafka", "kafka-v2")),
    ("netapp", ("netapp", "netapp-eseries")),
    ("opsgenie", ("opsgenie", "ops genie", "ops-genie")),
    ("aws", ("aws", "amazon web services")),
    ("azure", ("azure", "microsoft azure")),
]


def is_bot_param_lookup(question: str) -> bool:
    q = (question or "").lower()
    if BOT_TOKEN_RE.search(question or ""):
        return True
    if any(w in q for w in ("parameter", "parameters")):
        return True
    if re.search(r"\b(take|takes|accepts|expects)\b", q):
        return True
    return q.strip().startswith(("what parameters", "which parameters"))


def bot_family_hints(question: str) -> list[str]:
    """Extension / bot family slugs from question text and explicit bot tokens."""
    hints: list[str] = []
    q = question or ""
    qlow = q.lower()

    token_ops = {op for _, op in explicit_bot_tokens(q)}
    # Hyphenated slugs from NL — but never treat an explicit @family:op suffix as a family
    # (that caused @snowv2:list-incidents → @opsgenie:list-incidents).
    for h in bot_name_hints(question):
        if h.lower() in token_ops:
            continue
        hints.append(h)

    for match in BOT_TOKEN_RE.finditer(q):
        family = match.group(1).lower()
        if family and family not in hints:
            hints.append(family)

    for match in _V2_NORMALIZE_RE.finditer(qlow):
        base = match.group(1).replace("_", "-")
        for variant in (f"{base}-v2", f"{base}_v2", f"{base}v2"):
            if variant not in hints:
                hints.append(variant)

    # Phase 4: product names (Zabbix, New Relic, …) become retrieve family hints
    for canonical, aliases in PRODUCT_FAMILY_ALIASES:
        if canonical == "servicenow":
            if (
                "servicenow" in qlow
                or "service now" in qlow
                or "sn ticketing" in qlow
                or re.search(r"\bsn\b", qlow)
                or re.search(r"\bsnow\b", qlow)
                or "snowv2" in qlow
            ):
                if canonical not in hints:
                    hints.append(canonical)
            continue
        if any(a in qlow for a in aliases):
            if canonical not in hints:
                hints.append(canonical)

    # De-dupe preserving order; prefer longer (more specific) slugs first for catalog lookup.
    seen: set[str] = set()
    ordered: list[str] = []
    for h in sorted(hints, key=len, reverse=True):
        key = h.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(h)
    return ordered


def bot_operation_hints(question: str) -> list[str]:
    """Suffix fragments to disambiguate bots within an extension catalog."""
    q = (question or "").lower()
    hints: list[str] = []

    for match in BOT_TOKEN_RE.finditer(question or ""):
        op = match.group(2).lower()
        if op and op not in hints:
            hints.append(op)

    for keywords, suffix in OPERATION_SUFFIX_HINTS:
        if any(kw in q for kw in keywords):
            if suffix not in hints:
                hints.append(suffix)

    return hints


def _chunk_family_slug(chunk: dict) -> str:
    meta = chunk.get("metadata") or {}
    bot_name = (meta.get("bot_name") or "").lower()
    source = (meta.get("source") or "").lower().removesuffix(".md")
    if bot_name:
        for prefix in ("@", "#", "*"):
            if bot_name.startswith(prefix):
                family = bot_name[1:].split(":", 1)[0]
                if family:
                    return family
    return source


def _family_matches(chunk: dict, families: list[str]) -> bool:
    if not families:
        return False
    meta = chunk.get("metadata") or {}
    bot_name = (meta.get("bot_name") or "").lower()
    source = (meta.get("source") or "").lower()
    extension = (meta.get("extension") or meta.get("family") or "").lower()
    slug = _chunk_family_slug(chunk)
    for fam in families:
        fl = fam.lower()
        variants = {fl, fl.replace("-", "_"), fl.replace("_", "-")}
        blob = f"{bot_name} {source} {extension} {slug}"
        if any(v in blob for v in variants):
            return True
    return False


def hybrid_boost_chunks(
    question: str,
    chunks: list[dict],
) -> list[dict]:
    """
    Phase 4 light hybrid: dense score + keyword family match on bot_name/source/extension.
    Demotes OS-inventory chunks when the question only mentions a host OS as red herring.
    """
    if not chunks:
        return chunks
    families = bot_family_hints(question)
    ops = bot_operation_hints(question)
    qlow = (question or "").lower()
    host_noise = any(
        w in qlow
        for w in ("debian", "ubuntu", "fedora", "centos", "rhel", "alpine", "linux host", "runs on")
    )
    # Bare host OS mention without linux-inventory product intent
    demote_linux = host_noise and "linux-inventory" not in qlow and "linux-os" not in qlow

    def sort_key(chunk: dict):
        meta = chunk.get("metadata") or {}
        bot_name = (meta.get("bot_name") or "").lower()
        source = (meta.get("source") or "").lower()
        dense = float(chunk.get("score") or 0)
        boost = 0.0
        if families and _family_matches(chunk, families):
            boost += 50.0
        for op in ops:
            if op.lower() in bot_name:
                boost += 20.0
        # Keyword hybrid: exact product token in source filename
        for fam in families:
            fl = fam.lower().replace("_", "-")
            if fl in source.replace("_", "-") or fl in bot_name:
                boost += 25.0
        if demote_linux and (
            "linux-inventory" in bot_name
            or "linux-inventory" in source
            or "linux-os" in bot_name
        ):
            boost -= 80.0
        return (boost + dense * 10.0, dense)

    return sorted(chunks, key=sort_key, reverse=True)


def score_chunk_for_lookup(
    chunk: dict,
    families: list[str],
    operation_hints: list[str],
) -> float:
    meta = chunk.get("metadata") or {}
    bot_name = (meta.get("bot_name") or "").lower()
    text = chunk.get("text") or ""
    score = float(chunk.get("score") or 0)

    if not _family_matches(chunk, families):
        return score - 100.0

    score += 50.0
    for fam in families:
        if fam.lower() in bot_name:
            score += 20.0

    for op in operation_hints:
        ol = op.lower()
        if ol in bot_name:
            score += 30.0 + len(ol) * 0.5

    if _PARAM_TABLE_HEADER_RE.search(text, re.MULTILINE):
        score += 25.0
    elif parse_bot_param_table(text):
        score += 25.0

    return score


def rerank_lookup_chunks(
    chunks: list[dict],
    families: list[str],
    operation_hints: list[str],
) -> list[dict]:
    if not chunks or not families:
        return chunks
    return sorted(
        chunks,
        key=lambda c: score_chunk_for_lookup(c, families, operation_hints),
        reverse=True,
    )


def prune_lookup_chunks_for_families(
    chunks: list[dict],
    families: list[str],
    operation_hints: list[str],
    *,
    limit: int = 5,
) -> list[dict]:
    if not chunks:
        return chunks
    ranked = rerank_lookup_chunks(chunks, families, operation_hints)
    matched = [c for c in ranked if _family_matches(c, families)]
    if not matched:
        return ranked[:limit]
    return matched[:limit]


def split_bot_catalog_sections(md_text: str) -> list[dict]:
    if not md_text:
        return []
    sections: list[dict] = []
    current_name = ""
    current_lines: list[str] = []

    for line in md_text.splitlines():
        if line.startswith("## Bot "):
            if current_name and current_lines:
                sections.append({
                    "bot_name": current_name,
                    "text": "\n".join(current_lines).strip(),
                })
            h2 = line.lstrip("#").strip()
            meta = extract_bot_metadata(h2)
            current_name = meta.get("bot_name") or h2.removeprefix("Bot ").strip()
            current_lines = [line]
        elif current_name:
            current_lines.append(line)

    if current_name and current_lines:
        sections.append({
            "bot_name": current_name,
            "text": "\n".join(current_lines).strip(),
        })
    return sections


def _section_has_param_table(text: str) -> bool:
    return bool(_PARAM_TABLE_HEADER_RE.search(text or "", re.MULTILINE))


def score_bot_section(
    section: dict,
    families: list[str],
    operation_hints: list[str],
) -> float:
    bot_name = (section.get("bot_name") or "").lower()
    text = section.get("text") or ""
    score = 0.0

    for fam in families:
        fl = fam.lower()
        if fl in bot_name or fl.replace("-", "_") in bot_name or fl.replace("_", "-") in bot_name:
            score += 40.0

    for op in operation_hints:
        ol = op.lower()
        if ol in bot_name:
            score += 35.0 + len(ol) * 0.5

    if _section_has_param_table(text):
        score += 30.0
    else:
        return score - 50.0

    return score


def pick_bot_section(
    sections: list[dict],
    families: list[str],
    operation_hints: list[str],
) -> dict | None:
    if not sections:
        return None
    ranked = sorted(
        sections,
        key=lambda s: score_bot_section(s, families, operation_hints),
        reverse=True,
    )
    best = ranked[0]
    if score_bot_section(best, families, operation_hints) < 10:
        return None
    return best


def _resolve_catalog_path(family_slug: str) -> str | None:
    slug = family_slug.strip()
    if not slug:
        return None
    variants = [
        slug,
        slug.replace("-", "_"),
        slug.replace("_", "-"),
    ]
    seen: set[str] = set()
    for variant in variants:
        if variant in seen:
            continue
        seen.add(variant)
        rel = f"{BOTS_REL_PREFIX}/{variant}.md"
        fpath = os.path.join(BOTS_DIR, f"{variant}.md")
        if os.path.isfile(fpath):
            return rel
    return None


def lookup_bot_params_from_catalog(
    family_slug: str,
    operation_hints: list[str] | None = None,
) -> tuple[str, list[dict], str] | None:
    """
    Load full bot catalog MD, pick section, parse param table.
    Returns (bot_name, rows, catalog_rel_path) or None.
    """
    rel_path = _resolve_catalog_path(family_slug)
    if not rel_path:
        return None
    text = expand_page(rel_path, max_chars=32_000)
    if not text:
        return None
    sections = split_bot_catalog_sections(text)
    ops = operation_hints or []
    section = pick_bot_section(sections, [family_slug], ops)
    if not section:
        return None
    rows = parse_bot_param_table(section.get("text") or "")
    if not rows:
        return None
    bot_name = section.get("bot_name") or family_slug
    logger.info(
        "bot_lookup: catalog family=%s bot=%s rows=%s",
        family_slug,
        bot_name,
        len(rows),
    )
    return bot_name, rows, rel_path


def format_param_answer(bot_name: str, rows: list[dict], section_text: str = "") -> str:
    required_rows = [r for r in rows if r["required"]]
    optional_rows = [r for r in rows if not r["required"]]
    title = bot_name.strip() if bot_name else "The bot"
    lines = [f"{title} takes the following parameters:"]
    for r in required_rows + optional_rows:
        req = " *" if r["required"] else ""
        desc = r["description"] or ""
        if r["default"]:
            desc = (desc + f" (default {r['default']})").strip()
        lines.append(f"- **{r['name']}**{req}: {desc}".strip())
    return "\n".join(lines).strip()


def extract_example_snippet(section_text: str) -> str | None:
    text = section_text or ""
    if "Example" in text:
        return text.split("Example", 1)[-1][:300].strip()
    return None


def best_chunk_lookup(
    chunks: list[dict],
    families: list[str],
    operation_hints: list[str],
) -> tuple[str, list[dict], str] | None:
    """Try ranked chunks; return (bot_name, rows, chunk_text) or None."""
    ranked = rerank_lookup_chunks(chunks, families, operation_hints)
    best: tuple[str, list[dict], str] | None = None
    best_score = -1.0
    for chunk in ranked:
        if not _family_matches(chunk, families):
            continue
        text = chunk.get("text") or ""
        rows = parse_bot_param_table(text)
        if not rows:
            continue
        score = score_chunk_for_lookup(chunk, families, operation_hints)
        if score > best_score:
            best_score = score
            bot_name = (chunk.get("metadata") or {}).get("bot_name") or ""
            best = (bot_name, rows, text)
    if best:
        logger.info(
            "bot_lookup: chunk family=%s bot=%s rows=%s",
            families[0] if families else "?",
            best[0],
            len(best[1]),
        )
    return best


def catalog_source_dict(rel_path: str, bot_name: str) -> dict:
    return {
        "title": bot_name,
        "url": public_doc_url(rel_path),
        "excerpt": f"Bot parameter table from {rel_path}",
    }
