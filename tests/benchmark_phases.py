"""
benchmark_phases.py — Phase 1–4 quality gates (page expand, bot fast path, KB params, family retrieve).

Run (stop API first — Qdrant lock):
  PYTHONPATH=src python3 tests/benchmark_phases.py

Exit 0 only when all phase checks PASS.
Writes tests/benchmark_phases_results.txt (gitignored).
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "benchmark_phases_results.txt")


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"id": name, "ok": ok, "detail": detail}


def phase1_page_expand() -> list[dict]:
    from page_expand import expand_context, expand_page, normalize_source_path

    results = []
    path = normalize_source_path("kafka-v2.md", chunk_type="bot")
    results.append(_check(
        "p1_normalize_bot_path",
        path == "Bots/kafka-v2.md" or (path or "").endswith("kafka-v2.md"),
        f"path={path}",
    ))

    text = expand_page("Bots/kafka-v2.md")
    results.append(_check(
        "p1_expand_kafka_page",
        bool(text) and "read-stream" in (text or "").lower() and "batch_size" in (text or "").lower().replace("\\_", "_"),
        f"chars={len(text or '')}",
    ))

    sched = expand_page("beginners_guide/scheduled_pipelines.md")
    results.append(_check(
        "p1_expand_scheduled_pipelines",
        bool(sched) and "scheduled_pipelines" in (sched or "").lower() and "cron" in (sched or "").lower(),
        f"chars={len(sched or '')}",
    ))

    # Synthetic retrieve agreement → expand_context
    kb_entries = [
        {"source": "Bots/kafka-v2.md", "title": "kafka", "text": "x", "kind": "fact", "url": ""},
        {"source": "Bots/kafka-v2.md", "title": "kafka2", "text": "y", "kind": "fact", "url": ""},
    ]
    chunks = [
        {"text": "params", "metadata": {"source": "kafka-v2.md", "type": "bot"}, "score": 0.9},
    ]
    pages = expand_context(kb_entries, chunks)
    results.append(_check(
        "p1_expand_context_agrees",
        len(pages) >= 1 and "kafka" in pages[0].get("path", "").lower(),
        f"pages={[p.get('path') for p in pages]}",
    ))
    return results


def phase2_bot_fast_path(client) -> list[dict]:
    from agent import answer
    from bot_lookup import (
        bot_family_hints,
        bot_operation_hints,
        is_bot_param_lookup,
        lookup_bot_params_from_catalog,
    )

    results = []
    results.append(_check(
        "p2_param_gate",
        is_bot_param_lookup("What parameters does the kafka-v2 consume / read bot take?")
        and not is_bot_param_lookup("Walk me through wiring Kafka into Fabrix"),
    ))
    results.append(_check(
        "p2_family_ops",
        "kafka-v2" in bot_family_hints("kafka-v2 consume / read bot")
        or any("kafka" in h for h in bot_family_hints("kafka-v2 consume / read bot")),
        f"ops={bot_operation_hints('consume / read')}",
    ))

    cat = lookup_bot_params_from_catalog("kafka-v2", ["read-stream"])
    results.append(_check(
        "p2_catalog_read_stream",
        cat is not None and "read-stream" in cat[0].lower()
        and {"name", "group", "offset_reset", "batch_size"}
        <= {r["name"] for r in cat[1]},
        f"bot={cat[0] if cat else None}",
    ))

    if client is None:
        results.append(_check("p2_answer_kafka_zero_llm", False, "no qdrant client"))
        results.append(_check("p2_answer_timed_loop_zero_llm", False, "no qdrant client"))
        return results

    t0 = time.perf_counter()
    r = answer("What parameters does the kafka-v2 consume / read bot take?", client=client)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    low = (r.answer or "").lower()
    llm = int((r.timing or {}).get("llm_calls") or 0)
    results.append(_check(
        "p2_answer_kafka_zero_llm",
        llm == 0
        and "read-stream" in low
        and all(x in low for x in ("name", "group", "offset_reset", "batch_size")),
        f"llm={llm} ms={ms}",
    ))
    results.append(_check(
        "p2_kafka_latency_under_3s",
        ms < 3000,
        f"ms={ms}",
    ))

    r2 = answer("What parameters does the timed-loop bot take?", client=client)
    llm2 = int((r2.timing or {}).get("llm_calls") or 0)
    low2 = (r2.answer or "").lower()
    results.append(_check(
        "p2_answer_timed_loop_zero_llm",
        llm2 == 0 and "interval" in low2,
        f"llm={llm2}",
    ))
    return results


def phase3_kb_params(client) -> list[dict]:
    from bot_lookup import clear_kb_bot_cache, lookup_bot_params_from_kb
    from kb.extract import extract_bots
    from kb.schema import KnowledgeBase
    from kb.store import load_kb

    results = []
    # Fresh extract (does not require rebuilt embeddings)
    kb_mem = KnowledgeBase()
    extract_bots(kb_mem)
    with_params = [
        e for e in kb_mem.entities
        if e.kind == "bot" and (e.metadata or {}).get("parameters")
    ]
    results.append(_check(
        "p3_extract_bots_with_params",
        len(with_params) >= 50,
        f"count={len(with_params)}",
    ))

    disk = load_kb()
    disk_ok = False
    if disk is not None:
        disk_bots = [
            e for e in disk.entities
            if e.kind == "bot" and (e.metadata or {}).get("parameters")
        ]
        disk_ok = len(disk_bots) >= 10
        results.append(_check(
            "p3_disk_kb_has_params",
            disk_ok,
            f"count={len(disk_bots)}",
        ))
    else:
        results.append(_check("p3_disk_kb_has_params", False, "kb.json missing — run build_kb.py"))

    clear_kb_bot_cache()
    hit = lookup_bot_params_from_kb(["kafka-v2"], ["read-stream"])
    # If disk KB is stale (no params), fall back to verifying extract-only scoring path
    if hit is None and not disk_ok:
        # Build ephemeral: temporarily save is too heavy; just fail with hint
        results.append(_check(
            "p3_kb_lookup_kafka",
            False,
            "rebuild kb.json with python3 src/build_kb.py",
        ))
    else:
        results.append(_check(
            "p3_kb_lookup_kafka",
            hit is not None
            and "read-stream" in hit[0].lower()
            and {"name", "group", "offset_reset", "batch_size"} <= {r["name"] for r in hit[1]},
            f"bot={hit[0] if hit else None}",
        ))

    if client is not None and hit is not None:
        from agent import answer

        r = answer("What parameters does the kafka-v2 consume / read bot take?", client=client)
        llm = int((r.timing or {}).get("llm_calls") or 0)
        src_blob = " ".join(
            f"{s.get('title')} {s.get('url')} {s.get('excerpt')}" for s in (r.sources or [])
        ).lower()
        results.append(_check(
            "p3_answer_uses_structured_params",
            llm == 0 and ("kb" in src_blob or "structured" in src_blob or "kafka-v2" in src_blob),
            f"llm={llm} sources={len(r.sources or [])}",
        ))
    else:
        results.append(_check(
            "p3_answer_uses_structured_params",
            hit is not None,
            "skipped live answer (no client or no kb hit)",
        ))
    return results


def phase4_family_retrieve(client) -> list[dict]:
    """Phase 4: product family hints + chunk filter demote OS-inventory contamination."""
    from agent import (
        _filter_chunks_to_families,
        _rank_chunks_for_product,
        answer,
    )
    from bot_lookup import bot_family_hints, hybrid_boost_chunks

    results = []
    hints = bot_family_hints(
        "How do I add Zabbix as a datasource? Our collector also runs on a Linux host if that matters."
    )
    results.append(_check(
        "p4_zabbix_family_hint",
        "zabbix" in [h.lower() for h in hints],
        f"hints={hints}",
    ))
    nr = bot_family_hints("Wire New Relic into Fabrix — edge box is Debian")
    results.append(_check(
        "p4_newrelic_family_hint",
        any("newrelic" in h.lower() or "new" in h.lower() for h in nr) or "newrelic" in nr,
        f"hints={nr}",
    ))

    mixed = [
        {
            "text": "linux inventory bot",
            "metadata": {"bot_name": "@linux-inventory:list", "source": "linux-inventory.md", "type": "bot"},
            "score": 0.95,
        },
        {
            "text": "zabbix alerts",
            "metadata": {"bot_name": "@zabbix:zabbix-alerts", "source": "zabbix.md", "type": "bot"},
            "score": 0.80,
        },
    ]
    q = "How do I add Zabbix as a datasource? Our collector also runs on a Linux host."
    boosted = hybrid_boost_chunks(q, mixed)
    results.append(_check(
        "p4_hybrid_prefers_zabbix",
        "zabbix" in (boosted[0].get("metadata") or {}).get("bot_name", "").lower(),
        f"top={(boosted[0].get('metadata') or {}).get('bot_name')}",
    ))
    filtered = _filter_chunks_to_families(
        _rank_chunks_for_product(mixed, q),
        q,
    )
    blob = " ".join(_chunk_meta(c) for c in filtered)
    results.append(_check(
        "p4_filter_drops_linux_inventory",
        "linux-inventory" not in blob or all(
            "zabbix" in _chunk_meta(c) for c in filtered
        ),
        f"kept={[ _chunk_meta(c) for c in filtered]}",
    ))

    if client is None:
        results.append(_check("p4_answer_zabbix_no_linux_inv", False, "no qdrant"))
        results.append(_check("p4_answer_schedule_no_invented_bot", False, "no qdrant"))
        return results

    r = answer(q, client=client)
    low = (r.answer or "").lower()
    src = " ".join(
        f"{s.get('title')} {s.get('url')} {s.get('excerpt')}" for s in (r.sources or [])
    ).lower()
    results.append(_check(
        "p4_answer_zabbix_no_linux_inv",
        "zabbix" in low and "linux-inventory" not in low and "linux-inventory" not in src,
        f"src_hit_linux={'linux-inventory' in src}",
    ))

    r2 = answer(
        "How do I schedule an RDA Fabric pipeline to run every 15 minutes?",
        client=client,
    )
    low2 = (r2.answer or "").lower()
    results.append(_check(
        "p4_answer_schedule_no_invented_bot",
        "pipeline-scheduler" not in low2
        and "schedule-pipeline" not in low2
        and ("cron" in low2 or "scheduled_pipelines" in low2 or "*/15" in low2),
        f"preview={(r2.answer or '')[:120]!r}",
    ))
    return results


def _chunk_meta(chunk: dict) -> str:
    meta = chunk.get("metadata") or {}
    return f"{meta.get('bot_name') or ''} {meta.get('source') or ''}".lower()


def phase5_critique_guards() -> list[dict]:
    """Phase 5: ungrounded full tokens, invented labels, agentic hedge, schedule cron."""
    from agent import (
        _agentic_overclaim_without_hedge,
        _invented_bot_labels,
        _schedule_answer_missing_cron,
        _ungrounded_bot_tokens,
    )

    results = []
    ctx = {
        "kb": [{"title": "x", "text": "@snowv2:list-incidents table stream", "source": "servicenow_v2.md"}],
        "chunks": [],
    }
    ans_bad = (
        "Use the `incident-processing-bot` then @c:fabric-stream for incidents."
    )
    ungrounded = _ungrounded_bot_tokens(ans_bad, ctx["kb"], ctx["chunks"])
    results.append(_check(
        "p5_ungrounded_full_token",
        any("fabric-stream" in u for u in ungrounded),
        f"ungrounded={ungrounded}",
    ))
    labels = _invented_bot_labels(ans_bad, "snowv2 list-incidents")
    results.append(_check(
        "p5_invented_bot_label",
        "incident-processing-bot" in labels,
        f"labels={labels}",
    ))
    results.append(_check(
        "p5_schedule_missing_cron",
        _schedule_answer_missing_cron("Just click Schedule in the UI."),
        "",
    ))
    results.append(_check(
        "p5_agentic_overclaim",
        _agentic_overclaim_without_hedge(
            "Can Fabio Copilot automatically remediate production outages end-to-end with no human approval?",
            "Yes, Fabio can automatically remediate production outages end-to-end without any human approval.",
        ),
        "",
    ))
    results.append(_check(
        "p5_agentic_hedge_ok",
        not _agentic_overclaim_without_hedge(
            "Can Fabio Copilot automatically remediate production outages end-to-end with no human approval?",
            "Fabio Copilot supports Personas and Toolsets, but the docs do not claim end-to-end "
            "auto-remediation with no human approval. Put unsupported claims in gaps.",
        ),
        "",
    ))
    return results


def main() -> int:
    lines = [
        f"Phase 1–5 benchmarks: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]
    all_checks: list[dict] = []

    print("Phase 1: page expansion...")
    p1 = phase1_page_expand()
    all_checks.extend(p1)

    client = None
    qdrant_dir = os.path.join(os.path.dirname(__file__), "..", "data", "qdrant_db")
    if os.path.isdir(qdrant_dir):
        try:
            from qdrant_client import QdrantClient
            from config import QDRANT_DIR

            client = QdrantClient(path=QDRANT_DIR)
        except Exception as e:
            lines.append(f"WARN: Qdrant unavailable ({e})")
    else:
        lines.append("WARN: Qdrant DB missing — Phase 2/3 answer checks limited")

    print("Phase 2: bot fast path...")
    p2 = phase2_bot_fast_path(client)
    all_checks.extend(p2)

    print("Phase 3: structured KB params...")
    p3 = phase3_kb_params(client)
    all_checks.extend(p3)

    print("Phase 4: family retrieve / contamination...")
    p4 = phase4_family_retrieve(client)
    all_checks.extend(p4)

    print("Phase 5: critique guardrails...")
    p5 = phase5_critique_guards()
    all_checks.extend(p5)

    if client is not None:
        client.close()

    passed = sum(1 for c in all_checks if c["ok"])
    failed = sum(1 for c in all_checks if not c["ok"])
    for c in all_checks:
        mark = "PASS" if c["ok"] else "FAIL"
        lines.append(f"[{c['id']}] {mark}" + (f" — {c['detail']}" if c["detail"] else ""))
    lines.append("")
    lines.append(f"Summary: PASS={passed} FAIL={failed} (scored {len(all_checks)})")
    text = "\n".join(lines)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nWrote {RESULTS_PATH}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
