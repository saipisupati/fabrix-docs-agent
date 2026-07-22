"""
eval_customer_bakeoff.py — public/customer-facing gate for the 8 real bakeoff questions.

Treats the agent as production: soft asserts for accuracy, honesty, contamination,
and no invented bots. Run via live /ask (API up) OR local answer() (API stopped).

  # Prefer live API (customer path):
  python3 tests/eval_customer_bakeoff.py

  # Local (stop API first — Qdrant lock):
  BAKEOFF_MODE=local PYTHONPATH=src python3 tests/eval_customer_bakeoff.py

Exit 0 only when all cases PASS.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval_customer_bakeoff_results.txt")
API_URL = os.environ.get("BAKEOFF_API", "http://127.0.0.1:8080/ask")
MODE = os.environ.get("BAKEOFF_MODE", "api")  # api | local

CASES = [
    {
        "id": "sn_ticketing_stream",
        "question": (
            "How do I get ServiceNow incidents into a Fabrix stream "
            "so downstream bots can act on them?"
        ),
        "need_any": ["servicenow", "snow", "incident", "stream"],
        "forbid": ["linux-inventory", "incident-processing-bot", "@c:fabric-stream", "@c:service-now-integration"],
        "prefer_bot_token": True,
        "need_bot_family_any": ["snow", "servicenow"],
    },
    {
        "id": "cfxql_full_vs_restricted",
        "question": "What is the difference between Full CFXQL and Restricted CFXQL?",
        "need_any": ["full", "restricted"],
    },
    {
        "id": "pstream_vs_dataset",
        "question": (
            "When should an engineer use a persistent stream instead of a regular "
            "dataset in RDA Fabric?"
        ),
        "need_any": ["pstream", "persistent stream", "dataset"],
    },
    {
        "id": "zabbix_linux",
        "question": (
            "How do I add Zabbix as a datasource? Our collector also runs on a Linux "
            "host if that matters."
        ),
        "need_any": ["zabbix"],
        "forbid": ["linux-inventory"],
    },
    {
        "id": "kafka_consume_params",
        "question": "What parameters does the kafka-v2 consume / read bot take?",
        "need_any": ["name", "group", "offset_reset", "batch_size"],
        "need_all_soft": ["read-stream"],
        "expect_zero_llm": True,
        "max_latency_s": 3.0,
    },
    {
        "id": "fabio_auto_remediate",
        "question": (
            "Can Fabio Copilot automatically remediate production outages end-to-end "
            "with no human approval?"
        ),
        "need_any": ["fabio", "copilot", "agentic", "persona", "toolset"],
        "require_hedge": True,
    },
    {
        "id": "pipeline_15m",
        "question": "How do I schedule an RDA Fabric pipeline to run every 15 minutes?",
        "need_any": ["cron", "scheduled_pipelines", "*/15", "15"],
        "forbid": ["pipeline-scheduler", "schedule-pipeline"],
    },
    {
        "id": "p1_sla",
        "question": "What is Fabrix's contractual P1 support SLA response time?",
        "need_any": ["couldn't find", "outside", "not in", "out of scope"],
        "expect_abstain": True,
    },
]


def _ask_api(question: str) -> dict:
    req = urllib.request.Request(
        API_URL,
        data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def _ask_local(question: str, client):
    from agent import answer

    r = answer(question, client=client)
    return {
        "answer": r.answer,
        "sources": r.sources,
        "examples": r.examples,
        "gaps": r.gaps,
        "scope": r.scope,
        "used_inference": r.used_inference,
        "timing": r.timing,
    }


def _score(case: dict, data: dict, latency_s: float) -> tuple[str, list[str]]:
    notes: list[str] = []
    ok = True
    ans = data.get("answer") or ""
    low = ans.lower()
    src = " ".join(
        f"{s.get('title') or ''} {s.get('url') or ''} {s.get('excerpt') or ''}"
        for s in (data.get("sources") or [])
    ).lower()
    ex = "\n".join(data.get("examples") or []).lower()
    blob = low + "\n" + src + "\n" + ex
    llm = int((data.get("timing") or {}).get("llm_calls") or 0)

    need = case.get("need_any") or []
    if need and not any(n.lower() in low for n in need):
        ok = False
        notes.append(f"missing need_any={need}")

    for soft in case.get("need_all_soft") or []:
        if soft.lower() not in low:
            notes.append(f"soft miss: {soft}")

    for bad in case.get("forbid") or []:
        if bad.lower() in blob:
            ok = False
            notes.append(f"forbid hit: {bad}")

    if case.get("expect_zero_llm") and llm != 0:
        ok = False
        notes.append(f"expected llm=0 got {llm}")

    max_lat = case.get("max_latency_s")
    if max_lat is not None and latency_s > max_lat:
        ok = False
        notes.append(f"latency {latency_s}s > {max_lat}s")

    if case.get("expect_abstain"):
        if not any(
            x in low
            for x in ("couldn't find", "could not find", "outside", "out of scope", "not in")
        ):
            ok = False
            notes.append("expected abstain")

    if case.get("require_hedge"):
        hedge = any(
            h in low
            for h in (
                "does not", "do not", "don't", "not explicitly", "not documented",
                "gaps", "approval", "oversight", "unsupported", "cannot", "not claim",
                "however", "important to note",
            )
        )
        overclaim = any(
            p in low
            for p in (
                "without any human", "no human approval required",
                "fully automatic end-to-end", "yes, fabio can automatically",
            )
        )
        if overclaim and not hedge:
            ok = False
            notes.append("agentic overclaim without hedge")
        elif not hedge and not any(g for g in (data.get("gaps") or [])):
            # soft: prefer hedge or gaps
            notes.append("weak honesty signal (no clear hedge/gaps)")

    if case.get("prefer_bot_token"):
        if not re.search(r"[@*][a-z0-9_-]+\s*:", low):
            # Prefer concrete tokens; fail if we also invented a *-bot label
            if re.search(r"[a-z0-9_-]+-bot", low):
                ok = False
                notes.append("invented *-bot label without concrete @family:op token")
            else:
                notes.append("no concrete @bot:token (prefer for customer wiring)")

    fam_need = case.get("need_bot_family_any") or []
    if fam_need:
        tokens = re.findall(r"[@*]([a-z0-9_-]+)\s*:", low)
        if not any(any(f in t for f in fam_need) for t in tokens):
            ok = False
            notes.append(f"missing bot family token in {fam_need}")

    if "```json" in low:
        ok = False
        notes.append("leaked json fence")

    grade = "PASS" if ok else "FAIL"
    return grade, notes


def main() -> int:
    lines = [
        f"Customer bakeoff (public gate): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"mode={MODE} cases={len(CASES)}",
        "",
    ]
    client = None
    if MODE == "local":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from qdrant_client import QdrantClient
        from config import QDRANT_DIR

        client = QdrantClient(path=QDRANT_DIR)

    grades = {"PASS": 0, "FAIL": 0}
    for case in CASES:
        t0 = time.time()
        try:
            if MODE == "local":
                data = _ask_local(case["question"], client)
            else:
                data = _ask_api(case["question"])
        except Exception as e:
            grades["FAIL"] += 1
            lines.append(f"[{case['id']}] FAIL — request error: {e}")
            lines.append("")
            continue
        latency = round(time.time() - t0, 1)
        grade, notes = _score(case, data, latency)
        grades[grade] += 1
        llm = (data.get("timing") or {}).get("llm_calls")
        lines.append(
            f"[{case['id']}] {grade} {latency}s llm={llm} scope={data.get('scope')}"
        )
        lines.append(f"  Q: {case['question']}")
        preview = (data.get("answer") or "").replace("\n", " ")[:200]
        lines.append(f"  A: {preview}")
        for n in notes:
            lines.append(f"  - {n}")
        lines.append("")

    if client is not None:
        client.close()

    lines.append(f"Summary: PASS={grades['PASS']} FAIL={grades['FAIL']} (scored {len(CASES)})")
    text = "\n".join(lines)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nWrote {RESULTS_PATH}")
    return 0 if grades["FAIL"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
