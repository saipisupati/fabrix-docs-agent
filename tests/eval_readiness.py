"""
eval_readiness.py — production-readiness gate: correctness + latency.

Run: python3 tests/eval_readiness.py
Writes tests/eval_readiness_results.txt (gitignored).
Stop the API first (local Qdrant file lock).

Gate (local bar, not ship):
- PASS rate >= 95% on the readiness subset
- p95 wall time <= READINESS_P95_MS (default 45000)
Do not embed on the docs site until this gate is green twice in a row.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from qdrant_client import QdrantClient

from agent import answer
from config import KB_DIR, QDRANT_DIR
from eval_production import PRODUCTION_CASES, score_case

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval_readiness_results.txt")

# Fixed subset spanning lookup, synthesis, trap, break locks
READINESS_IDS = [
    "lookup_bot",
    "lookup_cfxql",
    "synth_k8s",
    "synth_snow",
    "vague_dash",
    "oos_billing",
    "break_format_pstream",
    "break_slang_sn",
    "break_trap_sla",
    "break_day2_pstream",
    "break_prometheus_ubuntu",
    "c2_format_cfxql",
    "c2_trap_quote",
    "c2_wire_datadog",
]

PASS_RATE_MIN = 0.95
P95_MS_DEFAULT = 45000


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def main():
    missing = [k for k in ("OPENROUTER_API_KEY", "OPENAI_API_KEY") if k not in os.environ]
    if missing:
        print(f"SKIP: missing env vars: {', '.join(missing)}")
        sys.exit(0)
    if not os.path.isdir(QDRANT_DIR):
        print(f"SKIP: Qdrant DB not found at {QDRANT_DIR}")
        sys.exit(0)
    if not os.path.isfile(os.path.join(KB_DIR, "kb.json")):
        print("SKIP: KB not found; run python3 src/build_kb.py")
        sys.exit(0)

    by_id = {c["id"]: c for c in PRODUCTION_CASES}
    cases = [by_id[i] for i in READINESS_IDS if i in by_id]
    if len(cases) < len(READINESS_IDS):
        missing_ids = [i for i in READINESS_IDS if i not in by_id]
        print(f"WARN: missing cases {missing_ids}")

    p95_budget = float(os.environ.get("READINESS_P95_MS", P95_MS_DEFAULT))
    client = QdrantClient(path=QDRANT_DIR)

    report = [
        f"Readiness gate: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"cases={len(cases)} pass_rate_min={PASS_RATE_MIN} p95_budget_ms={p95_budget}",
        "",
    ]
    grades = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
    totals: list[float] = []
    llm_calls: list[int] = []

    for case in cases:
        t0 = time.perf_counter()
        result = answer(case["question"], client=client)
        wall_ms = round((time.perf_counter() - t0) * 1000, 1)
        timing = result.timing or {}
        total_ms = float(timing.get("total_ms") or wall_ms)
        totals.append(total_ms)
        llm_calls.append(int(timing.get("llm_calls") or 0))
        score = score_case(case, result)
        grades[score["grade"]] += 1
        report.append(
            f"[{case['id']}] {score['grade']} total_ms={total_ms} "
            f"llm_calls={timing.get('llm_calls')} "
            f"scope={timing.get('scope_ms')} retrieve={timing.get('retrieve_ms')} "
            f"generate={timing.get('generate_ms')} critique={timing.get('critique_ms')}"
        )
        report.append(f"  Q: {case['question']}")
        report.append(f"  A: {(result.answer or '').replace(chr(10), ' ')[:160]}")
        for n in score["notes"]:
            report.append(f"  - {n}")
        report.append("")

    n = len(cases) or 1
    pass_rate = grades["PASS"] / n
    sorted_totals = sorted(totals)
    p50 = round(_percentile(sorted_totals, 0.50), 1)
    p95 = round(_percentile(sorted_totals, 0.95), 1)
    avg_llm = round(sum(llm_calls) / n, 2) if llm_calls else 0

    pass_ok = pass_rate >= PASS_RATE_MIN
    latency_ok = p95 <= p95_budget
    gate = "GREEN" if pass_ok and latency_ok else "RED"

    report.append(
        f"Summary: PASS={grades['PASS']} PARTIAL={grades['PARTIAL']} FAIL={grades['FAIL']} "
        f"pass_rate={pass_rate:.0%} p50_ms={p50} p95_ms={p95} avg_llm_calls={avg_llm}"
    )
    report.append(
        f"Gate: {gate} (pass_rate>={PASS_RATE_MIN:.0%} → {pass_ok}; "
        f"p95<={p95_budget:.0f}ms → {latency_ok})"
    )
    report.append(
        "Policy: do not embed on docs.fabrix.ai until this gate is GREEN twice in a row."
    )

    text = "\n".join(report)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nWrote {RESULTS_PATH}")
    client.close()
    sys.exit(0 if gate == "GREEN" else 1)


if __name__ == "__main__":
    main()
