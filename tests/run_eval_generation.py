"""
run_eval_generation.py: full-pipeline generation eval against eval_set.py.

Runs retrieve + generate for each case and scores the answer text.

Usage:
    python tests/run_eval_generation.py

Requires: OPENROUTER_API_KEY, OPENAI_API_KEY, existing data/qdrant_db/
Writes: tests/eval_generation_results.txt
"""

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qdrant_client import QdrantClient

from config import QDRANT_DIR
from eval_scoring import grade_generation, score_generation
from eval_set import EVAL_SET, FILTER_BY_CATEGORY, TOP_K_BY_CATEGORY, retrieval_params
from query_qdrant import run_pipeline

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval_generation_results.txt")


def format_case_report(case, answer, score):
    lines = [
        f"[{case['id']}] ({case['category']}) {case['question']}",
        f"  Answer preview: {answer[:200].replace(chr(10), ' ')}...",
    ]

    if case["category"] == "negative":
        lines.append(f"  Abstained: {score.get('abstained')}")
        lines.append(f"  Hallucinated: {score.get('hallucinated')}")
    else:
        lines.append(
            f"  Fact coverage: {score['fact_score']:.0%} "
            f"({len(score['facts_found'])}/{len(case['expected_facts'])})"
        )
        if score.get("contradiction"):
            lines.append(f"  Contradiction: {score['contradiction']}")

    if score["facts_missing"]:
        lines.append("  Missing facts:")
        for fact in score["facts_missing"]:
            lines.append(f"    - {fact}")

    lines.append(f"  Grade: {grade_generation(score, case)}")
    return lines


def main():
    missing = [k for k in ("OPENROUTER_API_KEY", "OPENAI_API_KEY") if k not in os.environ]
    if missing:
        print(f"SKIP: missing env vars: {', '.join(missing)}")
        sys.exit(0)

    if not os.path.isdir(QDRANT_DIR):
        print(f"SKIP: Qdrant DB not found at {QDRANT_DIR}: run ingest first")
        sys.exit(0)

    client = QdrantClient(path=QDRANT_DIR)
    report_lines = [
        f"Generation eval: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"filters: {FILTER_BY_CATEGORY}, top_k: {TOP_K_BY_CATEGORY}, cases={len(EVAL_SET)}",
        "",
    ]

    grades = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
    for case in EVAL_SET:
        params = retrieval_params(case)
        category = case["category"]
        _, answer = run_pipeline(
            case["question"],
            client,
            top_k=params["top_k"],
            filter_dict=params["filter_dict"],
            category=category,
        )
        score = score_generation(case, answer)
        grades[grade_generation(score, case)] += 1
        report_lines.extend(format_case_report(case, answer, score))
        report_lines.append("")

    report_lines.append(
        f"Summary: PASS={grades['PASS']} PARTIAL={grades['PARTIAL']} "
        f"FAIL={grades['FAIL']} (scored {len(EVAL_SET)}/{len(EVAL_SET)})"
    )

    report = "\n".join(report_lines)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(report)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
