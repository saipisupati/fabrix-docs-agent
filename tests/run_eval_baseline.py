"""
run_eval_baseline.py, retrieval-only eval (no LLM generation).

Run: python3 tests/run_eval_baseline.py
Checks if the right chunks show up in top-k before we blame the model.
"""

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qdrant_client import QdrantClient

from config import QDRANT_DIR
from eval_scoring import chunk_ref, grade_retrieval, score_retrieval
from eval_set import EVAL_SET, FILTER_BY_CATEGORY, TOP_K_BY_CATEGORY, retrieval_params
from query_qdrant import retrieve

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval_baseline_results.txt")


def format_case_report(case, chunks, score):
    lines = [
        f"[{case['id']}] ({case['category']}) {case['question']}",
        f"  Expected source: {case['expected_source']}",
        f"  Retrieved ({len(chunks)} chunks):",
    ]
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"    [{i}] score={chunk['score']:.3f}  {chunk_ref(chunk)}")

    if score.get("note"):
        lines.append(f"  {score['note']}")
        lines.append("  Grade: SKIP (retrieval-only)")
        return lines

    lines.append(f"  Source hit: {score['source_hit']}")
    lines.append(
        f"  Fact coverage: {score['fact_score']:.0%} "
        f"({len(score['facts_found'])}/{len(case['expected_facts'])})"
    )
    if score["facts_missing"]:
        lines.append("  Missing facts:")
        for fact in score["facts_missing"]:
            lines.append(f"    - {fact}")
    lines.append(f"  Grade: {grade_retrieval(score)}")
    return lines


def main():
    if "OPENROUTER_API_KEY" not in os.environ:
        print("SKIP: OPENROUTER_API_KEY not set")
        sys.exit(0)

    if not os.path.isdir(QDRANT_DIR):
        print(f"SKIP: Qdrant DB not found at {QDRANT_DIR}: run ingest first")
        sys.exit(0)

    client = QdrantClient(path=QDRANT_DIR)
    report_lines = [
        f"Retrieval baseline: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"filters: {FILTER_BY_CATEGORY}, top_k: {TOP_K_BY_CATEGORY}, cases={len(EVAL_SET)}",
        "",
    ]

    grades = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "SKIP": 0}
    for case in EVAL_SET:
        params = retrieval_params(case)
        chunks = retrieve(case["question"], client, **params)
        score = score_retrieval(case, chunks)
        grades[grade_retrieval(score)] += 1
        report_lines.extend(format_case_report(case, chunks, score))
        report_lines.append("")

    scored = grades["PASS"] + grades["PARTIAL"] + grades["FAIL"]
    if scored:
        report_lines.append(
            f"Summary: PASS={grades['PASS']} PARTIAL={grades['PARTIAL']} "
            f"FAIL={grades['FAIL']} SKIP={grades['SKIP']} "
            f"(scored {scored}/{len(EVAL_SET)})"
        )
    else:
        report_lines.append(f"Summary: all {grades['SKIP']} cases skipped")

    report = "\n".join(report_lines)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(report)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
