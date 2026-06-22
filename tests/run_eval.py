"""
run_eval.py — runs every question in eval_set.py through the real pipeline
(query_qdrant.ask) and prints the result next to the expected facts, so you
can grade each one by eye (pass/fail/partial) and track accuracy over time.

Usage:
    python tests/run_eval.py            # run all cases
    python tests/run_eval.py lookup_01  # run a single case by id
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval_set import EVAL_SET, retrieval_params
from query_qdrant import ask


def run_case(case):
    print("\n" + "=" * 70)
    print(f"[{case['id']}] ({case['category']}) {case['question']}")
    print("=" * 70)
    print(f"Expected source: {case['expected_source']}")
    print("Expected facts:")
    for fact in case["expected_facts"]:
        print(f"  - {fact}")

    print("\n--- ACTUAL PIPELINE OUTPUT ---")
    ask(case["question"], category=case["category"], **retrieval_params(case))

    print("--- GRADE THIS ONE ---")
    print("  PASS  = all expected facts present, no contradictions/hallucination")
    print("  PARTIAL = some facts present, some missing or slightly wrong")
    print("  FAIL  = wrong, contradictory, or hallucinated")


def main():
    if len(sys.argv) > 1:
        case_id = sys.argv[1]
        matches = [c for c in EVAL_SET if c["id"] == case_id]
        if not matches:
            print(f"No case found with id '{case_id}'")
            print(f"Available ids: {[c['id'] for c in EVAL_SET]}")
            return
        run_case(matches[0])
    else:
        print(f"Running all {len(EVAL_SET)} eval cases...\n")
        for case in EVAL_SET:
            run_case(case)
            input("\nPress Enter to continue to next case...")


if __name__ == "__main__":
    main()

