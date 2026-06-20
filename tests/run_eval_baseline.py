"""
run_eval_baseline.py — retrieval-only baseline against eval_set.py.

Scores whether query_qdrant.retrieve returns chunks containing the
expected source and expected facts. No LLM generation step.

Usage:
    python tests/run_eval_baseline.py

Requires: OPENROUTER_API_KEY, existing data/qdrant_db/ from ingest
Writes: tests/eval_baseline_results.txt
"""

import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qdrant_client import QdrantClient

from config import QDRANT_DIR
from eval_set import EVAL_SET
from query_qdrant import retrieve

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval_baseline_results.txt")
TOP_K = 5


def load_dotenv():
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def chunk_ref(chunk):
    meta = chunk["metadata"]
    if meta.get("type") == "bot":
        return meta.get("bot_name", "?")
    return meta.get("source", "?")


def expected_source_hit(case, chunks):
    expected = case["expected_source"].lower()
    if expected.startswith("none"):
        return None

    refs = [chunk_ref(c).lower() for c in chunks]

    if expected.startswith("@") or expected.startswith("*"):
        return any(expected in ref or ref in expected for ref in refs)

    if "cfxql" in expected:
        has_ref = any("cfxql" in ref for ref in refs)
        if "full" in expected and "restricted" in expected:
            types = [c["metadata"].get("cfxql_type", "").lower() for c in chunks]
            return has_ref and "full" in types and "restricted" in types
        return has_ref

    return any(expected in ref for ref in refs)


def fact_tokens(fact):
    stop = {
        "should", "state", "could", "find", "this", "that", "not", "the", "and",
        "does", "must", "are", "for", "with", "from", "into", "only", "all", "has",
        "have", "what", "how", "explicitly", "invent", "plausible", "sounding",
        "fabricated", "specific", "number",
    }
    return [
        w for w in re.findall(r"[a-z0-9]+", fact.lower())
        if len(w) > 2 and w not in stop
    ]


def fact_hit(fact, text):
    tokens = fact_tokens(fact)
    if not tokens:
        return True
    hits = sum(1 for token in tokens if token in text)
    return hits >= max(1, int(len(tokens) * 0.5))


def score_retrieval(case, chunks):
    if case["category"] == "negative":
        return {
            "source_hit": None,
            "facts_found": [],
            "facts_missing": [],
            "fact_score": None,
            "note": "Negative cases are graded on generation, not retrieval",
        }

    text = " ".join(c["text"] for c in chunks).lower()
    found = [fact for fact in case["expected_facts"] if fact_hit(fact, text)]
    missing = [fact for fact in case["expected_facts"] if fact not in found]
    total = len(case["expected_facts"])

    return {
        "source_hit": expected_source_hit(case, chunks),
        "facts_found": found,
        "facts_missing": missing,
        "fact_score": len(found) / total if total else 1.0,
    }


def grade_case(score):
    if score.get("note"):
        return "SKIP"
    source_ok = score["source_hit"]
    fact_score = score["fact_score"]
    if source_ok and fact_score >= 0.8:
        return "PASS"
    if source_ok or fact_score >= 0.5:
        return "PARTIAL"
    return "FAIL"


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
    lines.append(f"  Grade: {grade_case(score)}")
    return lines


def main():
    load_dotenv()

    if "OPENROUTER_API_KEY" not in os.environ:
        print("SKIP: OPENROUTER_API_KEY not set")
        sys.exit(0)

    if not os.path.isdir(QDRANT_DIR):
        print(f"SKIP: Qdrant DB not found at {QDRANT_DIR} — run ingest first")
        sys.exit(0)

    client = QdrantClient(path=QDRANT_DIR)
    report_lines = [
        f"Retrieval baseline — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"top_k={TOP_K}, cases={len(EVAL_SET)}",
        "",
    ]

    grades = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "SKIP": 0}
    for case in EVAL_SET:
        chunks = retrieve(case["question"], client, top_k=TOP_K)
        score = score_retrieval(case, chunks)
        grades[grade_case(score)] += 1
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
