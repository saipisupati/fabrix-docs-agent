"""
run_eval_kb.py — score KB/scope/inference/examples/empty-sources agent behavior.

Run: python3 tests/run_eval_kb.py
Writes tests/eval_kb_results.txt (gitignored).
Requires OPENAI_API_KEY, OPENROUTER_API_KEY, data/qdrant_db/, and data/kb/.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qdrant_client import QdrantClient

from agent import answer
from config import KB_DIR, QDRANT_DIR
from eval_set import EVAL_SET

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval_kb_results.txt")

KB_CATEGORIES = {"scope", "inference", "examples", "gaps"}


def _abstained(text: str) -> bool:
    t = (text or "").lower()
    return any(
        p in t
        for p in (
            "couldn't find",
            "could not find",
            "not in the documentation",
            "outside the scope",
            "out of scope",
        )
    )


def score_kb_case(case: dict, result) -> dict:
    """Return {pass: bool, notes: [str]} for KB-specific expectations."""
    notes = []
    ok = True
    answer_text = result.answer or ""
    sources = result.sources or []
    examples = result.examples or []
    gaps = result.gaps or []
    scope = result.scope or ""

    if case.get("expect_empty_sources"):
        if sources:
            ok = False
            notes.append(f"expected empty sources, got {len(sources)}")
        else:
            notes.append("sources empty (ok)")

    if case.get("expect_scope"):
        if scope != case["expect_scope"]:
            # Allow related→out_of_scope miss only if still abstained with empty sources
            if not (_abstained(answer_text) and not sources):
                ok = False
                notes.append(f"expected scope={case['expect_scope']}, got {scope}")
            else:
                notes.append(f"scope={scope} but abstained with empty sources (ok)")
        else:
            notes.append(f"scope={scope} (ok)")

    if case.get("expect_scope_in"):
        if scope not in case["expect_scope_in"]:
            ok = False
            notes.append(f"expected scope in {case['expect_scope_in']}, got {scope}")
        else:
            notes.append(f"scope={scope} (ok)")

    if case.get("expect_examples"):
        has_ex = bool(examples) or ("## examples" in answer_text.lower())
        if not has_ex:
            ok = False
            notes.append("expected examples in response")
        else:
            notes.append(f"examples present ({len(examples)}) (ok)")

    if case.get("expect_inferred"):
        used = bool(getattr(result, "used_inference", False))
        summary = (getattr(result, "inferred_summary", None) or "").strip()
        disclosure = "includes inferred fabrix guidance" in answer_text.lower()
        if not (used or disclosure):
            ok = False
            notes.append("expected used_inference or disclosure line in answer")
        elif used and not summary and not disclosure:
            ok = False
            notes.append("used_inference set but missing inferred_summary/disclosure")
        else:
            if summary and re.search(r"\[\d+\]", summary):
                ok = False
                notes.append("inferred_summary must not contain [n] citations")
            else:
                notes.append(
                    f"inference disclosed (used_inference={used}, summary_len={len(summary)}) (ok)"
                )
        # Unified answer should not use old report headings
        if re.search(r"##\s*Inferred\s*\(not in documentation\)", answer_text, re.IGNORECASE):
            ok = False
            notes.append("unified answer should not contain ## Inferred heading")

    if case.get("expect_any_substrings"):
        low = answer_text.lower()
        if not any(s.lower() in low for s in case["expect_any_substrings"]):
            ok = False
            notes.append(
                f"expected one of {case['expect_any_substrings']} in answer"
            )
        else:
            notes.append("expected topic substring present (ok)")

    if case.get("forbid_substrings"):
        blob = (answer_text + "\n" + "\n".join(examples)).lower()
        hit_bad = False
        for bad in case["forbid_substrings"]:
            if bad.lower() in blob:
                ok = False
                hit_bad = True
                notes.append(f"forbidden substring present: {bad}")
        if not hit_bad:
            notes.append("no forbidden substrings (ok)")

    if case.get("forbid_undisclosed_claims"):
        low = answer_text.lower()
        disclosure = "includes inferred fabrix guidance" in low
        used = bool(getattr(result, "used_inference", False))
        for claim in case["forbid_undisclosed_claims"]:
            if claim.lower() in low and not (disclosure or used):
                ok = False
                notes.append(f"undisclosed overclaim: {claim}")
        # Even with disclosure, prefer not stating auto-dashboard as hard fact with [n]
        for claim in case["forbid_undisclosed_claims"]:
            if claim.lower() in low and re.search(
                rf"{re.escape(claim)}.{{0,40}}\[\d+\]", answer_text, re.IGNORECASE | re.DOTALL
            ):
                ok = False
                notes.append(f"overclaim cited as documented: {claim}")

    if case.get("max_sudoers_mentions") is not None:
        count = len(re.findall(r"sudoers|NOPASSWD", answer_text, re.IGNORECASE))
        if count > case["max_sudoers_mentions"]:
            ok = False
            notes.append(f"sudoers/NOPASSWD mentioned {count} times (max {case['max_sudoers_mentions']})")
        else:
            notes.append(f"sudoers mentions={count} (ok)")

    if case.get("expect_ordered_list_renumbered"):
        nums = [int(m.group(1)) for m in re.finditer(r"(?m)^(\d+)\.\s+", answer_text)]
        if len(nums) >= 3 and all(n == 1 for n in nums):
            ok = False
            notes.append("ordered list still all 1. after cleanup")
        elif len(nums) >= 3:
            notes.append(f"ordered list numbers={nums[:8]} (ok)")
        else:
            notes.append("fewer than 3 ordered lines (skip renumber check)")

    # If answer talks about dashboard/dataset handoff, require disclosure
    if case.get("id") == "inference_04":
        low = answer_text.lower()
        handoff = any(w in low for w in ("dashboard", "dataset"))
        disclosure = "includes inferred fabrix guidance" in low
        used = bool(getattr(result, "used_inference", False))
        if handoff and not (disclosure or used):
            ok = False
            notes.append("dashboard/dataset handoff without inference disclosure")

    if case["category"] == "inference" and not case.get("expect_inferred"):
        used = bool(getattr(result, "used_inference", False))
        disclosure = "includes inferred fabrix guidance" in answer_text.lower()
        if not (_abstained(answer_text) or used or disclosure or sources):
            ok = False
            notes.append("inference case: need grounding, used_inference, or clear abstain")
        else:
            notes.append(
                f"inference path ok (used={used}, sources={len(sources)}, abstain={_abstained(answer_text)})"
            )
    if case["category"] == "gaps":
        has_gap = bool(gaps) or ("## gaps" in answer_text.lower()) or _abstained(answer_text)
        if not has_gap:
            ok = False
            notes.append("expected gaps section or abstention")
        else:
            notes.append("gaps/abstain present (ok)")
        if case.get("expect_empty_sources_if_abstain") and _abstained(answer_text) and sources:
            ok = False
            notes.append("abstained but sources not empty")

    if case["category"] == "examples" and case["id"] == "examples_02":
        if not sources:
            ok = False
            notes.append("bot lookup should have sources")
        else:
            notes.append(f"sources={len(sources)} (ok)")
        for fact in case.get("expected_facts") or []:
            if fact.lower() not in answer_text.lower():
                ok = False
                notes.append(f"missing fact: {fact}")

    if case["category"] == "scope" and not case.get("expect_empty_sources"):
        if _abstained(answer_text) and sources:
            ok = False
            notes.append("abstained with non-empty sources")

    return {"pass": ok, "notes": notes, "scope": scope, "n_sources": len(sources),
            "n_examples": len(examples), "n_gaps": len(gaps)}


def main():
    missing = [k for k in ("OPENROUTER_API_KEY", "OPENAI_API_KEY") if k not in os.environ]
    if missing:
        print(f"SKIP: missing env vars: {', '.join(missing)}")
        sys.exit(0)

    if not os.path.isdir(QDRANT_DIR):
        print(f"SKIP: Qdrant DB not found at {QDRANT_DIR}")
        sys.exit(0)

    kb_json = os.path.join(KB_DIR, "kb.json")
    if not os.path.isfile(kb_json):
        print(f"SKIP: KB not found at {kb_json}: run python3 src/build_kb.py")
        sys.exit(0)

    cases = [c for c in EVAL_SET if c["category"] in KB_CATEGORIES]
    if not cases:
        print("SKIP: no KB category cases in EVAL_SET")
        sys.exit(0)

    client = QdrantClient(path=QDRANT_DIR)
    report = [
        f"KB/scope eval: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"cases={len(cases)}",
        "",
    ]
    passed = failed = 0

    for case in cases:
        result = answer(case["question"], client=client)
        score = score_kb_case(case, result)
        grade = "PASS" if score["pass"] else "FAIL"
        if score["pass"]:
            passed += 1
        else:
            failed += 1
        report.append(f"[{case['id']}] ({case['category']}) {case['question']}")
        report.append(
            f"  scope={score['scope']} sources={score['n_sources']} "
            f"examples={score['n_examples']} gaps={score['n_gaps']}"
        )
        preview = (result.answer or "").replace("\n", " ")[:180]
        report.append(f"  Answer: {preview}...")
        for n in score["notes"]:
            report.append(f"  - {n}")
        report.append(f"  Grade: {grade}")
        report.append("")

    report.append(f"Summary: PASS={passed} FAIL={failed} (scored {len(cases)}/{len(cases)})")
    text = "\n".join(report)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
