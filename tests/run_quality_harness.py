"""
run_quality_harness.py: step-by-step quality gate for the Fabrix docs agent

Runs production → full break → readiness → Phase 1–5 benchmarks → customer
bakeoff (local), writes a failure digest, and tracks readiness GREEN streak.
Exit 0 only when all raised bars pass and streak >= 2 (local) or readiness
GREEN once (HARNESS_CI_MODE=1).

Stop the API first (local Qdrant file lock)

  python3 tests/run_quality_harness.py

CI (GitHub Actions): set HARNESS_CI_MODE=1 for single-readiness GREEN gate;
HARNESS_SKIP_IF_NO_DATA=1 skips gracefully when secrets or runtime data are absent.

Outputs (gitignored):
  tests/quality_harness_digest.md
  tests/readiness_streak.json
  tests/benchmark_phases_results.txt
  tests/eval_customer_bakeoff_results.txt
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)

DIGEST_PATH = os.path.join(TESTS_DIR, "quality_harness_digest.md")
STREAK_PATH = os.path.join(TESTS_DIR, "readiness_streak.json")

PROD_RESULTS = os.path.join(TESTS_DIR, "eval_production_results.txt")
BREAK_RESULTS = os.path.join(TESTS_DIR, "eval_break_results.txt")
READY_RESULTS = os.path.join(TESTS_DIR, "eval_readiness_results.txt")
BENCH_RESULTS = os.path.join(TESTS_DIR, "benchmark_phases_results.txt")
BAKEOFF_RESULTS = os.path.join(TESTS_DIR, "eval_customer_bakeoff_results.txt")

STREAK_TARGET = 2  # local pre-deploy; CI uses single GREEN (see _streak_target)
SRC_PYTHONPATH = os.path.join(ROOT, "src")


def _ci_mode() -> bool:
    return os.environ.get("HARNESS_CI_MODE", "").strip().lower() in ("1", "true", "yes")


def _skip_if_no_data() -> bool:
    return os.environ.get("HARNESS_SKIP_IF_NO_DATA", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _has_runtime_data() -> bool:
    qdrant = os.path.join(ROOT, "data", "qdrant_db")
    kb = os.path.join(ROOT, "data", "kb", "kb.json")
    return os.path.isdir(qdrant) and os.path.isfile(kb)


def _has_api_keys() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY")) and bool(os.environ.get("OPENAI_API_KEY"))


def _preflight() -> tuple[bool, str]:
    missing: list[str] = []
    if not _has_api_keys():
        missing.append("OPENROUTER_API_KEY and/or OPENAI_API_KEY")
    if not _has_runtime_data():
        missing.append("data/qdrant_db/ and/or data/kb/kb.json")
    if not missing:
        return True, ""
    return False, "; ".join(missing)


def _streak_target() -> int:
    return 1 if _ci_mode() else STREAK_TARGET


TAG_TO_FIX = {
    "contamination": "family fidelity / INTEGRATION_FAMILIES / source filter",
    "overclaim": "honesty prompt / critique / Next (inferred) + gaps",
    "wrong_facet": "retrieve seeds / facet plan / path-first prompt",
    "abstain_fail": "scope latch / topic seeds / anti-abstain",
    "empty_sources_leak": "force OOS traps / empty sources on abstain",
    "thin_wiring": "wiring shape prompt / critique sink check",
    "format": "list polish / format-noise strip / anti-abstain",
}


def _run_suite(script: str, env: dict | None = None) -> int:
    cmd = [sys.executable, os.path.join(TESTS_DIR, script)]
    merged = os.environ.copy()
    if env:
        merged.update(env)
    # Full break battery: do not pass BREAK_CYCLE
    merged.pop("BREAK_CYCLE", None)
    print(f"\n========== {script} ==========\n", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, env=merged)
    return proc.returncode


def _parse_summary(text: str) -> dict:
    m = re.search(
        r"Summary:\s*PASS=(\d+)\s+PARTIAL=(\d+)\s+FAIL=(\d+)\s+\(scored\s+(\d+)\)",
        text,
    )
    if m:
        return {
            "PASS": int(m.group(1)),
            "PARTIAL": int(m.group(2)),
            "FAIL": int(m.group(3)),
            "scored": int(m.group(4)),
        }
    # readiness-style summary
    m2 = re.search(
        r"Summary:\s*PASS=(\d+)\s+PARTIAL=(\d+)\s+FAIL=(\d+).*?pass_rate=([\d.]+)%",
        text,
    )
    if m2:
        return {
            "PASS": int(m2.group(1)),
            "PARTIAL": int(m2.group(2)),
            "FAIL": int(m2.group(3)),
            "scored": int(m2.group(1)) + int(m2.group(2)) + int(m2.group(3)),
            "pass_rate": float(m2.group(4)) / 100.0,
        }
    # benchmark / bakeoff: PASS + FAIL only
    m3 = re.search(
        r"Summary:\s*PASS=(\d+)\s+FAIL=(\d+)\s+\(scored\s+(\d+)\)",
        text,
    )
    if m3:
        return {
            "PASS": int(m3.group(1)),
            "PARTIAL": 0,
            "FAIL": int(m3.group(2)),
            "scored": int(m3.group(3)),
        }
    return {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "scored": 0}


def _parse_case_blocks(text: str) -> list[dict]:
    """Extract FAIL/PARTIAL case blocks from eval result files."""
    cases: list[dict] = []
    # [id] (attack)? GRADE ...
    pattern = re.compile(
        r"^\[([^\]]+)\](?:\s+\(([^)]+)\))?\s+(PASS|PARTIAL|FAIL)\b(.*)$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    for i, m in enumerate(matches):
        grade = m.group(3)
        if grade == "PASS":
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        tags = []
        tm = re.search(r"tags=(\[[^\]]*\])", m.group(0) + (m.group(4) or ""))
        if tm:
            try:
                tags = ast_literal_list(tm.group(1))
            except Exception:
                tags = []
        preview = ""
        for ln in block.splitlines():
            if ln.strip().startswith("A:"):
                preview = ln.strip()[2:].strip()[:180]
                break
        notes = [
            ln.strip()[2:].strip()
            for ln in block.splitlines()
            if ln.strip().startswith("- ")
        ]
        cases.append(
            {
                "id": m.group(1),
                "attack": (m.group(2) or "").strip(),
                "grade": grade,
                "tags": tags,
                "preview": preview,
                "notes": notes,
            }
        )
    return cases


def ast_literal_list(s: str) -> list:
    import ast

    v = ast.literal_eval(s)
    return list(v) if isinstance(v, (list, tuple)) else []


def _suggest_fix(tags: list[str], attack: str) -> str:
    for t in tags:
        if t in TAG_TO_FIX:
            return TAG_TO_FIX[t]
    if attack in ("contamination", "overclaim", "trap_abstain", "format_stress", "thin_wiring"):
        return TAG_TO_FIX.get(
            {
                "contamination": "contamination",
                "overclaim": "overclaim",
                "trap_abstain": "abstain_fail",
                "format_stress": "format",
                "thin_wiring": "thin_wiring",
            }.get(attack, ""),
            "review retrieve + prompt (generic rule, not per-question)",
        )
    return "review retrieve + prompt (generic rule, not per-question)"


def _load_streak() -> dict:
    if not os.path.isfile(STREAK_PATH):
        return {"streak": 0, "last_gate": None, "history": []}
    try:
        with open(STREAK_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"streak": 0, "last_gate": None, "history": []}


def _save_streak(data: dict) -> None:
    with open(STREAK_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _readiness_green(text: str, exit_code: int) -> bool:
    if exit_code != 0:
        return False
    return "Gate: GREEN" in text


def _port_8080_listening() -> bool:
    try:
        import socket

        with socket.create_connection(("127.0.0.1", 8080), timeout=0.5):
            return True
    except OSError:
        return False


def main() -> int:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ci = _ci_mode()
    streak_target = _streak_target()
    digest: list[str] = [
        f"# Quality harness digest — {ts}",
        "",
        "## Raised bar",
        "- production: 100% PASS (0 FAIL, 0 PARTIAL)",
        "- break (full cycle1+2): ≥95% PASS and 0 FAIL",
        (
            "- readiness: single GREEN (CI mode)"
            if ci
            else f"- readiness: GREEN twice in a row (streak ≥ {STREAK_TARGET})"
        ),
        "- benchmark_phases (Phase 1–5): exit 0 / 0 FAIL",
        "- customer bakeoff (local): exit 0 / 0 FAIL",
        "",
        "Hard rule: fix agent with generic families/rules — never question-specific branches.",
        "",
    ]

    ok, reason = _preflight()
    if not ok:
        if _skip_if_no_data():
            print(f"HARNESS SKIP: {reason}", flush=True)
            digest.append("## Verdict")
            digest.append("")
            digest.append(f"**SKIP** — {reason}. Wire repo secrets + runtime tarball for full CI gate.")
            digest.append("")
            with open(DIGEST_PATH, "w", encoding="utf-8") as f:
                f.write("\n".join(digest) + "\n")
            return 0
        print(f"HARNESS FAIL preflight: {reason}", flush=True)
        digest.append("## Verdict")
        digest.append("")
        digest.append(f"**FAIL** — preflight: {reason}")
        digest.append("")
        with open(DIGEST_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(digest) + "\n")
        return 1

    if _port_8080_listening():
        digest.append(
            "**WARNING:** something is listening on :8080 — stop the API before harness "
            "(Qdrant file lock). Continuing anyway; expect lock failures if API holds Qdrant."
        )
        digest.append("")
        print(
            "WARNING: port 8080 is in use. Stop the API first (Qdrant file lock).",
            flush=True,
        )

    # --- Run suites (full break: no BREAK_CYCLE) ---
    rc_prod = _run_suite("eval_production.py")
    rc_break = _run_suite("eval_break.py")
    rc_ready = _run_suite("eval_readiness.py")
    rc_bench = _run_suite(
        "benchmark_phases.py",
        env={"PYTHONPATH": SRC_PYTHONPATH},
    )
    rc_bake = _run_suite(
        "eval_customer_bakeoff.py",
        env={
            "PYTHONPATH": SRC_PYTHONPATH,
            "BAKEOFF_MODE": "local",
        },
    )

    prod_text = _read_file(PROD_RESULTS)
    break_text = _read_file(BREAK_RESULTS)
    ready_text = _read_file(READY_RESULTS)
    bench_text = _read_file(BENCH_RESULTS)
    bake_text = _read_file(BAKEOFF_RESULTS)

    prod_sum = _parse_summary(prod_text)
    break_sum = _parse_summary(break_text)
    ready_sum = _parse_summary(ready_text)
    bench_sum = _parse_summary(bench_text)
    bake_sum = _parse_summary(bake_text)

    prod_ok = rc_prod == 0 and prod_sum.get("FAIL", 1) == 0 and prod_sum.get("PARTIAL", 1) == 0
    break_n = break_sum.get("scored") or 1
    break_pass_rate = break_sum.get("PASS", 0) / break_n
    break_ok = (
        rc_break == 0
        and break_sum.get("FAIL", 1) == 0
        and break_pass_rate >= 0.95
    )
    ready_green = _readiness_green(ready_text, rc_ready)
    bench_ok = rc_bench == 0 and bench_sum.get("FAIL", 1) == 0
    bake_ok = rc_bake == 0 and bake_sum.get("FAIL", 1) == 0

    # Streak (local pre-deploy tracks consecutive GREEN; CI resets each run)
    if ci:
        streak_data = {"streak": 0, "last_gate": None, "history": []}
        if ready_green:
            streak_data["streak"] = 1
            streak_data["last_gate"] = "GREEN"
        else:
            streak_data["last_gate"] = "RED"
    else:
        streak_data = _load_streak()
        if ready_green:
            streak_data["streak"] = int(streak_data.get("streak") or 0) + 1
            streak_data["last_gate"] = "GREEN"
        else:
            streak_data["streak"] = 0
            streak_data["last_gate"] = "RED"
        streak_data.setdefault("history", [])
        streak_data["history"].append(
            {"ts": ts, "gate": streak_data["last_gate"], "streak": streak_data["streak"]}
        )
        streak_data["history"] = streak_data["history"][-20:]
        _save_streak(streak_data)
    streak = int(streak_data["streak"])

    digest.append("## Suite results")
    digest.append("")
    digest.append(
        f"| Suite | PASS | PARTIAL | FAIL | scored | exit | bar |"
    )
    digest.append("|-------|------|---------|------|--------|------|-----|")
    digest.append(
        f"| production | {prod_sum.get('PASS', '?')} | {prod_sum.get('PARTIAL', '?')} | "
        f"{prod_sum.get('FAIL', '?')} | {prod_sum.get('scored', '?')} | {rc_prod} | "
        f"{'OK' if prod_ok else 'FAIL'} |"
    )
    digest.append(
        f"| break (full) | {break_sum.get('PASS', '?')} | {break_sum.get('PARTIAL', '?')} | "
        f"{break_sum.get('FAIL', '?')} | {break_sum.get('scored', '?')} | {rc_break} | "
        f"{'OK' if break_ok else 'FAIL'} |"
    )
    digest.append(
        f"| readiness | {ready_sum.get('PASS', '?')} | {ready_sum.get('PARTIAL', '?')} | "
        f"{ready_sum.get('FAIL', '?')} | {ready_sum.get('scored', '?')} | {rc_ready} | "
        f"{'GREEN' if ready_green else 'RED'} |"
    )
    digest.append(
        f"| benchmark_phases | {bench_sum.get('PASS', '?')} | {bench_sum.get('PARTIAL', 0)} | "
        f"{bench_sum.get('FAIL', '?')} | {bench_sum.get('scored', '?')} | {rc_bench} | "
        f"{'OK' if bench_ok else 'FAIL'} |"
    )
    digest.append(
        f"| customer_bakeoff | {bake_sum.get('PASS', '?')} | {bake_sum.get('PARTIAL', 0)} | "
        f"{bake_sum.get('FAIL', '?')} | {bake_sum.get('scored', '?')} | {rc_bake} | "
        f"{'OK' if bake_ok else 'FAIL'} |"
    )
    digest.append("")
    digest.append(
        f"**readiness_streak={streak}/{streak_target}** "
        f"(last={streak_data['last_gate']}, ci={ci})"
    )
    digest.append("")

    # Failures
    failures = _parse_case_blocks(prod_text) + _parse_case_blocks(break_text)
    # readiness failures
    for c in _parse_case_blocks(ready_text):
        c["id"] = f"ready:{c['id']}"
        failures.append(c)
    for c in _parse_case_blocks(bench_text):
        c["id"] = f"bench:{c['id']}"
        failures.append(c)
    for c in _parse_case_blocks(bake_text):
        c["id"] = f"bakeoff:{c['id']}"
        failures.append(c)

    digest.append("## FAIL / PARTIAL cases")
    digest.append("")
    if not failures:
        digest.append("_None._")
        digest.append("")
    else:
        by_tag: dict[str, list[str]] = {}
        for c in failures:
            tags = c.get("tags") or ["(untagged)"]
            for t in tags:
                by_tag.setdefault(t, []).append(c["id"])
            digest.append(
                f"### `{c['id']}` — {c['grade']}"
                + (f" ({c['attack']})" if c.get("attack") else "")
            )
            digest.append(f"- tags: {c.get('tags')}")
            digest.append(f"- suggested fix: {_suggest_fix(c.get('tags') or [], c.get('attack') or '')}")
            if c.get("preview"):
                digest.append(f"- preview: {c['preview']}")
            for n in c.get("notes") or []:
                digest.append(f"- note: {n}")
            digest.append("")
        digest.append("### Tag groups")
        digest.append("")
        for t, ids in sorted(by_tag.items()):
            digest.append(f"- **{t}**: {', '.join(ids)}")
        digest.append("")
        digest.append(
            "**Triage:** pick one fix category per round; apply a generic agent rule; re-run harness."
        )
        digest.append("")

    phase_gates_ok = bench_ok and bake_ok
    if ci:
        all_ok = prod_ok and break_ok and ready_green and phase_gates_ok
    else:
        all_ok = prod_ok and break_ok and streak >= streak_target and phase_gates_ok
    digest.append("## Verdict")
    digest.append("")
    if all_ok:
        if ci:
            digest.append(
                "**PASS** — production + break + readiness GREEN + Phase 1–5 "
                "benchmarks + customer bakeoff (CI single-run gate)."
            )
        else:
            digest.append(
                f"**PASS** — production + break bars met, readiness streak ≥ {streak_target}, "
                "Phase 1–5 benchmarks + customer bakeoff. "
                "Ready to discuss docs embed (separate decision)."
            )
    else:
        reasons = []
        if not prod_ok:
            reasons.append("production bar")
        if not break_ok:
            reasons.append("break bar")
        if ci:
            if not ready_green:
                reasons.append("readiness not GREEN")
        elif streak < streak_target:
            reasons.append(f"readiness streak {streak}/{streak_target}")
        if not bench_ok:
            reasons.append("benchmark_phases")
        if not bake_ok:
            reasons.append("customer bakeoff")
        digest.append(f"**FAIL** — need: {', '.join(reasons)}.")
        digest.append("")
        digest.append("Next: see Step 2 in `docs/QUALITY_LOOP.md`.")
    digest.append("")

    text = "\n".join(digest)
    with open(DIGEST_PATH, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nWrote {DIGEST_PATH}")
    if ci:
        print(f"CI mode: streak not persisted (this run={streak}/{streak_target})")
    else:
        print(f"Wrote {STREAK_PATH} (streak={streak})")

    return 0 if all_ok else 1


def _read_file(path: str) -> str:
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    sys.exit(main())
