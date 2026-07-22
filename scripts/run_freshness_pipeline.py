#!/usr/bin/env python3
"""
run_freshness_pipeline.py — Phase 6 end-to-end freshness gate.

  check-live → scrape → audit → ingest → build_kb →
  benchmark_phases → eval_customer_bakeoff (BAKEOFF_MODE=local)

Stop the API (uvicorn) before running — Qdrant file lock.

  python3 scripts/run_freshness_pipeline.py

Env: DOCS_ROOT, BOTS_DIR, CFXQL_FILE, OPENROUTER_API_KEY, OPENAI_API_KEY.
Exit non-zero on any step failure.
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TESTS = os.path.join(ROOT, "tests")


def _load_dotenv() -> None:
    env_path = os.path.join(ROOT, ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip("'").strip('"'))


def _run(cmd: list[str], env: dict | None = None) -> int:
    print("\n==========", " ".join(cmd), "==========\n", flush=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.call(cmd, cwd=ROOT, env=merged)


def main() -> int:
    _load_dotenv()
    print(
        "Phase 6 freshness pipeline\n"
        "NOTE: Stop uvicorn/API first (Qdrant file lock).\n",
        flush=True,
    )

    sync = os.path.join(ROOT, "scripts", "sync_docs_and_rebuild.py")
    py = sys.executable

    steps: list[tuple[list[str], dict | None]] = [
        ([py, sync, "--check-live"], None),
        ([py, sync, "--scrape"], None),
        ([py, sync, "--audit-only"], None),
        ([py, os.path.join(ROOT, "src", "ingest_qdrant.py")], None),
        ([py, os.path.join(ROOT, "src", "build_kb.py")], None),
        (
            [py, os.path.join(TESTS, "benchmark_phases.py")],
            {"PYTHONPATH": os.path.join(ROOT, "src")},
        ),
        (
            [py, os.path.join(TESTS, "eval_customer_bakeoff.py")],
            {
                "PYTHONPATH": os.path.join(ROOT, "src"),
                "BAKEOFF_MODE": "local",
            },
        ),
    ]

    for cmd, env in steps:
        # check-live failure is warn-only so a transient docs CDN blip does not block rebuild
        rc = _run(cmd, env)
        if rc != 0:
            if cmd[-1] == "--check-live":
                print(
                    f"WARN: check-live exited {rc}; continuing freshness pipeline…",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            print(f"FAILED ({rc}): {' '.join(cmd)}", file=sys.stderr, flush=True)
            return rc

    print("\nFreshness pipeline PASS.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
