#!/usr/bin/env python3
"""
run_freshness_pipeline.py — Phase 6 end-to-end freshness gate.

  check-live → scrape → audit → (ingest → build_kb → evals if content changed)

Stop the API (uvicorn) before running — Qdrant file lock.

  python3 scripts/run_freshness_pipeline.py
  python3 scripts/run_freshness_pipeline.py --force

Env: DOCS_ROOT, BOTS_DIR, CFXQL_FILE, OPENROUTER_API_KEY, OPENAI_API_KEY.
     FORCE_REBUILD=1 same as --force.
Exit non-zero on any step failure.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TESTS = os.path.join(ROOT, "tests")
sys.path.insert(0, os.path.join(ROOT, "src"))


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild ingest/KB even when scrape hashes are unchanged",
    )
    args = parser.parse_args()
    force = args.force or os.environ.get("FORCE_REBUILD", "").strip() in ("1", "true", "yes")

    print(
        "Phase 6 freshness pipeline\n"
        "NOTE: Stop uvicorn/API first (Qdrant file lock).\n",
        flush=True,
    )

    from config import FRESHNESS_STATUS_PATH, SCRAPE_MANIFEST_PATH
    from freshness import load_manifest, should_rebuild, utc_now, write_freshness_status

    sync = os.path.join(ROOT, "scripts", "sync_docs_and_rebuild.py")
    py = sys.executable

    prefix: list[tuple[list[str], dict | None]] = [
        ([py, sync, "--check-live"], None),
        ([py, sync, "--scrape"], None),
        ([py, sync, "--audit-only"], None),
    ]
    rebuild_steps: list[tuple[list[str], dict | None]] = [
        (
            [
                py,
                os.path.join(ROOT, "src", "ingest_qdrant.py"),
                "--full" if force else "--incremental",
            ],
            None,
        ),
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

    for cmd, env in prefix:
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

    man = load_manifest(SCRAPE_MANIFEST_PATH)
    rebuild = force or should_rebuild(man)
    status = {
        "last_scrape_at": man.get("scraped_at"),
        "has_content_changes": man.get("has_content_changes"),
        "changed_n": len(man.get("changed_paths") or []),
        "added_n": len(man.get("added_paths") or []),
        "removed_n": len(man.get("removed_paths") or []),
        "rebuild": rebuild,
        "forced": force,
        "incremental": bool(rebuild and not force),
        "running": False,
        "finished_at": utc_now(),
    }
    if not rebuild:
        print(
            "No content changes — skipping ingest/build_kb/evals "
            "(use --force or FORCE_REBUILD=1 to rebuild anyway).",
            flush=True,
        )
        write_freshness_status(status, FRESHNESS_STATUS_PATH)
        print("\nFreshness pipeline PASS (scrape-only).", flush=True)
        return 0

    for cmd, env in rebuild_steps:
        rc = _run(cmd, env)
        if rc != 0:
            print(f"FAILED ({rc}): {' '.join(cmd)}", file=sys.stderr, flush=True)
            status["rebuild_error"] = " ".join(cmd)
            write_freshness_status(status, FRESHNESS_STATUS_PATH)
            return rc

    write_freshness_status(status, FRESHNESS_STATUS_PATH)
    print("\nFreshness pipeline PASS.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
