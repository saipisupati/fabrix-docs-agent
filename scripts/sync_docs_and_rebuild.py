#!/usr/bin/env python3
"""
sync_docs_and_rebuild.py — keep the local corpus aligned with public Fabrix docs.

Preferred flow (batch, not per-chat):
  1. Refresh markdown under DOCS_ROOT / BOTS_DIR via scrape (--scrape) or rsync/git.
  2. Audit that every ingest path is still public-safe.
  3. Re-chunk + embed into Qdrant, then rebuild the structured KB.

This is NOT live scraping on every /ask. Refresh the snapshot, then retrieve locally.

Examples:
  python3 scripts/sync_docs_and_rebuild.py --check-live
  python3 scripts/sync_docs_and_rebuild.py --scrape
  # Stop API first (Qdrant lock):
  python3 scripts/sync_docs_and_rebuild.py --scrape-rebuild
  python3 scripts/sync_docs_and_rebuild.py --rebuild

Env: BOTS_DIR, DOCS_ROOT, CFXQL_FILE, OPENROUTER_API_KEY (for --rebuild).
"""

from __future__ import annotations

import argparse
import os
import ssl
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from doc_urls import PUBLIC_DOCS_BASE  # noqa: E402

# Pages ChatGPT often lands on for install / VM / prerequisites
LIVE_PROBE_PATHS = (
    "/",
    "/installation_guides/",
    "/installation_guides/deployment/",
    "/beginners_guide/",
    "/ai_fabric/",
)


def _ssl_context() -> ssl.SSLContext:
    """Prefer certifi CAs when present (macOS system Python often lacks them)."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


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
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            os.environ.setdefault(key, val)


def check_live(timeout: float = 15.0) -> int:
    """HEAD/GET key public docs URLs; print status. Exit 1 if any fail."""
    print(f"Probing live docs at {PUBLIC_DOCS_BASE}")
    ctx = _ssl_context()
    failed = 0
    for path in LIVE_PROBE_PATHS:
        url = PUBLIC_DOCS_BASE.rstrip("/") + path
        try:
            req = Request(url, method="HEAD", headers={"User-Agent": "fabrix-docs-agent-sync/1.0"})
            with urlopen(req, timeout=timeout, context=ctx) as resp:
                code = getattr(resp, "status", None) or resp.getcode()
            if code >= 400:
                # Some hosts dislike HEAD — retry GET
                req = Request(url, headers={"User-Agent": "fabrix-docs-agent-sync/1.0"})
                with urlopen(req, timeout=timeout, context=ctx) as resp:
                    code = getattr(resp, "status", None) or resp.getcode()
            print(f"  {code}  {url}")
            if int(code) >= 400:
                failed += 1
        except (HTTPError, URLError, TimeoutError, OSError) as e:
            print(f"  FAIL {url}  ({e})")
            failed += 1
    if failed:
        print(f"\n{failed} live probe(s) failed.")
        return 1
    print("\nLive probes OK. Next: --scrape or refresh local markdown, then --rebuild.")
    return 0


def run_scrape() -> int:
    """Mirror docs.fabrix.ai → DOCS_ROOT via scrape_docs_site.py."""
    docs_root = os.environ.get("DOCS_ROOT", "").strip()
    if not docs_root:
        docs_root = os.path.join(ROOT, "data", "docs_live_scrape")
        os.environ["DOCS_ROOT"] = docs_root
        print(f"DOCS_ROOT unset; using {docs_root}")
    os.makedirs(docs_root, exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "scrape_docs_site.py"),
        "--out",
        docs_root,
    ]
    print("+", " ".join(cmd))
    print("NOTE: Stop uvicorn/API before --rebuild (Qdrant file lock).")
    return subprocess.call(cmd, cwd=ROOT)


def run_audit() -> int:
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "audit_ingest_sources.py")]
    print("+", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def run_rebuild() -> int:
    """Ingest Qdrant + build KB. Stop the API first (local Qdrant file lock)."""
    print("NOTE: Stop uvicorn/API before rebuild (Qdrant file lock).")
    steps = [
        [sys.executable, os.path.join(ROOT, "src", "ingest_qdrant.py"), "--incremental"],
        [sys.executable, os.path.join(ROOT, "src", "build_kb.py")],
    ]
    for cmd in steps:
        print("+", " ".join(cmd))
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            print(f"FAILED ({rc}): {' '.join(cmd)}", file=sys.stderr)
            return rc
    print("\nRebuild complete. Run: python3 tests/run_quality_harness.py")
    return 0


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-live",
        action="store_true",
        help="Probe docs.fabrix.ai key URLs (incl. /installation_guides/)",
    )
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="Mirror docs.fabrix.ai → DOCS_ROOT via scrape_docs_site.py",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run scripts/audit_ingest_sources.py only",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="After MD refresh: ingest_qdrant.py + build_kb.py (stop API first)",
    )
    parser.add_argument(
        "--scrape-rebuild",
        action="store_true",
        help="scrape → audit → rebuild (stop API first)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="check-live → audit → rebuild (requires local MD already updated)",
    )
    args = parser.parse_args()

    if not any(
        (
            args.check_live,
            args.scrape,
            args.audit_only,
            args.rebuild,
            args.scrape_rebuild,
            args.all,
        )
    ):
        parser.print_help()
        print(
            "\nTypical order:\n"
            "  1. python3 scripts/sync_docs_and_rebuild.py --check-live\n"
            "  2. Stop API (Qdrant lock)\n"
            "  3. python3 scripts/sync_docs_and_rebuild.py --scrape-rebuild\n"
            "  4. python3 tests/run_quality_harness.py\n"
            "Do not scrape on every /ask — batch refresh only.\n"
        )
        return 0

    if args.scrape_rebuild:
        rc = run_scrape()
        if rc != 0:
            return rc
        rc = run_audit()
        if rc != 0:
            return rc
        return run_rebuild()

    if args.all or args.check_live:
        rc = check_live()
        if rc != 0 and not args.all:
            return rc
        if rc != 0 and args.all:
            print("Continuing after live probe failures (--all)…", file=sys.stderr)

    if args.scrape:
        rc = run_scrape()
        if rc != 0:
            return rc
        if not (args.rebuild or args.all or args.audit_only):
            return 0

    if args.all or args.audit_only or args.rebuild:
        rc = run_audit()
        if rc != 0:
            return rc
        if args.audit_only and not (args.rebuild or args.all):
            return 0

    if args.all or args.rebuild:
        return run_rebuild()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
