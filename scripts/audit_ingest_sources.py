"""
audit_ingest_sources.py: verify ingest sources stay within the public docs export.

Mirrors the file discovery in ingest_qdrant.load_and_chunk_all() without chunking
or embedding. Fails if any source resolves outside DOCS_ROOT / BOTS_DIR, uses
legacy data/raw fallbacks when the markdown catalog is available, or matches
obvious secret path patterns.

Usage:
    python scripts/audit_ingest_sources.py
    python scripts/audit_ingest_sources.py --verify-urls   # optional HEAD checks vs docs.fabrix.ai
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import BOTS_DIR, CFXQL_FILE, DOCS_INCLUDE_DIRS, DOCS_ROOT

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
SKIP_RAW_FILES = {"c_extension_loop_bots.txt", "exec_and_dm_sink_bots.txt"}
PUBLIC_DOCS_BASE = "https://docs.fabrix.ai"
SECRET_PATH_MARKERS = (".env", "credentials", "secrets", "private", ".pem", ".key")


def _cfxql_rel_path():
    if os.path.isfile(CFXQL_FILE):
        return os.path.relpath(CFXQL_FILE, DOCS_ROOT).replace("\\", "/")
    return "reference_guides/cfxql.md"


def _is_under(path, root):
    path = os.path.realpath(path)
    root = os.path.realpath(root)
    return path == root or path.startswith(root + os.sep)


def _secret_path(path):
    lower = path.lower()
    return any(marker in lower for marker in SECRET_PATH_MARKERS)


def enumerate_ingest_files():
    """Return (abs_path, rel_label, category) for every file ingest would load."""
    files = []
    cfxql_loaded = os.path.isfile(CFXQL_FILE)

    if cfxql_loaded:
        rel = os.path.relpath(CFXQL_FILE, DOCS_ROOT).replace("\\", "/")
        files.append((CFXQL_FILE, rel, "cfxql"))

    for filename in sorted(os.listdir(RAW_DIR)):
        if not filename.endswith(".txt"):
            continue
        if filename in SKIP_RAW_FILES:
            continue
        if filename == "cfxql_reference.txt" and cfxql_loaded:
            continue
        filepath = os.path.join(RAW_DIR, filename)
        files.append((filepath, f"data/raw/{filename}", "raw_fallback"))

    if os.path.isdir(BOTS_DIR):
        for filename in sorted(f for f in os.listdir(BOTS_DIR) if f.endswith(".md")):
            filepath = os.path.join(BOTS_DIR, filename)
            rel = os.path.relpath(filepath, DOCS_ROOT).replace("\\", "/")
            files.append((filepath, rel, "bot"))

    if DOCS_INCLUDE_DIRS and os.path.isdir(DOCS_ROOT):
        skip_rel = {_cfxql_rel_path()}
        for subdir in DOCS_INCLUDE_DIRS:
            dir_path = os.path.join(DOCS_ROOT, subdir)
            if not os.path.isdir(dir_path):
                continue
            for root, _, names in os.walk(dir_path):
                for filename in sorted(names):
                    if not filename.endswith(".md"):
                        continue
                    filepath = os.path.join(root, filename)
                    rel = os.path.relpath(filepath, DOCS_ROOT).replace("\\", "/")
                    if rel in skip_rel:
                        continue
                    files.append((filepath, rel, "narrative"))

    return files


def public_doc_url(rel_path):
    """Map a repo-relative docs path to the public docs.fabrix.ai URL."""
    path = rel_path.replace("\\", "/")
    if path.endswith(".md"):
        path = path[:-3]
    parts = path.split("/")
    if parts[-1] in ("index", "index_release"):
        parts = parts[:-1]
    encoded = "/".join(quote(part, safe="") for part in parts if part)
    if not encoded:
        return f"{PUBLIC_DOCS_BASE}/"
    return f"{PUBLIC_DOCS_BASE}/{encoded}/"


def head_url(url):
    req = Request(url, method="HEAD")
    with urlopen(req, timeout=15) as resp:
        return resp.status


def verify_public_urls(rel_paths, workers=12):
    """HEAD-check each unique public URL; return list of (rel_path, url, error)."""
    unique = sorted(set(rel_paths))
    failures = []

    def check(rel):
        url = public_doc_url(rel)
        try:
            status = head_url(url)
            if status >= 400:
                return rel, url, f"HTTP {status}"
        except (HTTPError, URLError) as e:
            if isinstance(e, HTTPError):
                return rel, url, f"HTTP {e.code}"
            return rel, url, str(e.reason)
        except Exception as e:
            return rel, url, str(e)
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(check, rel): rel for rel in unique}
        for future in as_completed(futures):
            result = future.result()
            if result:
                failures.append(result)
    return failures


def main():
    parser = argparse.ArgumentParser(description="Audit ingest sources for scope leaks")
    parser.add_argument(
        "--verify-urls",
        action="store_true",
        help=f"HEAD-check each source against {PUBLIC_DOCS_BASE} (requires network)",
    )
    args = parser.parse_args()

    errors = []
    warnings = []

    if not os.path.isdir(DOCS_ROOT):
        print(f"ERROR: DOCS_ROOT does not exist: {DOCS_ROOT}")
        sys.exit(1)

    docs_root = os.path.realpath(DOCS_ROOT)
    bots_root = os.path.realpath(BOTS_DIR) if os.path.isdir(BOTS_DIR) else None

    print(f"DOCS_ROOT: {docs_root}")
    print(f"BOTS_DIR:  {bots_root or '(missing)'}")
    print(f"CFXQL_FILE: {CFXQL_FILE}")
    print(f"include:   {DOCS_INCLUDE_DIRS}\n")

    if os.path.isfile(CFXQL_FILE) and not _is_under(CFXQL_FILE, docs_root):
        errors.append(f"CFXQL_FILE outside DOCS_ROOT: {CFXQL_FILE}")

    if bots_root and not _is_under(bots_root, docs_root):
        warnings.append(
            f"BOTS_DIR is outside DOCS_ROOT (unusual but allowed if still your public export): {bots_root}"
        )

    ingest_files = enumerate_ingest_files()
    by_category = {}
    rel_for_url = []

    for abspath, rel, category in ingest_files:
        by_category[category] = by_category.get(category, 0) + 1
        rel_for_url.append(rel)

        if not os.path.isfile(abspath):
            errors.append(f"Missing ingest file: {abspath}")
            continue

        if _secret_path(abspath):
            errors.append(f"Secret-like path blocked: {rel}")

        allowed = _is_under(abspath, docs_root) or (bots_root and _is_under(abspath, bots_root))
        if not allowed:
            errors.append(f"Outside allowed roots: {rel} ({abspath})")

        if os.path.islink(abspath):
            target = os.path.realpath(abspath)
            if not (_is_under(target, docs_root) or (bots_root and _is_under(target, bots_root))):
                errors.append(f"Symlink escapes allowed roots: {rel} -> {target}")

        if category == "raw_fallback" and os.path.isdir(BOTS_DIR) and os.path.isfile(CFXQL_FILE):
            errors.append(
                f"Legacy raw fallback would be ingested (markdown catalog available): {rel}"
            )

    print("Ingest file counts:")
    for category in ("cfxql", "bot", "narrative", "raw_fallback"):
        if category in by_category:
            print(f"  {category}: {by_category[category]}")
    print(f"  total: {len(ingest_files)}\n")

    if warnings:
        print("Warnings:")
        for msg in warnings:
            print(f"  WARN  {msg}")
        print()

    if errors:
        print("Errors:")
        for msg in errors:
            print(f"  FAIL  {msg}")
        print(f"\nAudit FAILED ({len(errors)} error(s))")
        sys.exit(1)

    if args.verify_urls:
        print(f"Verifying {len(set(rel_for_url))} public URLs on {PUBLIC_DOCS_BASE} ...")
        url_failures = verify_public_urls(rel_for_url)
        if url_failures:
            print("URL check failures:")
            for rel, url, err in sorted(url_failures):
                print(f"  FAIL  {rel}")
                print(f"        {url} ({err})")
            print(f"\nAudit FAILED ({len(url_failures)} URL(s) not public)")
            sys.exit(1)
        print(f"  All {len(set(rel_for_url))} sources returned OK on {PUBLIC_DOCS_BASE}\n")

    print("Audit PASSED: all ingest sources are within the public docs export tree.")


if __name__ == "__main__":
    main()
