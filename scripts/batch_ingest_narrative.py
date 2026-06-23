"""
batch_ingest_narrative.py: chunk-only validation for narrative markdown folders.

Usage:
    python3 scripts/batch_ingest_narrative.py beginners_guide reference_guides
    python3 scripts/batch_ingest_narrative.py   # uses DOCS_INCLUDE_DIRS from config
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import CFXQL_FILE, DOCS_INCLUDE_DIRS, DOCS_ROOT
from ingest_qdrant import chunk_narrative_markdown


def cfxql_rel_path():
    if os.path.isfile(CFXQL_FILE):
        return os.path.relpath(CFXQL_FILE, DOCS_ROOT).replace("\\", "/")
    return "reference_guides/cfxql.md"


def chunk_dirs(include_dirs):
    skip_rel = {cfxql_rel_path()}
    all_chunks = []
    errors = []
    file_count = 0

    for subdir in include_dirs:
        dir_path = os.path.join(DOCS_ROOT, subdir)
        if not os.path.isdir(dir_path):
            print(f"  {subdir}: not found under {DOCS_ROOT}")
            continue
        for root, _, files in os.walk(dir_path):
            for filename in sorted(files):
                if not filename.endswith(".md"):
                    continue
                filepath = os.path.join(root, filename)
                rel_source = os.path.relpath(filepath, DOCS_ROOT).replace("\\", "/")
                if rel_source in skip_rel:
                    continue
                doc_section = rel_source.split("/")[0]
                try:
                    chunks = chunk_narrative_markdown(filepath, rel_source, doc_section)
                    all_chunks.extend(chunks)
                    file_count += 1
                except Exception as e:
                    errors.append((rel_source, str(e)))

    return all_chunks, file_count, errors


def main():
    include_dirs = sys.argv[1:] if len(sys.argv) > 1 else DOCS_INCLUDE_DIRS
    if not include_dirs:
        print("Usage: python3 scripts/batch_ingest_narrative.py [dir ...]")
        sys.exit(1)

    print(f"DOCS_ROOT: {DOCS_ROOT}")
    print(f"Folders: {include_dirs}\n")

    all_chunks, file_count, errors = chunk_dirs(include_dirs)
    lengths = [len(c["text"]) for c in all_chunks]
    avg_len = sum(lengths) / len(lengths) if lengths else 0

    print(f"{'='*60}")
    print(f"Files chunked: {file_count}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Avg chunk size: {avg_len:.0f} chars")
    print(f"Errors: {len(errors)}")
    print("=" * 60)

    if errors:
        print("\nFailed files:")
        for rel, err in errors:
            print(f"  - {rel}: {err}")

    bad_type = sum(1 for c in all_chunks if c["metadata"].get("type") != "narrative")
    print(f"\nChunks with type != narrative: {bad_type} (should be 0)")


if __name__ == "__main__":
    main()
