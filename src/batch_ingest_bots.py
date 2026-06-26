"""
batch_ingest_bots.py, chunk-only smoke test for the whole Bots/ folder.

Run: python3 src/batch_ingest_bots.py /path/to/Bots/
Reports total chunks and any files that failed to chunk.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ingest_qdrant import chunk_bot_catalog_markdown


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 batch_ingest_bots.py /path/to/Bots/")
        sys.exit(1)

    bots_dir = sys.argv[1]
    md_files = [f for f in os.listdir(bots_dir) if f.endswith(".md")]
    md_files.sort()

    print(f"Found {len(md_files)} files to process\n")

    all_chunks = []
    errors = []

    for i, filename in enumerate(md_files, 1):
        filepath = os.path.join(bots_dir, filename)
        try:
            chunks = chunk_bot_catalog_markdown(filepath, filename)
            all_chunks.extend(chunks)
            if i % 25 == 0 or i == len(md_files):
                print(f"  [{i}/{len(md_files)}] processed, {len(all_chunks)} total chunks so far")
        except Exception as e:
            errors.append((filename, str(e)))
            print(f"  [{i}/{len(md_files)}] FAILED on {filename}: {e}")

    print(f"\n{'='*60}")
    print(f"DONE. {len(md_files)} files processed, {len(errors)} errors")
    print(f"Total chunks produced: {len(all_chunks)}")
    print('='*60)

    if errors:
        print("\nFiles that failed:")
        for filename, error in errors:
            print(f"  - {filename}: {error}")

    # quick sanity check: how many chunks have NO bot_name (shouldn't happen)
    unknown_count = sum(1 for c in all_chunks if c["metadata"]["bot_name"] == "unknown")
    print(f"\nChunks with unknown bot_name: {unknown_count} (should be 0 or very low)")

    # prefix distribution --> quick sanity check on variety
    from collections import Counter
    prefix_counts = Counter(c["metadata"]["prefix"] for c in all_chunks)
    print(f"\nPrefix distribution: {dict(prefix_counts)}")


if __name__ == "__main__":
    main()
