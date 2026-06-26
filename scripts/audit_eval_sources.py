"""
audit_eval_sources.py, sanity check that eval cases point at bots that exist on disk.

Run before eval: python3 scripts/audit_eval_sources.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from config import BOTS_DIR
from eval_set import EVAL_SET

BOT_NAME_RE = re.compile(r"[@*][a-z0-9:-]+", re.IGNORECASE)


def bot_names_from_eval():
    names = []
    for case in EVAL_SET:
        expected = case["expected_source"]
        if expected.lower().startswith("none"):
            continue
        for match in BOT_NAME_RE.findall(expected):
            names.append((case["id"], match))
    return names


def search_bots_dir(bot_name):
    hits = []
    if not os.path.isdir(BOTS_DIR):
        return hits
    needle = bot_name.lower()
    for root, _, files in os.walk(BOTS_DIR):
        for filename in files:
            if not filename.endswith(".md"):
                continue
            path = os.path.join(root, filename)
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except OSError:
                continue
            if needle in text.lower():
                rel = os.path.relpath(path, BOTS_DIR)
                hits.append(rel)
    return hits


def main():
    print(f"BOTS_DIR: {BOTS_DIR}")
    if not os.path.isdir(BOTS_DIR):
        print("ERROR: BOTS_DIR does not exist: set BOTS_DIR env var")
        sys.exit(1)

    names = bot_names_from_eval()
    if not names:
        print("No bot names to audit in eval_set.py")
        return

    found = 0
    missing = 0
    for case_id, bot_name in names:
        hits = search_bots_dir(bot_name)
        if hits:
            found += 1
            print(f"FOUND   [{case_id}] {bot_name}")
            for hit in hits:
                print(f"          → {hit}")
        else:
            missing += 1
            print(f"MISSING [{case_id}] {bot_name}")

    print(f"\nSummary: {found} found, {missing} missing (of {len(names)} bot references)")


if __name__ == "__main__":
    main()
