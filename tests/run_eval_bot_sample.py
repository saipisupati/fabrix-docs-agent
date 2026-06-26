"""
run_eval_bot_sample.py, random bot retrieval spot check (default 15 bots, seed 42).

Run: python3 tests/run_eval_bot_sample.py
Catches generic slugs that vector search misses.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from qdrant_client import QdrantClient

from config import BOTS_DIR, QDRANT_DIR
from eval_set import EVAL_SET
from query_qdrant import retrieve

BOT_NAME_RE = re.compile(r"@([a-z0-9]+):([a-z0-9-]+)", re.IGNORECASE)
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval_bot_sample_results.txt")
DEFAULT_SAMPLE_SIZE = 15
DEFAULT_SEED = 42


def eval_bot_slugs():
    slugs = set()
    for case in EVAL_SET:
        expected = case.get("expected_source", "")
        if expected.startswith("@"):
            slugs.add(expected.split(":")[-1].lower())
    return slugs


def discover_bots():
    bots = []
    if not os.path.isdir(BOTS_DIR):
        return bots
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
            for prefix, slug in BOT_NAME_RE.findall(text):
                bot_name = f"@{prefix}:{slug}"
                bots.append({"bot_name": bot_name, "slug": slug.lower()})
    # stable unique by bot_name
    seen = set()
    unique = []
    for bot in sorted(bots, key=lambda b: b["bot_name"].lower()):
        if bot["bot_name"] not in seen:
            seen.add(bot["bot_name"])
            unique.append(bot)
    return unique


def sample_bots(all_bots, sample_size, seed):
    exclude = eval_bot_slugs()
    pool = [b for b in all_bots if b["slug"] not in exclude]
    rng = random.Random(seed)
    if sample_size >= len(pool):
        return pool
    return rng.sample(pool, sample_size)


def question_for(bot):
    return f"What parameters does the {bot['slug']} bot take?"


def grade_top_hit(bot, chunks):
    if not chunks:
        return "FAIL", "no chunks retrieved"
    top_name = chunks[0]["metadata"].get("bot_name", "").lower()
    expected = bot["bot_name"].lower()
    if top_name == expected:
        return "PASS", top_name
    return "FAIL", f"expected {expected}, got {top_name}"


def main():
    parser = argparse.ArgumentParser(description="Random bot retrieval sample eval")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if "OPENROUTER_API_KEY" not in os.environ:
        print("SKIP: missing OPENROUTER_API_KEY")
        sys.exit(0)

    if not os.path.isdir(QDRANT_DIR):
        print(f"SKIP: Qdrant DB not found at {QDRANT_DIR}")
        sys.exit(0)

    all_bots = discover_bots()
    if not all_bots:
        print(f"SKIP: no bots found under BOTS_DIR={BOTS_DIR}")
        sys.exit(0)

    picked = sample_bots(all_bots, args.sample_size, args.seed)
    client = QdrantClient(path=QDRANT_DIR)

    lines = [
        f"Bot sample eval: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"mode: retrieval-only, sample_size={len(picked)}, seed={args.seed}",
        f"pool={len(all_bots)} bots, excluded {len(eval_bot_slugs())} already in eval_set",
        "",
    ]

    grades = {"PASS": 0, "FAIL": 0}
    for bot in picked:
        q = question_for(bot)
        chunks = retrieve(q, client, top_k=5, filter_dict={"type": "bot"})
        grade, detail = grade_top_hit(bot, chunks)
        grades[grade] += 1
        lines.append(f"[{grade}] {bot['bot_name']}")
        lines.append(f"  Q: {q}")
        lines.append(f"  Top hit: {detail}")
        lines.append("")

    lines.append(
        f"Summary: PASS={grades['PASS']} FAIL={grades['FAIL']} "
        f"(scored {len(picked)}/{len(picked)})"
    )

    report = "\n".join(lines)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(report)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
