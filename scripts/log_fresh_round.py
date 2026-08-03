#!/usr/bin/env python3
"""
Log a fresh-question round to JSONL (additive testing support only).

Does not change agent routing, guardrails, API, ingest, KB, or eval harness.
Stop the API first if you hit a Qdrant lock (local file DB is single-writer).

Usage:
  PYTHONPATH=src ./venv/bin/python scripts/log_fresh_round.py \\
    "What's the difference between a Pipeline and a Blueprint?"

  PYTHONPATH=src ./venv/bin/python scripts/log_fresh_round.py \\
    --note "wrong facet toward AI Fabric" --bug-tag "wrong_facet" \\
    "How do service pipelines differ from event-driven?"

  # one question per line
  PYTHONPATH=src ./venv/bin/python scripts/log_fresh_round.py --file qs.txt

  # interactive: type questions, blank line to quit
  PYTHONPATH=src ./venv/bin/python scripts/log_fresh_round.py -i

Default output: data/fresh_rounds.jsonl
Override with FRESH_ROUNDS_OUT or --out.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_OUT = ROOT / "data" / "fresh_rounds.jsonl"


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _source_strings(sources: list[dict] | None) -> list[str]:
    out: list[str] = []
    for s in sources or []:
        url = (s.get("url") or "").strip()
        title = (s.get("title") or "").strip()
        if url and title:
            out.append(f"{title} | {url}")
        elif url:
            out.append(url)
        elif title:
            out.append(title)
    return out


def _append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_one_direct(
    question: str,
    *,
    client,
) -> tuple[str, list[str]]:
    from agent import answer

    result = answer(question, client=client)
    return result.answer or "", _source_strings(result.sources)


def _run_one_via_api(question: str, *, api_url: str) -> tuple[str, list[str]]:
    """Call the running API (shares its Qdrant lock) — same agent.answer() path."""
    import urllib.error
    import urllib.request

    url = api_url.rstrip("/") + "/ask"
    payload = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"API request failed ({url}): {e}. "
            "Is uvicorn up, or use direct mode (no --via-api) with API stopped?"
        ) from e
    answer_text = data.get("answer") or ""
    sources = _source_strings(data.get("sources") or [])
    return answer_text, sources


def _run_one(
    question: str,
    *,
    out_path: Path,
    note: str,
    bug_tag: str | None,
    client=None,
    via_api: str | None = None,
) -> dict:
    q = (question or "").strip()
    if not q:
        raise ValueError("empty question")

    if via_api:
        answer_text, sources = _run_one_via_api(q, api_url=via_api)
    else:
        if client is None:
            raise ValueError("client required unless --via-api is set")
        answer_text, sources = _run_one_direct(q, client=client)

    record = {
        "timestamp": _iso_now(),
        "question": q,
        "agent_answer": answer_text,
        "sources": sources,
        "note": note or "",
        "bug_tag": bug_tag,
    }
    _append_record(out_path, record)
    return record


def _print_record(record: dict) -> None:
    print("--- logged ---")
    print(f"timestamp: {record['timestamp']}")
    print(f"question:  {record['question']}")
    print(f"bug_tag:   {record.get('bug_tag')}")
    print(f"note:      {record.get('note') or '(none)'}")
    print(f"sources:   {len(record.get('sources') or [])}")
    ans = record.get("agent_answer") or ""
    preview = ans if len(ans) <= 400 else ans[:400] + "…"
    print(preview)
    print("--------------")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Log fresh agent questions to JSONL (no agent behavior changes)."
    )
    parser.add_argument(
        "questions",
        nargs="*",
        help="One or more questions (or use --file / -i)",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        help="Text file with one question per line (# comments ok)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Prompt for questions until a blank line",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"JSONL path (default: {DEFAULT_OUT} or $FRESH_ROUNDS_OUT)",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Optional free-text note stored on each record this run",
    )
    parser.add_argument(
        "--bug-tag",
        default=None,
        help="Optional free-text bug tag (omit for null)",
    )
    parser.add_argument(
        "--prompt-meta",
        action="store_true",
        help="After each answer, prompt for note / bug_tag (overrides --note/--bug-tag)",
    )
    parser.add_argument(
        "--via-api",
        action="store_true",
        help=(
            "Call POST /ask on a running API instead of opening local Qdrant "
            "(same agent.answer() path). Default URL http://127.0.0.1:8080; "
            "override with --api-url."
        ),
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8080",
        help="API base URL when --via-api is set (default http://127.0.0.1:8080)",
    )
    args = parser.parse_args()

    _load_dotenv()

    out_path = args.out
    if out_path is None:
        out_path = Path(os.environ.get("FRESH_ROUNDS_OUT") or DEFAULT_OUT)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    questions: list[str] = []
    for q in args.questions:
        if q.strip():
            questions.append(q.strip())
    if args.file:
        for line in args.file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            questions.append(line)

    if args.interactive:
        print("Interactive fresh round — blank line to quit.")
        print(f"Logging to {out_path}")
        while True:
            try:
                q = input("Q> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                break
            questions.append(q)

    if not questions:
        parser.error("provide a question, --file, or -i")

    via_api = args.api_url if args.via_api else None
    print(f"Appending {len(questions)} record(s) → {out_path}")
    if via_api:
        print(f"Via API: {via_api}")
    else:
        print("(Stop the API first if Qdrant lock errors appear.)")

    client = None
    if not via_api:
        from qdrant_client import QdrantClient
        from config import QDRANT_DIR

        client = QdrantClient(path=QDRANT_DIR)

    try:
        for q in questions:
            note = args.note
            bug_tag = args.bug_tag
            if args.prompt_meta:
                # Run first so you tag what you saw.
                record = _run_one(
                    q,
                    out_path=out_path,
                    note="",
                    bug_tag=None,
                    client=client,
                    via_api=via_api,
                )
                _print_record(record)
                try:
                    note = input("note (enter to keep empty)> ").strip()
                    tag_raw = input("bug_tag (enter for null)> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    note, tag_raw = "", ""
                bug_tag = tag_raw or None
                # Rewrite last line with meta (file is append-only otherwise).
                _rewrite_last_meta(out_path, note=note, bug_tag=bug_tag)
                continue

            record = _run_one(
                q,
                out_path=out_path,
                note=note,
                bug_tag=bug_tag,
                client=client,
                via_api=via_api,
            )
            _print_record(record)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    print(f"Done. Log: {out_path}")
    return 0


def _rewrite_last_meta(path: Path, *, note: str, bug_tag: str | None) -> None:
    """Update note/bug_tag on the last JSONL record (prompt-meta path only)."""
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return
    last = json.loads(lines[-1])
    last["note"] = note or ""
    last["bug_tag"] = bug_tag
    lines[-1] = json.dumps(last, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"updated meta: note={last['note']!r} bug_tag={last['bug_tag']!r}")


if __name__ == "__main__":
    raise SystemExit(main())
