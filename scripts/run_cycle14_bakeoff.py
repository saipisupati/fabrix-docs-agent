#!/usr/bin/env python3
"""
Cycle 14 discovery bakeoff: Fabrix agent (local) vs GPT-4o-mini + live docs browse proxy.

Not the ChatGPT web UI — same method as data/customer_bakeoff.json.
Stop the API first (Qdrant lock). Writes data/cycle14_bakeoff.json.

  PYTHONPATH=src ./venv/bin/python3 scripts/run_cycle14_bakeoff.py
"""

from __future__ import annotations

import json
import os
import re
import ssl
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from openai import OpenAI  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from agent import answer  # noqa: E402
from live_docs import html_to_text  # noqa: E402

OUT_PATH = Path(
    os.environ.get("BAKEOFF_OUT")
    or (ROOT / "data" / "cycle14_bakeoff.json")
)
QDRANT_DIR = str(ROOT / "data" / "qdrant_db")

QUESTIONS = [
    {
        "id": "c14_splunk_dashboard",
        "theme": "day2_ops / vague product",
        "q": "How do I map Splunk events into a Fabrix dashboard that ops can watch?",
    },
    {
        "id": "c14_datadog_slack_wiring",
        "theme": "multi_product wiring",
        "q": (
            "Walk me through wiring Datadog metrics into a persistent stream, "
            "then alerting Slack."
        ),
    },
    {
        "id": "c14_schedule_debug",
        "theme": "day2_ops",
        "q": "What's the right way to debug why a scheduled pipeline didn't fire last night?",
    },
    {
        "id": "c14_concurrent_dataset",
        "theme": "day2_ops / overclaim risk",
        "q": (
            "How does Fabrix handle concurrent writes to the same dataset "
            "from two pipelines?"
        ),
    },
    {
        "id": "c14_worker_limits",
        "theme": "day2_ops / specificity",
        "q": "Where do I configure an RDA worker's max concurrent bots for a busy site?",
    },
    {
        "id": "c14_snow_change_schedule",
        "theme": "integration wiring + schedule",
        "q": (
            "What's the easiest path to pull ServiceNow change tickets into Fabrix "
            "on a recurring schedule?"
        ),
    },
    {
        "id": "c14_kafka_to_dataset",
        "theme": "vague product / building blocks",
        "q": "Explain the building blocks I'd use to turn Kafka messages into a searchable dataset.",
    },
    {
        "id": "c14_weekday_schedule_ui",
        "theme": "pipelines / honesty",
        "q": "Can I use the Fabrix UI to set a pipeline to run only on weekdays 9–5?",
    },
    {
        "id": "c14_enterprise_price_trap",
        "theme": "out of scope",
        "q": "What is Fabrix's list price for the enterprise AI Fabric add-on?",
    },
    {
        "id": "c14_snowv2_list_params",
        "theme": "bot lookup",
        "q": "What parameters does @snowv2:list-incidents take?",
    },
    {
        "id": "c14_newrelic_dashboard",
        "theme": "integration wiring",
        "q": "How would I connect New Relic APM as a datasource and land it in a dashboard?",
    },
    {
        "id": "c14_dashboard_kickoff",
        "theme": "overclaim / capability",
        "q": (
            "Is there a documented way for dashboards to kick off remediation "
            "pipelines when an alert fires?"
        ),
    },
]

try:
    import certifi

    CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    CTX = ssl.create_default_context()

oa = OpenAI()


def fetch(url: str, timeout: float = 25) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "fabrix-cycle14-bakeoff/1.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.read().decode("utf-8", "replace")


def pick_pages(docs: list, question: str, k: int = 3) -> list[tuple[str, str]]:
    qlow = question.lower()
    terms = set(re.findall(r"[a-z0-9_-]{3,}", qlow))
    scored = []
    for d in docs:
        loc = (d.get("location") or "").split("#")[0]
        if not loc or loc.endswith((".csv", ".parquet", ".png", ".jpg")):
            continue
        title = d.get("title") or ""
        text = (d.get("text") or "")[:500]
        blob = (loc + " " + title + " " + text).lower()
        score = sum(1 for t in terms if t in blob)
        if score:
            scored.append((score, loc, title))
    scored.sort(key=lambda x: (-x[0], x[1]))
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for _, loc, title in scored:
        if loc in seen:
            continue
        seen.add(loc)
        out.append((loc, title))
        if len(out) >= k:
            break
    seeds = [
        "beginners_guide/scheduled_pipelines",
        "Datasource_Integrations/datadog",
        "Datasource_Integrations/splunk",
        "Bots/kafka-v2",
        "Bots/servicenow_v2",
        "ai_fabric/fabio_copilot",
        "beginners_guide",
    ]
    for seed in seeds:
        if len(out) >= k:
            break
        if seed not in seen:
            out.append((seed, seed))
            seen.add(seed)
    return out[:k]


def ask_agent(q: str, client: QdrantClient) -> tuple[dict, float]:
    t0 = time.time()
    r = answer(q, client=client)
    return {
        "answer": r.answer,
        "scope": r.scope,
        "sources": r.sources,
        "gaps": r.gaps,
        "used_inference": r.used_inference,
        "timing": r.timing,
    }, time.time() - t0


def ask_chatgpt_proxy(q: str, pages: list[tuple[str, str]]) -> tuple[str, float, list[str]]:
    ctx = [f"### PAGE: {url}\n{text}\n" for url, text in pages]
    prompt = f"""You are ChatGPT answering a customer question using ONLY the Fabrix docs pages below
(as if you browsed docs.fabrix.ai). Be practical and structured. If the docs don't cover something,
say so clearly. Prefer concrete commands/parameters from the pages.

CUSTOMER QUESTION:
{q}

BROWSED DOCS:
{chr(10).join(ctx)}
"""
    t0 = time.time()
    resp = oa.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    ans = resp.choices[0].message.content or ""
    return ans, time.time() - t0, [u for u, _ in pages]


def grade(q: str, agent_ans: str, gpt_ans: str, page_texts: list[tuple[str, str]]) -> dict:
    joined = " ".join(t for _, t in page_texts)[:12000]
    prompt = f"""Grade two answers to a Fabrix customer question against the docs excerpts.
Return JSON only:
{{
  "agent": {{"accuracy":"strong|mixed|weak|wrong","docs_grounded":true/false,"issues":["short"],"strengths":["short"],"fail_class":"none|thin_wiring|contamination|overclaim|wrong_facet|abstain_fail|empty_sources_leak|latency|other"}},
  "chatgpt_proxy": {{"accuracy":"strong|mixed|weak|wrong","docs_grounded":true/false,"issues":["short"],"strengths":["short"],"fail_class":"none|thin_wiring|contamination|overclaim|wrong_facet|abstain_fail|empty_sources_leak|latency|other"}},
  "winner":"agent|chatgpt_proxy|tie",
  "reason":"one sentence",
  "agent_miss_for_cycle": true/false
}}
Rules: punish invented commands/params, wrong product drift, false confidence on OOS topics.
Reward correct params, honest gaps, right product family, complete cred→bot→stream wiring.
Prefer honesty on out-of-scope. Set agent_miss_for_cycle=true if the agent lost OR has weak/wrong accuracy OR a non-none fail_class worth locking in eval.

QUESTION: {q}

DOCS EXCERPTS:
{joined}

AGENT ANSWER:
{(agent_ans or "")[:3500]}

CHATGPT_PROXY ANSWER:
{(gpt_ans or "")[:3500]}
"""
    resp = oa.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content or "{}"
    m = re.search(r"\{[\s\S]*\}", raw)
    try:
        return json.loads(m.group(0) if m else raw)
    except Exception:
        return {
            "winner": "tie",
            "reason": "grade parse failed",
            "agent": {},
            "chatgpt_proxy": {},
            "agent_miss_for_cycle": False,
        }


def main() -> int:
    print("fetching search index…", flush=True)
    idx = json.loads(fetch("https://docs.fabrix.ai/search/search_index.json"))
    docs = idx.get("docs") or []
    print("index docs", len(docs), flush=True)

    client = QdrantClient(path=QDRANT_DIR)
    results = []
    try:
        for i, item in enumerate(QUESTIONS, 1):
            q = item["q"]
            print(f"[{i}/{len(QUESTIONS)}] {item['id']} …", flush=True)
            pages_meta = pick_pages(docs, q, k=3)
            page_texts: list[tuple[str, str]] = []
            for loc, _title in pages_meta:
                path = loc.strip("/")
                url = "https://docs.fabrix.ai/" + (path + "/" if path else "")
                try:
                    html = fetch(url)
                    page_texts.append((url, (html_to_text(html) or "")[:9000]))
                    print(f"  browsed {url} ({len(page_texts[-1][1])} chars)", flush=True)
                except Exception as e:
                    print(f"  browse fail {url}: {e}", flush=True)

            try:
                agent, at = ask_agent(q, client)
            except Exception as e:
                agent = {"answer": f"ERROR: {e}", "scope": "error", "sources": [], "gaps": []}
                at = 0.0

            gpt_ans, gt, gpt_urls = ask_chatgpt_proxy(
                q, page_texts or [("https://docs.fabrix.ai/", "(no pages)")]
            )
            g = grade(q, agent.get("answer") or "", gpt_ans, page_texts)

            srcs = []
            for s in (agent.get("sources") or [])[:6]:
                if isinstance(s, dict):
                    srcs.append(s.get("url") or s.get("source") or "")
                else:
                    srcs.append(str(s))

            row = {
                "id": item["id"],
                "theme": item["theme"],
                "question": q,
                "agent": {
                    "answer": (agent.get("answer") or "")[:2200],
                    "scope": agent.get("scope"),
                    "sources": srcs,
                    "gaps": (agent.get("gaps") or [])[:4],
                    "latency_s": round(at, 1),
                    "used_inference": agent.get("used_inference"),
                },
                "chatgpt_proxy": {
                    "answer": gpt_ans[:2200],
                    "sources": gpt_urls,
                    "latency_s": round(gt, 1),
                    "note": (
                        "GPT-4o-mini + live docs.fabrix.ai pages (browse proxy), "
                        "not ChatGPT web UI"
                    ),
                },
                "grade": g,
            }
            results.append(row)
            print(
                f"  winner={g.get('winner')} "
                f"agent={(g.get('agent') or {}).get('accuracy')} "
                f"gpt={(g.get('chatgpt_proxy') or {}).get('accuracy')} "
                f"miss={g.get('agent_miss_for_cycle')}",
                flush=True,
            )
    finally:
        client.close()

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "agent=local answer() ; chatgpt_proxy=gpt-4o-mini+live docs browse ; judge=gpt-4o-mini",
        "disclaimer": (
            "ChatGPT column is an automated browse proxy (same model family + live docs pages), "
            "not chat.openai.com UI."
        ),
        "questions": results,
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print("WROTE", OUT_PATH)
    print("WINNERS", dict(Counter(r["grade"].get("winner") for r in results)))
    misses = [r["id"] for r in results if r["grade"].get("agent_miss_for_cycle")]
    print("AGENT_MISSES", misses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
