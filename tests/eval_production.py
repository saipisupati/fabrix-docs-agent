"""
eval_production.py — production-style Fabrix ops battery for the reasoning agent.

Run: python3 tests/eval_production.py
Writes tests/eval_production_results.txt (gitignored).
Requires OPENAI_API_KEY, OPENROUTER_API_KEY, data/qdrant_db/, data/kb/.
Stop the API first (local Qdrant file lock).
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qdrant_client import QdrantClient

from agent import answer
from config import KB_DIR, QDRANT_DIR

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval_production_results.txt")

PRODUCTION_CASES = [
    {
        "id": "synth_k8s",
        "question": "How would I chain a Kubernetes inventory collection into a dashboard-friendly dataset?",
        "need_any": ["kubernetes-inventory", "kubectl", "ssh", "http"],
        "forbid": ["linux-inventory"],
        "expect_infer": True,
        "need_dataset_or_infer": True,
    },
    {
        "id": "synth_snow",
        "question": (
            "Our NOC wants ServiceNow incidents in a Fabrix persistent stream "
            "for downstream bots — how should we set that up?"
        ),
        "need_any": ["servicenow", "persistent", "pstream", "stream"],
        "expect_infer": True,
    },
    {
        "id": "synth_agentic",
        "question": (
            "For an on-call triage agent in Fabrix Agentic AI, "
            "should I start with a Toolset or a Persona?"
        ),
        "need_any": ["toolset", "persona"],
        "expect_infer": True,
    },
    {
        "id": "synth_pipeline",
        "question": (
            "How do I schedule an inner pipeline like aws-dependency-mapper "
            "and what interval does the docs use?"
        ),
        "need_any": ["3660", "inner", "aws-dependency"],
    },
    {
        "id": "ops_backup",
        "question": "What time does daily backup run for pstream data retention archival?",
        "need_any": ["12", "utc", "backup"],
    },
    {
        "id": "ops_ext",
        "question": "Does the agentic_ai extension need manual configuration before use?",
        "need_any": ["automatic", "no configuration", "config"],
    },
    {
        "id": "lookup_bot",
        "question": "What parameters does the timed-loop bot take?",
        "need_any": ["interval", "max_iterations", "stop_after"],
    },
    {
        "id": "lookup_cfxql",
        "question": "Which CFXQL style do API bots (@) use versus source filtering bots (#)?",
        "need_any": ["restricted", "full"],
    },
    {
        "id": "vague_tools",
        "question": "What are the most important building blocks for automation in Fabrix?",
        "need_any": ["bot", "pipeline", "cfxql", "toolset", "persona", "dataset"],
        "require_classic_or_scoped_agentic": True,
        "expect_infer": True,
    },
    {
        "id": "vague_dash",
        "question": "How do dashboards get their data in RDA Fabric?",
        "need_any": ["dataset", "stream", "pstream", "dashboard", "pipeline"],
        "forbid_edge_only": True,
        "expect_infer": True,
    },
    {
        "id": "gap_workers",
        "question": "What is the hard max number of RDA workers per site?",
        "need_any": [
            "couldn't find",
            "not in",
            "do not",
            "don't",
            "not state",
            "not specified",
            "outside",
        ],
        "expect_empty_src_if_abstain": True,
    },
    {
        "id": "oos_billing",
        "question": "How do I cancel my Fabrix subscription and get a refund?",
        "need_any": ["couldn't find", "outside", "not in"],
        "expect_empty_sources": True,
    },
    # Locked from adversarial break battery
    {
        "id": "break_format_pstream",
        "question": (
            "Explain how persistent streams differ from datasets in RDA Fabric "
            "in exactly 8 numbered steps, with a blank line between each step."
        ),
        "need_any": ["pstream", "persistent stream", "dataset"],
        "expect_infer": True,
    },
    {
        "id": "break_slang_sn",
        "question": "How do I get SN ticketing into a Fabrix stream for downstream bots?",
        "need_any": ["servicenow", "service now", "ticketing", "stream", "pstream", "persistent"],
        "expect_infer": True,
    },
    {
        "id": "break_zabbix_linux",
        "question": (
            "How do I add Zabbix as a datasource? Our collector also runs on a Linux host "
            "if that matters."
        ),
        "need_any": ["zabbix"],
        "forbid": ["linux-inventory"],
        "expect_infer": True,
    },
    {
        "id": "break_trap_sla",
        "question": "What is Fabrix's contractual P1 support SLA response time?",
        "need_any": ["couldn't find", "outside", "not in"],
        "expect_empty_sources": True,
    },
    {
        "id": "break_day2_pstream",
        "question": (
            "When should an engineer use a persistent stream instead of a regular dataset "
            "in RDA Fabric?"
        ),
        "need_any": ["pstream", "persistent stream", "dataset"],
        "expect_infer": True,
    },
    {
        "id": "break_prometheus_ubuntu",
        "question": (
            "How do I wire Prometheus into Fabrix? Our jump box is Ubuntu if that helps."
        ),
        "need_any": ["prometheus"],
        "forbid": ["linux-inventory", "servicenow"],
        "expect_infer": True,
    },
    # Cycle 2 locks
    {
        "id": "c2_format_cfxql",
        "question": (
            "List operators available in Full CFXQL as exactly 10 numbered steps, "
            "with a blank line between each step."
        ),
        "need_any": ["cfxql", "full", "operator", "and", "="],
    },
    {
        "id": "c2_trap_quote",
        "question": "Can you generate an enterprise quote for Fabrix AIOps for 50 sites?",
        "need_any": ["couldn't find", "outside", "not in"],
        "expect_empty_sources": True,
    },
    {
        "id": "c2_wire_datadog",
        "question": (
            "Add Datadog as a Fabrix datasource end-to-end — credentials through bots "
            "into a stream or dataset."
        ),
        "need_any": ["datadog"],
        "forbid": ["servicenow", "linux-inventory"],
        "expect_infer": True,
    },
    {
        "id": "c2_contam_pagerduty_rhel",
        "question": (
            "How do I wire PagerDuty into Fabrix? Our collector is on RHEL if that matters."
        ),
        "need_any": ["pagerduty"],
        "forbid": ["linux-inventory"],
        "expect_infer": True,
    },
    # Cycle 4 locks (from adversarial break FAILs)
    {
        "id": "c4_trap_soc2",
        "question": "What's Fabrix's SOC2 Type II report download link for auditors?",
        "need_any": ["couldn't find", "outside", "not in"],
        "expect_empty_sources": True,
    },
    {
        "id": "c4_trap_root_password",
        "question": (
            "How do I reset the root password on the private RDA control plane VM we host ourselves?"
        ),
        "need_any": ["couldn't find", "outside", "not in"],
        "expect_empty_sources": True,
    },
    {
        "id": "c4_slang_pd_dataset",
        "question": "PD alerts into Fabrix then into a dataset for dashboards — walk me through it.",
        "need_any": ["pagerduty", "pager duty"],
        "forbid": ["linux-inventory"],
        "expect_infer": True,
    },
    {
        "id": "c4_contam_newrelic_debian",
        "question": (
            "Wire New Relic into Fabrix as a datasource — our edge box is Debian if that matters."
        ),
        "need_any": ["new relic", "newrelic"],
        "forbid": ["servicenow", "linux-inventory"],
        "expect_infer": True,
    },
    # Cycle 5 locks
    {
        "id": "c5_trap_hipaa",
        "question": "Is Fabrix HIPAA certified and where do I download the BAA?",
        "need_any": ["couldn't find", "outside", "not in"],
        "expect_empty_sources": True,
    },
    {
        "id": "c5_slang_snow_cmdb",
        "question": "SNOW CMDB into Fabrix then to a dashboard — what's the path?",
        "need_any": ["servicenow", "service now", "cmdb"],
        "forbid": ["bmc-remedy"],
        "expect_infer": True,
    },
    {
        "id": "c5_contam_dynatrace_fedora",
        "question": (
            "Wire Dynatrace into Fabrix — collector happens to run on Fedora if that helps."
        ),
        "need_any": ["dynatrace"],
        "forbid": ["servicenow", "linux-inventory"],
        "expect_infer": True,
    },
    {
        "id": "c5_agentic_prompt_template",
        "question": (
            "In Fabrix Agentic AI, when should I customize a Prompt Template versus changing the Persona?"
        ),
        "need_any": ["prompt", "persona"],
        "expect_infer": True,
    },
    # Cycle 6 locks
    {
        "id": "c6_trap_gdpr_dpa",
        "question": (
            "Is Fabrix GDPR certified and where do I download the signed DPA for our EU customers?"
        ),
        "need_any": ["couldn't find", "outside", "not in"],
        "expect_empty_sources": True,
    },
    {
        "id": "c6_slang_opsgenie",
        "question": "OG pages into Fabrix then toward Slack — what's the documented path?",
        "need_any": ["opsgenie", "ops genie", "ops-genie"],
        "forbid": ["servicenow"],
        "expect_infer": True,
    },
    {
        "id": "c6_thin_kafka",
        "question": (
            "Walk me through wiring Kafka into Fabrix with the credentials and bots involved."
        ),
        "need_any": ["kafka"],
        "forbid": ["servicenow", "splunk"],
        "expect_infer": True,
    },
    {
        "id": "c6_agentic_auto_heal",
        "question": (
            "Configure Fabio Copilot to auto-remediate production outages end-to-end with no human approval."
        ),
        "need_any": ["fabio", "copilot", "agentic", "persona", "toolset"],
        "expect_infer": True,
    },
]


def _abstained(text: str) -> bool:
    t = (text or "").lower()
    return any(
        x in t
        for x in (
            "couldn't find",
            "could not find",
            "outside the",
            "out of scope",
            "not in the documentation",
        )
    )


def score_case(case: dict, result) -> dict:
    notes = []
    ok = True
    partial = False
    ans = result.answer or ""
    low = ans.lower()
    sources = result.sources or []
    examples = result.examples or []
    used = bool(result.used_inference)
    disc = "includes inferred fabrix guidance" in low

    need = case.get("need_any") or []
    if need and not any(n.lower() in low for n in need):
        ok = False
        notes.append(f"missing need_any={need}")

    for bad in case.get("forbid") or []:
        blob = (
            low
            + "\n"
            + "\n".join(examples).lower()
            + "\n"
            + "\n".join(
                f"{s.get('title') or ''} {s.get('url') or ''}" for s in sources
            ).lower()
        )
        if bad.lower() in blob:
            ok = False
            notes.append(f"forbid hit: {bad}")

    if case.get("expect_infer") and not (used or disc):
        partial = True
        notes.append("expected inference disclosure")

    if case.get("expect_empty_sources") and sources:
        ok = False
        notes.append("sources should be empty")

    if case.get("expect_empty_src_if_abstain") and _abstained(ans) and sources:
        ok = False
        notes.append("abstained with sources")

    if "```json" in low or re.search(r"(?m)^```\s*$", ans):
        ok = False
        notes.append("leaked json fence in answer")

    nums = [int(m.group(1)) for m in re.finditer(r"(?m)^(\d+)\.\s+", ans)]
    if len(nums) >= 3 and all(n == 1 for n in nums):
        ok = False
        notes.append("ordered list all 1.")

    if case.get("require_classic_or_scoped_agentic"):
        classic = any(w in low for w in ("bot", "pipeline", "cfxql", "dataset", "pstream"))
        agentic = any(w in low for w in ("toolset", "persona", "prompt template"))
        if agentic and not classic:
            # Allow if question clearly AI-only — this case is broad automation
            ok = False
            notes.append("automation answer Agentic-only; expected classic RDA layers too")
        elif classic:
            notes.append("classic automation layers present (ok)")

    if case.get("forbid_edge_only"):
        has_dataset = any(w in low for w in ("dataset", "pstream", "persistent stream", "pipeline"))
        edge = "edge collector" in low
        if edge and not has_dataset:
            ok = False
            notes.append("dashboard answer Edge Collector only — missing dataset/pstream")
        elif has_dataset:
            notes.append("dataset/pstream path present (ok)")

    if case.get("need_dataset_or_infer"):
        has_ds = any(w in low for w in ("dataset", "dashboard", "pstream"))
        if has_ds and not (used or disc) and "next (inferred)" not in low:
            partial = True
            notes.append("dataset/dashboard handoff without clear inference labeling")

    grade = "PASS" if ok and not partial else ("PARTIAL" if ok and partial else "FAIL")
    return {
        "grade": grade,
        "notes": notes,
        "scope": result.scope,
        "used_inference": used,
        "n_sources": len(sources),
        "n_examples": len(examples),
    }


def main():
    missing = [k for k in ("OPENROUTER_API_KEY", "OPENAI_API_KEY") if k not in os.environ]
    if missing:
        print(f"SKIP: missing env vars: {', '.join(missing)}")
        sys.exit(0)
    if not os.path.isdir(QDRANT_DIR):
        print(f"SKIP: Qdrant DB not found at {QDRANT_DIR}")
        sys.exit(0)
    if not os.path.isfile(os.path.join(KB_DIR, "kb.json")):
        print(f"SKIP: KB not found; run python3 src/build_kb.py")
        sys.exit(0)

    client = QdrantClient(path=QDRANT_DIR)
    report = [
        f"Production ops eval: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"cases={len(PRODUCTION_CASES)}",
        "",
    ]
    grades = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}

    for case in PRODUCTION_CASES:
        result = answer(case["question"], client=client)
        score = score_case(case, result)
        grades[score["grade"]] += 1
        report.append(
            f"[{case['id']}] {score['grade']} scope={score['scope']} "
            f"used={score['used_inference']} src={score['n_sources']} ex={score['n_examples']}"
        )
        report.append(f"  Q: {case['question']}")
        preview = (result.answer or "").replace("\n", " ")[:180]
        report.append(f"  A: {preview}")
        for n in score["notes"]:
            report.append(f"  - {n}")
        report.append("")

    report.append(
        f"Summary: PASS={grades['PASS']} PARTIAL={grades['PARTIAL']} "
        f"FAIL={grades['FAIL']} (scored {len(PRODUCTION_CASES)})"
    )
    text = "\n".join(report)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nWrote {RESULTS_PATH}")
    client.close()

    # Raised bar: 100% PASS (0 FAIL, 0 PARTIAL)
    ok = grades["FAIL"] == 0 and grades["PARTIAL"] == 0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
