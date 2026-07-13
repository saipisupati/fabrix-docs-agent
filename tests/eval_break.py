"""
eval_break.py: adversarial battery meant to break the Fabrix docs agent.

Run: python3 tests/eval_break.py
Writes tests/eval_break_results.txt (gitignored).
Stop the API first (local Qdrant file lock).
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qdrant_client import QdrantClient

from agent import answer
from config import KB_DIR, QDRANT_DIR

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval_break_results.txt")

# Never-before-used questions (not in eval_set / eval_production)
BREAK_CASES = [
    {
        "id": "facet_monitoring",
        "attack": "wrong_facet",
        "question": "How should we design our monitoring stack in Fabrix for NOC visibility?",
        "need_any": ["dashboard", "alert", "stream", "pipeline", "dataset", "pstream", "zabbix", "integration"],
        "forbid_agentic_only": True,
    },
    {
        "id": "facet_visualize",
        "attack": "wrong_facet",
        "question": "How do we visualize ops data for executives in RDA Fabric?",
        "need_any": ["dashboard", "dataset", "persona"],
        "require_dataset_or_dashboard": True,
    },
    {
        "id": "contam_zabbix_linux",
        "attack": "contamination",
        "question": (
            "How do I add Zabbix as a datasource? Our collector also runs on a Linux host "
            "if that matters."
        ),
        "need_any": ["zabbix"],
        "forbid_in_examples": ["linux-inventory"],
        "forbid_in_sources": ["linux-inventory", "servicenow", "prometheus"],
        "prefer": ["zabbix"],
        "expect_infer_or_gaps": True,
    },
    {
        "id": "contam_prometheus_ubuntu",
        "attack": "contamination",
        "question": (
            "How do I wire Prometheus into Fabrix? Our jump box is Ubuntu if that helps."
        ),
        "need_any": ["prometheus"],
        "forbid_in_sources": ["servicenow", "zabbix", "linux-inventory"],
        "forbid_in_examples": ["linux-inventory", "servicenow"],
        "forbid_source_titles": ["aws"],
        "prefer": ["prometheus"],
        "expect_infer_or_gaps": True,
    },
    {
        "id": "contam_splunk_linux",
        "attack": "contamination",
        "question": (
            "How do I add Splunk as a datasource in Fabrix? Collector runs on a Linux host."
        ),
        "need_any": ["splunk"],
        "forbid_in_sources": ["servicenow", "prometheus", "zabbix", "linux-inventory"],
        "forbid_in_examples": ["linux-inventory"],
        "prefer": ["splunk"],
        "expect_infer_or_gaps": True,
    },
    {
        "id": "overclaim_runbook",
        "attack": "overclaim",
        "question": (
            "Give me a complete production NOC runbook for standing up Fabrix from zero "
            "through 24x7 on-call, including pager rotations and escalation SLAs."
        ),
        "expect_gaps_or_infer": True,
        "forbid_undisclosed_sla": True,
    },
    {
        "id": "slang_sn_stream",
        "attack": "partial_name",
        "question": "How do I get SN ticketing into a Fabrix stream for downstream bots?",
        "need_any": ["servicenow", "service now", "ticketing", "stream", "pstream", "persistent"],
    },
    {
        "id": "slang_k8s_dash",
        "attack": "partial_name",
        "question": "k8s inventory for dashboards — what's the Fabrix path?",
        "need_any": ["kubernetes", "kubectl", "kubernetes-inventory"],
        "forbid": ["linux-inventory"],
    },
    {
        "id": "multi_vmware_slack",
        "attack": "multi_intent",
        "question": (
            "Compare VMware vCenter vs vRealize Operations integration in Fabrix, "
            "and how would I wire inventory alerts toward Slack?"
        ),
        "need_any": ["vcenter", "vrealize", "vmware"],
        "expect_infer": True,
    },
    {
        "id": "format_eight_steps",
        "attack": "format_stress",
        "question": (
            "Explain how persistent streams differ from datasets in RDA Fabric "
            "in exactly 8 numbered steps, with a blank line between each step."
        ),
        "need_any": ["pstream", "persistent stream", "dataset"],
        "check_list_numbering": True,
        "forbid_json_leak": True,
    },
    {
        "id": "trap_sla",
        "attack": "trap_abstain",
        "question": "What is Fabrix's contractual P1 support SLA response time?",
        "expect_abstain": True,
        "expect_empty_sources": True,
    },
    {
        "id": "trap_vpn_keys",
        "attack": "trap_abstain",
        "question": (
            "How do I rotate encryption keys on our private VPN jump host "
            "that sits in front of RDA?"
        ),
        "expect_abstain": True,
        "expect_empty_sources": True,
    },
    {
        "id": "day2_stuck",
        "attack": "day2_ops",
        "question": "How do I evict a stuck pipeline run in RDA Fabric without restarting the site?",
        "need_any": ["evict", "stuck", "pipeline"],
    },
    {
        "id": "day2_pstream_vs_dataset",
        "attack": "day2_ops",
        "question": (
            "When should an engineer use a persistent stream instead of a regular dataset "
            "in RDA Fabric?"
        ),
        "need_any": ["pstream", "persistent stream", "dataset"],
    },
    {
        "id": "day2_ad_sso",
        "attack": "day2_ops",
        "question": "How do we configure AD SSO / external user authentication for RDA Fabric?",
        "need_any": ["sso", "ad", "active directory", "authentication", "ldap", "external"],
    },
    {
        "id": "ebond_pagerduty",
        "attack": "multi_intent",
        "question": (
            "What does the ebonding-stream-to-pagerduty pipeline do, and what would I "
            "still need to configure that the docs might not spell out end-to-end?"
        ),
        "need_any": ["pagerduty", "ebond"],
        "expect_infer_or_gaps": True,
    },
    # --- Cycle 2: push remaining soft spots ---
    {
        "id": "c2_wire_datadog",
        "attack": "thin_wiring",
        "cycle": 2,
        "question": (
            "Add Datadog as a Fabrix datasource end-to-end — credentials through bots "
            "into a stream or dataset."
        ),
        "need_any": ["datadog"],
        "forbid_in_sources": ["servicenow", "prometheus", "zabbix", "linux-inventory"],
        "require_wiring_shape": True,
        "expect_infer_or_gaps": True,
    },
    {
        "id": "c2_multi_snow_slack",
        "attack": "multi_intent",
        "cycle": 2,
        "question": (
            "Compare ServiceNow CMDB vs Ticketing modules in Fabrix, then how would I "
            "page Slack from resulting alerts?"
        ),
        "need_any": ["servicenow", "cmdb", "ticket"],
        "need_also_any": ["slack"],
        "forbid": ["bmc-remedy", "bmc_remedy", "@bmc"],
        "forbid_in_sources": ["bmc-remedy", "bmc_remedy"],
        "forbid_in_examples": ["bmc-remedy", "bmc_remedy"],
        "expect_infer": True,
        "require_next_inferred": True,
    },
    {
        "id": "c2_format_cfxql",
        "attack": "format_stress",
        "cycle": 2,
        "question": (
            "List operators available in Full CFXQL as exactly 10 numbered steps, "
            "with a blank line between each step."
        ),
        "need_any": ["cfxql", "full", "operator", "=", "and", "or"],
        "check_list_numbering": True,
        "forbid_json_leak": True,
    },
    {
        "id": "c2_contam_pagerduty_rhel",
        "attack": "contamination",
        "cycle": 2,
        "question": (
            "How do I wire PagerDuty into Fabrix? Our collector is on RHEL if that matters."
        ),
        "need_any": ["pagerduty"],
        "forbid_in_sources": ["servicenow", "prometheus", "zabbix", "linux-inventory", "splunk"],
        "forbid_in_examples": ["linux-inventory"],
        "prefer": ["pagerduty"],
        "expect_infer_or_gaps": True,
    },
    {
        "id": "c2_overclaim_cost",
        "attack": "overclaim",
        "cycle": 2,
        "question": (
            "Give me a cost model for running 200 RDA workers across 3 regions "
            "including list price and discounts."
        ),
        "expect_abstain": True,
        "expect_empty_sources": True,
    },
    {
        "id": "c2_trap_quote",
        "attack": "trap_abstain",
        "cycle": 2,
        "question": "Can you generate an enterprise quote for Fabrix AIOps for 50 sites?",
        "expect_abstain": True,
        "expect_empty_sources": True,
    },
    {
        "id": "c2_trap_mtls",
        "attack": "trap_abstain",
        "cycle": 2,
        "question": (
            "How do I rotate mTLS certificates on the private ingress in front of RDA Fabric?"
        ),
        "expect_abstain": True,
        "expect_empty_sources": True,
    },
    {
        "id": "c2_day2_clone_bp",
        "attack": "day2_ops",
        "cycle": 2,
        "question": (
            "How do I clone a service blueprint to another RDA site without re-entering secrets?"
        ),
        "need_any": ["blueprint", "clone", "copy", "export", "import", "site", "secret", "credential"],
        "expect_infer_or_gaps": True,
    },
    {
        "id": "c2_fabio_copilot",
        "attack": "agentic_boundary",
        "cycle": 2,
        "question": (
            "Can Fabio Copilot change production pipelines for me, and what audit trail exists?"
        ),
        "need_any": ["fabio", "copilot", "pipeline", "audit", "change", "edit", "modify"],
        "forbid_in_sources": ["ucsm", "cisco ucs"],
        "expect_infer_or_gaps": True,
    },
    {
        "id": "c2_multi_datadog_dash",
        "attack": "multi_intent",
        "cycle": 2,
        "question": (
            "Wire Datadog metrics into Fabrix and also explain how those land on a dashboard."
        ),
        "need_any": ["datadog"],
        "need_also_any": ["dashboard", "dataset", "pstream", "stream"],
        "forbid_in_sources": ["servicenow", "zabbix"],
        "expect_infer": True,
    },
    {
        "id": "c2_thin_nagios",
        "attack": "thin_wiring",
        "cycle": 2,
        "question": (
            "Walk me through adding Nagios XI as a Fabrix datasource with the exact bots involved."
        ),
        "need_any": ["nagios"],
        "require_wiring_shape": True,
        "forbid_in_sources": ["servicenow", "prometheus"],
    },
    {
        "id": "c2_agentic_toolset_persona",
        "attack": "synthesis_compare",
        "cycle": 2,
        "question": (
            "When building Agentic AI on Fabrix, should I start with a Toolset or a Persona "
            "for a read-only troubleshooting assistant?"
        ),
        "need_any": ["toolset", "persona"],
        "forbid_in_sources": ["servicenow", "kubernetes", "prometheus"],
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
            "not covered",
        )
    )


def score_break_case(case: dict, result) -> dict:
    notes = []
    tags = []
    ok = True
    partial = False
    ans = result.answer or ""
    low = ans.lower()
    sources = result.sources or []
    examples = result.examples or []
    gaps = result.gaps or []
    used = bool(result.used_inference)
    disc = "includes inferred fabrix guidance" in low
    ex_blob = "\n".join(examples).lower()

    need = case.get("need_any") or []
    if need and not any(n.lower() in low for n in need):
        ok = False
        tags.append("wrong_facet")
        notes.append(f"missing need_any={need}")

    need2 = case.get("need_also_any") or []
    if need2 and not any(n.lower() in low for n in need2):
        ok = False
        tags.append("wrong_facet")
        notes.append(f"missing need_also_any={need2}")

    if case.get("require_wiring_shape"):
        has_cred = any(
            w in low
            for w in (
                "credential", "username", "password", "token", "api key", "auth",
                "api access", "api integration", "read access", "service user",
            )
        )
        has_bot = "bot" in low or "@" in ans or "*" in ans or "nagios" in low
        has_sink = any(
            w in low
            for w in ("stream", "pstream", "dataset", "pipeline", "dashboard", "datasource")
        )
        if not (has_cred and has_bot and has_sink):
            partial = True
            tags.append("thin_wiring")
            notes.append(
                f"wiring shape weak cred={has_cred} bot={has_bot} sink={has_sink}"
            )

    if case.get("require_next_inferred") and "next (inferred)" not in low and not disc:
        partial = True
        tags.append("overclaim")
        notes.append("multi-intent missing Next (inferred)")

    for bad in case.get("forbid") or []:
        if bad.lower() in low or bad.lower() in ex_blob:
            ok = False
            tags.append("contamination")
            notes.append(f"forbid hit: {bad}")

    for bad in case.get("forbid_in_examples") or []:
        if bad.lower() in ex_blob:
            ok = False
            tags.append("contamination")
            notes.append(f"example contamination: {bad}")

    src_blob = "\n".join(
        f"{s.get('title') or ''} {s.get('url') or ''} {s.get('excerpt') or ''}"
        for s in sources
    ).lower()
    for bad in case.get("forbid_in_sources") or []:
        if bad.lower() in src_blob:
            ok = False
            tags.append("contamination")
            notes.append(f"source contamination: {bad}")
    for bad in case.get("forbid_source_titles") or []:
        titles = " ".join((s.get("title") or "").lower() for s in sources)
        # whole-token-ish: title equals or contains as product name
        if re.search(rf"(^|[^a-z]){re.escape(bad.lower())}([^a-z]|$)", titles):
            ok = False
            tags.append("contamination")
            notes.append(f"source title contamination: {bad}")

    if case.get("forbid_agentic_only"):
        agentic = any(w in low for w in ("toolset", "persona", "prompt template", "agentic"))
        classic = any(
            w in low
            for w in ("dashboard", "alert", "pipeline", "dataset", "pstream", "bot", "integration")
        )
        if agentic and not classic:
            ok = False
            tags.append("wrong_facet")
            notes.append("monitoring answer Agentic-only")

    if case.get("require_dataset_or_dashboard"):
        if not any(w in low for w in ("dashboard", "dataset")):
            ok = False
            tags.append("wrong_facet")
            notes.append("missing dashboard/dataset")

    if case.get("expect_abstain"):
        if not _abstained(ans):
            ok = False
            tags.append("abstain_fail")
            notes.append("expected abstain on trap question")
        if case.get("expect_empty_sources") and sources:
            ok = False
            tags.append("empty_sources_leak")
            notes.append("trap answered with sources")

    if case.get("forbid_undisclosed_sla"):
        sla_talk = any(w in low for w in ("sla", "15 minute", "1 hour", "p1 response"))
        if sla_talk and not (used or disc or _abstained(ans) or gaps):
            ok = False
            tags.append("overclaim")
            notes.append("SLA-like claims without gaps/infer/abstain")

    if case.get("expect_gaps_or_infer"):
        if not (gaps or used or disc or _abstained(ans) or "next (inferred)" in low):
            partial = True
            tags.append("overclaim")
            notes.append("full runbook ask without gaps/infer/abstain")

    if case.get("expect_infer") and not (used or disc or "next (inferred)" in low):
        partial = True
        tags.append("overclaim")
        notes.append("expected inference labeling")

    if case.get("expect_infer_or_gaps") and not (used or disc or gaps or "next (inferred)" in low):
        partial = True
        tags.append("overclaim")
        notes.append("expected gaps or inferred handoff")

    if "```json" in low or re.search(r"(?m)^```\s*$", ans):
        ok = False
        tags.append("json_leak")
        notes.append("leaked json fence")

    if case.get("check_list_numbering"):
        nums = [int(m.group(1)) for m in re.finditer(r"(?m)^(\d+)\.\s+", ans)]
        if len(nums) >= 3 and all(n == 1 for n in nums):
            ok = False
            tags.append("list_numbering")
            notes.append("ordered list all 1.")
        elif len(nums) >= 2 and nums.count(1) >= 2 and max(nums) > 1:
            # restart mid-list e.g. 1,2,1 or 1,1,2 after polish should be gone
            partial = True
            tags.append("list_numbering")
            notes.append(f"list restart nums={nums[:12]}")
        elif len(nums) >= 3:
            notes.append(f"list nums={nums[:10]}")
        if re.search(r"(?m)^\d+\)\s+", ans):
            partial = True
            tags.append("list_numbering")
            notes.append("paren-style 1) list markers")
        if re.search(r"(?m)^\d+\.\s+\S[^\n]*[ \t]+\d+\.\s+", ans):
            ok = False
            tags.append("list_numbering")
            notes.append("same-line nested numbering")

    # Prefer Zabbix over Linux for zabbix questions
    if case.get("prefer"):
        pref = case["prefer"][0].lower()
        if pref not in low and pref not in ex_blob:
            partial = True
            tags.append("wrong_facet")
            notes.append(f"preferred topic weak: {pref}")

    grade = "PASS" if ok and not partial else ("PARTIAL" if ok else "FAIL")
    return {
        "grade": grade,
        "tags": sorted(set(tags)),
        "notes": notes,
        "scope": result.scope,
        "used_inference": used,
        "n_sources": len(sources),
        "n_examples": len(examples),
        "n_gaps": len(gaps),
        "attack": case.get("attack"),
    }


def main():
    missing = [k for k in ("OPENROUTER_API_KEY", "OPENAI_API_KEY") if k not in os.environ]
    if missing:
        print(f"SKIP: missing env vars: {', '.join(missing)}")
        sys.exit(0)
    if not os.path.isdir(QDRANT_DIR):
        print(f"SKIP: Qdrant DB not found")
        sys.exit(0)
    if not os.path.isfile(os.path.join(KB_DIR, "kb.json")):
        print("SKIP: KB missing; run build_kb.py")
        sys.exit(0)

    cycle = os.environ.get("BREAK_CYCLE", "").strip()
    cases = BREAK_CASES
    if cycle == "2":
        cases = [c for c in BREAK_CASES if c.get("cycle") == 2]
    elif cycle == "1":
        cases = [c for c in BREAK_CASES if c.get("cycle") != 2]

    client = QdrantClient(path=QDRANT_DIR)
    report = [
        f"Break battery: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"cases={len(cases)}" + (f" cycle={cycle}" if cycle else ""),
        "",
    ]
    grades = Counter()
    tag_counts = Counter()

    for case in cases:
        result = answer(case["question"], client=client)
        score = score_break_case(case, result)
        grades[score["grade"]] += 1
        for t in score["tags"]:
            tag_counts[t] += 1
        report.append(
            f"[{case['id']}] ({case['attack']}) {score['grade']} "
            f"scope={score['scope']} used={score['used_inference']} "
            f"src={score['n_sources']} tags={score['tags']}"
        )
        report.append(f"  Q: {case['question']}")
        report.append(f"  A: {(result.answer or '').replace(chr(10), ' ')[:200]}")
        for n in score["notes"]:
            report.append(f"  - {n}")
        report.append("")

    report.append(
        f"Summary: PASS={grades['PASS']} PARTIAL={grades['PARTIAL']} "
        f"FAIL={grades['FAIL']} (scored {len(cases)})"
    )
    report.append(f"Tag counts: {dict(tag_counts)}")
    text = "\n".join(report)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nWrote {RESULTS_PATH}")
    client.close()

    n = len(cases) or 1
    pass_rate = grades["PASS"] / n
    # Raised bar: ≥95% PASS and 0 FAIL (PARTIAL allowed)
    ok = grades["FAIL"] == 0 and pass_rate >= 0.95
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
