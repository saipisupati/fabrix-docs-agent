"""
eval_set.py, hand-built Q&A cases with expected facts and sources.

Cases covering bots, CFXQL, guides, integrations, negative/hallucination,
plus KB scope / inference / examples / gaps.
Used by all run_eval_*.py scripts.
"""

FILTER_BY_CATEGORY = {
    "lookup": {"type": "bot"},
    "multi_part": {"type": "bot"},
    "guide": {"type": "narrative", "doc_section": "beginners_guide"},
    "install": {"type": "narrative", "doc_section": "installation_guides"},
    "ai_fabric": {"type": "narrative", "doc_section": "ai_fabric"},
    "pipeline": {"type": "narrative", "doc_section": "Pipelines"},
    "datasource": {"type": "narrative", "doc_section": "Datasource_Integrations"},
    "extensions": {"type": "narrative", "doc_section": "Extensions"},
    "releases": {"type": "narrative", "doc_section": "rda_releases"},
}

TOP_K_BY_CATEGORY = {
    "lookup": 5,
    "comparison": 10,
    "multi_part": 8,
    "negative": 5,
    "guide": 5,
    "install": 5,
    "ai_fabric": 5,
    "pipeline": 5,
    "datasource": 5,
    "extensions": 5,
    "releases": 5,
}
DEFAULT_TOP_K = 5


def retrieval_params(case):
    """top_k and filter_dict for a case (category defaults + per-case override)."""
    return {
        "top_k": TOP_K_BY_CATEGORY.get(case["category"], DEFAULT_TOP_K),
        "filter_dict": case.get("filter_dict") or FILTER_BY_CATEGORY.get(case["category"]),
    }


EVAL_SET = [
    {
        "id": "lookup_01",
        "question": "What parameters does the count loop bot take?",
        "category": "lookup",
        "expected_facts": [
            "name", "start", "end", "increment",
            "start defaults to 0", "increment must be a positive integer >= 1",
        ],
        "expected_source": "@c:count-loop",
        "notes": "Baseline easy case - already confirmed working correctly.",
    },
    {
        "id": "lookup_02",
        "question": "What does the data loop bot use to load a saved dataset?",
        "category": "lookup",
        "expected_facts": [
            "dataset parameter", "columns parameter",
            "comma separated list of columns",
        ],
        "expected_source": "@c:data-loop",
        "notes": "Second lookup case, different bot.",
    },
    {
        "id": "comparison_01",
        "question": "What is the difference between Full and Restricted CFXQL?",
        "category": "comparison",
        "expected_facts": [
            "Full CFXQL supports many operators (is, not in, !=, AND, OR, etc.)",
            "Full CFXQL has a Result Format / GET clause",
            "Restricted CFXQL only supports = and AND",
            "Restricted CFXQL does NOT support Result Format",
        ],
        "expected_source": "cfxql_reference (Full + Restricted sections)",
        "notes": "FAILED with size-based chunking, FIXED with section-aware chunking. Key regression test.",
    },
    {
        "id": "comparison_02",
        "question": "Which CFXQL type does a source filtering bot use, and how is that different from what an API bot uses?",
        "category": "comparison",
        "expected_facts": [
            "Source filtering bots (#) use Full CFXQL",
            "API bots (@) use Restricted CFXQL",
            "Full CFXQL is translated into a query/API call sent to the remote data source",
            "Restricted CFXQL extracts API parameters that control bot behavior",
        ],
        "expected_source": "cfxql_reference (bot types table + Full/Restricted sections)",
        "notes": "Harder version of comparison_01.",
    },
    {
        "id": "multi_part_01",
        "question": "What does the if-condition bot do, and what kind of CFXQL does it expect?",
        "category": "multi_part",
        "expected_facts": [
            "Runs subsequent bots only if the CFXQL query matches",
            "end-if bot must be called to exit the block",
            "Expects Full CFXQL",
            "Applies the query on data already loaded from a previous bot or source",
        ],
        "expected_source": "*exec:if-condition",
        "notes": "Two sub-questions in one ask.",
    },
    {
        "id": "guide_01",
        "question": "What endpoint types does the RDA Event Gateway support for data ingestion?",
        "category": "guide",
        "expected_facts": [
            "syslog_tcp",
            "syslog_udp",
            "http",
            "tcp_json",
            "filebeat",
            "file",
        ],
        "expected_source": "beginners_guide/data_ingestion.md",
        "notes": "Narrative eval: Event Gateway endpoint table.",
    },
    {
        "id": "guide_02",
        "question": "What is a primary design principle of RDA Fabric architecture?",
        "category": "guide",
        "expected_facts": [
            "perform data operations close to data source",
        ],
        "expected_source": "beginners_guide/architecture.md",
        "notes": "Narrative eval: architecture design principle.",
    },
    {
        "id": "guide_03",
        "question": "What messaging services does RDAF use for streaming data by default?",
        "category": "guide",
        "expected_facts": [
            "NATS",
            "Kafka",
        ],
        "expected_source": "beginners_guide/persistent_streams.md",
        "notes": "Narrative eval: persistent streams / messaging.",
    },
    {
        "id": "install_01",
        "question": "What time is daily backup configured to run for pstream data retention archival?",
        "category": "install",
        "expected_facts": [
            "12AM UTC",
            "Daily Backup",
        ],
        "expected_source": "installation_guides/data_retention.md",
        "notes": "Narrative eval: data retention backup schedule.",
    },
    {
        "id": "ai_01",
        "question": "What is LLM pooling used for in Agentic AI?",
        "category": "ai_fabric",
        "expected_facts": [
            "load balancing",
            "cost optimization",
            "multiple LLM endpoints",
        ],
        "expected_source": "ai_fabric/ai_administration/llm_pooling.md",
        "notes": "Narrative eval: LLM pool purpose in Agentic AI.",
    },
    {
        "id": "negative_01",
        "question": "How do I cancel my Fabrix.ai subscription?",
        "category": "negative",
        "expected_facts": [
            "Should explicitly state it could not find this in the documentation",
            "Should NOT invent a plausible-sounding but fabricated answer",
        ],
        "expected_source": "none - billing/subscription not covered in ingested docs",
        "notes": "Tests hallucination resistance; replaced after corpus expansion made password-reset question answerable.",
    },
    {
        "id": "negative_02",
        "question": "What is the maximum number of workers a single RDA site can have?",
        "category": "negative",
        "expected_facts": [
            "Should state it could not find this in the documentation",
            "Should NOT invent a specific number",
        ],
        "expected_source": "none - architecture.md is ingested but does not state a worker limit",
        "notes": "More tempting hallucination case since the topic is real.",
    },
    {
        "id": "lookup_03",
        "question": "What parameters does the timed-loop bot take?",
        "category": "lookup",
        "expected_facts": [
            "interval",
            "seconds between each loop iteration",
            "max_iterations",
            "stop_after",
        ],
        "expected_source": "@c:timed-loop",
        "notes": "Bot sample: control catalog, different from count-loop.",
    },
    {
        "id": "lookup_04",
        "question": "What parameters does the post-message-to-channel bot take?",
        "category": "lookup",
        "expected_facts": [
            "channel_col",
            "message_col",
            "as_user",
        ],
        "expected_source": "@slack:post-message-to-channel",
        "notes": "Bot sample: slack integration bot.",
    },
    {
        "id": "lookup_05",
        "question": "What does the end-if bot do?",
        "category": "lookup",
        "expected_facts": [
            "if-condition block",
            "returns the dataset",
        ],
        "expected_source": "@exec:end-if",
        "notes": "Bot sample: exec catalog, unique slug (run-pipeline is ambiguous).",
    },
    {
        "id": "pipeline_01",
        "question": "What inner pipeline does the aws dependency mapper pipeline run, and on what interval?",
        "category": "pipeline",
        "expected_facts": [
            "aws-dependency-mapper-inner-pipeline",
            "3660",
        ],
        "expected_source": "Pipelines/aws-dependency-mapper.md",
        "notes": "Narrative eval: pipeline code block (bot badges are HTML, not chunk text).",
    },
    {
        "id": "datasource_01",
        "question": "What ServiceNow modules does Fabrix AIOps integrate with?",
        "category": "datasource",
        "expected_facts": [
            "Ticketing",
            "CMDB",
        ],
        "expected_source": "Datasource_Integrations/servicenow.md",
        "notes": "Narrative eval: ServiceNow integration scope.",
    },
    {
        "id": "datasource_02",
        "question": "How can RDA collect Kubernetes cluster inventory?",
        "category": "datasource",
        "expected_facts": [
            "kubectl",
            "SSH",
            "HTTP API",
        ],
        "expected_source": "Datasource_Integrations/kubernetes.md",
        "notes": "Narrative eval: Kubernetes collection methods.",
    },
    {
        "id": "extensions_01",
        "question": "Does the agentic_ai extension require configuration?",
        "category": "extensions",
        "expected_facts": [
            "automatically initialized",
            "no configuration",
        ],
        "expected_source": "Extensions/extensions_A_B.md",
        "notes": "Narrative eval: extension list entry.",
    },
    {
        "id": "releases_01",
        "question": "What does cfxOIA stand for?",
        "category": "releases",
        "expected_facts": [
            "Ops Intelligence",
            "Analytics",
        ],
        "expected_source": "rda_releases/cfx_rda_oia.md",
        "notes": "Narrative eval: OIA release notes intro.",
    },
    # --- KB / scope / inference cases (structured KB + 1B labeling) ---
    {
        "id": "scope_01",
        "question": "How do I bake a chocolate cake?",
        "category": "scope",
        "expected_facts": [
            "Should classify as out of scope or abstain",
            "sources must be empty",
        ],
        "expected_source": "none",
        "expect_empty_sources": True,
        "expect_scope": "out_of_scope",
        "notes": "Out-of-scope: empty sources, no fake citations.",
    },
    {
        "id": "scope_02",
        "question": "How do I cancel my Fabrix.ai subscription?",
        "category": "scope",
        "expected_facts": [
            "Should abstain or say not in documentation",
            "sources must be empty when abstaining",
        ],
        "expected_source": "none",
        "expect_empty_sources": True,
        "notes": "Billing not in public docs; empty sources on abstain.",
    },
    {
        "id": "inference_01",
        "question": "After installing RDA Fabric, what Fabrix components and configs should I verify before running production pipelines?",
        "category": "inference",
        "expected_facts": [
            "May include install/pipeline verification from docs",
            "Inferred technical synthesis must be labeled if present",
        ],
        "expected_source": "installation_guides / pipelines (related)",
        "expect_scope_in": ["related", "in_scope"],
        "expect_inferred": True,
        "notes": "Fabrix technical verification: require used_inference / disclosure.",
    },
    {
        "id": "inference_02",
        "question": "How would I build a Fabrix pipeline that pulls ServiceNow ticket data and writes it to a persistent stream? Which bot types and CFXQL style should I use?",
        "category": "inference",
        "expected_facts": [
            "ServiceNow",
            "persistent stream or pstream",
            "Inferred composition labeled",
        ],
        "expected_source": "Datasource_Integrations / Pipelines / CFXQL (synthesis)",
        "expect_scope_in": ["related", "in_scope"],
        "expect_inferred": True,
        "notes": "Multi-doc pipeline composition; require used_inference disclosure.",
    },
    {
        "id": "inference_03",
        "question": "In Agentic AI on Fabrix, when should I use a Toolset versus a Persona for an ops automation task?",
        "category": "inference",
        "expected_facts": [
            "Toolset",
            "Persona",
            "Inferred guidance labeled",
        ],
        "expected_source": "ai_fabric (synthesis)",
        "expect_scope_in": ["related", "in_scope"],
        "expect_inferred": True,
        "notes": "Agentic tools synthesis; require used_inference disclosure.",
    },
    {
        "id": "inference_04",
        "question": "How would I chain a Kubernetes inventory collection into a dashboard-friendly dataset?",
        "category": "inference",
        "expected_facts": [
            "kubernetes-inventory or kubectl/SSH/HTTP inventory collection",
        ],
        "expected_source": "Datasource_Integrations/kubernetes.md",
        "expect_scope_in": ["related", "in_scope"],
        "expect_inferred": True,
        "expect_any_substrings": [
            "kubernetes-inventory",
            "kubectl",
            "ssh",
            "http api",
        ],
        "forbid_substrings": [
            "linux-inventory",
        ],
        "forbid_undisclosed_claims": [
            "automatically formatted into a suitable dataset",
            "automatically formatted for dashboards",
            "will be automatically formatted",
        ],
        "max_sudoers_mentions": 2,
        "expect_ordered_list_renumbered": True,
        "notes": "Path-first K8s→dashboard; no linux-inventory; no all-1. lists; sudoers not repeated endlessly.",
    },
    {
        "id": "examples_01",
        "question": "Show an example of Full CFXQL versus Restricted CFXQL.",
        "category": "examples",
        "expected_facts": [
            "Full CFXQL",
            "Restricted CFXQL",
        ],
        "expected_source": "cfxql_reference",
        "expect_examples": True,
        "notes": "Should surface Examples from docs when available.",
    },
    {
        "id": "examples_02",
        "question": "What parameters does the count loop bot take?",
        "category": "examples",
        "expected_facts": [
            "name", "start", "end", "increment",
        ],
        "expected_source": "@c:count-loop",
        "notes": "Bot lookup should remain grounded with sources.",
    },
    {
        "id": "gaps_01",
        "question": "What is the maximum number of workers a single RDA site can have?",
        "category": "gaps",
        "expected_facts": [
            "Should not invent a specific number",
            "Gaps or abstention about missing limit",
        ],
        "expected_source": "none or architecture without limit",
        "expect_empty_sources_if_abstain": True,
        "notes": "Gap identification: docs don't state a worker limit.",
    },
    {
        "id": "empty_sources_01",
        "question": "Who won the 2019 World Series?",
        "category": "scope",
        "expected_facts": [
            "out of scope or abstain",
            "sources empty",
        ],
        "expected_source": "none",
        "expect_empty_sources": True,
        "expect_scope": "out_of_scope",
        "notes": "Hard out-of-scope; sources must stay empty.",
    },
]


def print_eval_set():
    by_category = {}
    for case in EVAL_SET:
        by_category.setdefault(case["category"], []).append(case)
    for category, cases in by_category.items():
        print(f"\n=== {category.upper()} ({len(cases)} cases) ===")
        for case in cases:
            print(f"\n[{case['id']}] {case['question']}")
            print(f"  Expected source: {case['expected_source']}")
            print(f"  Expected facts:")
            for fact in case["expected_facts"]:
                print(f"    - {fact}")


if __name__ == "__main__":
    print_eval_set()
    print(f"\n\nTotal test cases: {len(EVAL_SET)}")
