"""
eval_set.py — a small, hand-built set of test questions with known-correct
answers, used to measure whether the pipeline (chunking + retrieval +
generation) is actually accurate, not just "looks fine when I try it."
"""

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
        "id": "negative_01",
        "question": "How do I reset my password for the Fabrix.ai dashboard?",
        "category": "negative",
        "expected_facts": [
            "Should explicitly state it could not find this in the documentation",
            "Should NOT invent a plausible-sounding but fabricated answer",
        ],
        "expected_source": "none - no chunk in our sample set covers this",
        "notes": "Tests hallucination resistance.",
    },
    {
        "id": "negative_02",
        "question": "What is the maximum number of workers a single RDA site can have?",
        "category": "negative",
        "expected_facts": [
            "Should state it could not find this in the documentation",
            "Should NOT invent a specific number",
        ],
        "expected_source": "none - architecture page not in our sample doc set",
        "notes": "More tempting hallucination case since the topic is real.",
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
