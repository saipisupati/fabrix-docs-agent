"""Shared scoring helpers for retrieval and generation eval scripts."""

import re

ABSTENTION_PHRASES = (
    "couldn't find",
    "could not find",
    "not in the documentation",
    "not in the excerpts",
    "don't have information",
    "do not have information",
    "no information",
)

FACT_STOP_WORDS = {
    "should", "state", "could", "find", "this", "that", "not", "the", "and",
    "does", "must", "are", "for", "with", "from", "into", "only", "all", "has",
    "have", "what", "how", "explicitly", "invent", "plausible", "sounding",
    "fabricated", "specific", "number",
}


def fact_tokens(fact):
    return [
        w for w in re.findall(r"[a-z0-9]+", fact.lower())
        if len(w) > 2 and w not in FACT_STOP_WORDS
    ]


def fact_hit(fact, text):
    tokens = fact_tokens(fact)
    if not tokens:
        return True
    hits = sum(1 for token in tokens if token in text)
    return hits >= max(1, int(len(tokens) * 0.5))


def chunk_ref(chunk):
    meta = chunk["metadata"]
    if meta.get("type") == "bot":
        return meta.get("bot_name", "?")
    return meta.get("source", "?")


def expected_source_hit(case, chunks):
    expected = case["expected_source"].lower()
    if expected.startswith("none"):
        return None

    refs = [chunk_ref(c).lower() for c in chunks]

    if expected.startswith("@") or expected.startswith("*"):
        return any(expected in ref or ref in expected for ref in refs)

    if "cfxql" in expected:
        has_ref = any("cfxql" in ref for ref in refs)
        if "full" in expected and "restricted" in expected:
            types = [c["metadata"].get("cfxql_type", "").lower() for c in chunks]
            return has_ref and "full" in types and "restricted" in types
        return has_ref

    return any(expected in ref for ref in refs)


def score_facts(expected_facts, text):
    text_lower = text.lower()
    found = [fact for fact in expected_facts if fact_hit(fact, text_lower)]
    missing = [fact for fact in expected_facts if fact not in found]
    total = len(expected_facts)
    return {
        "facts_found": found,
        "facts_missing": missing,
        "fact_score": len(found) / total if total else 1.0,
    }


def abstained(answer):
    text = answer.lower()
    return any(phrase in text for phrase in ABSTENTION_PHRASES)


def looks_hallucinated(answer):
    text = answer.lower()
    if abstained(answer):
        return False
    if re.search(r"\b\d+\b", text):
        return True
    if len(text.split()) > 40:
        return True
    procedural = ("step 1", "first,", "click", "navigate to", "go to settings")
    return any(marker in text for marker in procedural)


def score_retrieval(case, chunks):
    if case["category"] == "negative":
        return {
            "source_hit": None,
            "facts_found": [],
            "facts_missing": [],
            "fact_score": None,
            "note": "Negative cases are graded on generation, not retrieval",
        }

    text = " ".join(c["text"] for c in chunks).lower()
    score = score_facts(case["expected_facts"], text)
    score["source_hit"] = expected_source_hit(case, chunks)
    return score


def score_generation(case, answer):
    if case["category"] == "negative":
        did_abstain = abstained(answer)
        hallucinated = looks_hallucinated(answer)
        facts_found = []
        facts_missing = []
        if did_abstain:
            facts_found.append(case["expected_facts"][0])
        else:
            facts_missing.append(case["expected_facts"][0])
        if not hallucinated:
            facts_found.append(case["expected_facts"][1])
        else:
            facts_missing.append(case["expected_facts"][1])
        fact_score = len(facts_found) / len(case["expected_facts"])
        return {
            "facts_found": facts_found,
            "facts_missing": facts_missing,
            "fact_score": fact_score,
            "abstained": did_abstain,
            "hallucinated": hallucinated,
        }

    score = score_facts(case["expected_facts"], answer)
    text_lower = answer.lower()
    if "restricted cfxql" in text_lower and "full cfxql" not in text_lower:
        if any("full cfxql" in f.lower() for f in case["expected_facts"]):
            score["contradiction"] = "wrong CFXQL type"
    return score


def grade_retrieval(score):
    if score.get("note"):
        return "SKIP"
    source_ok = score["source_hit"]
    fact_score = score["fact_score"]
    if source_ok and fact_score >= 0.8:
        return "PASS"
    if source_ok or fact_score >= 0.5:
        return "PARTIAL"
    return "FAIL"


def grade_generation(score, case):
    if case["category"] == "negative":
        if score.get("abstained") and not score.get("hallucinated"):
            return "PASS"
        if score.get("abstained") or not score.get("hallucinated"):
            return "PARTIAL"
        return "FAIL"

    if score.get("contradiction"):
        return "FAIL"
    fact_score = score["fact_score"]
    if fact_score >= 0.8:
        return "PASS"
    if fact_score >= 0.5:
        return "PARTIAL"
    return "FAIL"
