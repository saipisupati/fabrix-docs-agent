# Continuous quality (keep testing & improving)

How to repeatedly harden the Fabrix docs agent after demos — without rewriting the architecture.

Companion docs:

- Mechanical harness steps: [QUALITY_LOOP.md](QUALITY_LOOP.md)
- Demo 2 talking points: [DEMO_2.md](DEMO_2.md)
- General demo script: [DEMO_SCRIPT.md](DEMO_SCRIPT.md)

**Hard rule:** fix with generic families / honesty / retrieve bias — never `if question == "..."` in `src/agent.py`.

---

## Cadence

| When | What | Time |
|------|------|------|
| Same day as a demo / user miss | Write down awkward questions → add 2–3 eval cases | 15–30 min |
| After adding cases | One fix category → harness | 1–2 hrs |
| Weekly (optional) | New hostile break cycle (~8–15 never-before-used questions) | half day |
| When Fabrix docs change | Re-ingest + rebuild KB → harness | as needed |

Ignore full CI tarball / docs embed until you explicitly want deploy.

---

## Same-day miss → test

1. Capture the exact question and what went wrong (wrong product, invented bot, no gaps, false confidence, etc.).
2. Add a case to `tests/eval_break.py` (adversarial) or `tests/eval_production.py` (must-always-work ops).
3. Use only soft asserts: `need_any`, `need_also_any`, `forbid_*`, `expect_abstain`, `expect_gaps`, `expect_infer_or_gaps`.
4. Do **not** special-case that question in the agent.

```bash
# stop API first (Qdrant lock)
python3 tests/run_quality_harness.py
```

---

## One class per round

Open `tests/quality_harness_digest.md`. Pick **one** tag:

| Tag | Plain meaning | Where to fix |
|-----|---------------|--------------|
| contamination | Wrong product / sibling bots | `INTEGRATION_FAMILIES`, source/example filters |
| overclaim | Too sure / invented details | honesty prompt, gaps latch, critique |
| wrong_facet | Wrong Fabrix topic | retrieve seeds, facet plan |
| abstain_fail | Said “don’t know” when docs had it | anti-abstain, topic seeds |
| empty_sources_leak | Trap answered with sources | OOS traps, empty sources on abstain |
| thin_wiring | Missing creds → bots → stream | wiring shape + critique |
| latency | p95 too high | skip clean critique, fewer LLM calls |

Fix that class generically → re-run harness → stop when exit 0.

---

## Next break cycle (hostile battery)

Use when quality feels stuck or after a big demo.

1. Add `"cycle": N` cases to `tests/eval_break.py` (~8–15 **new** questions). Cover: wrong facet, contamination, overclaim, slang/typos, multi-intent, format stress, traps, day-2.
2. Run only that cycle first:

```bash
BREAK_CYCLE=N python3 tests/eval_break.py
```

3. Triage FAIL/PARTIAL by tag; fix top 2–3 classes only.
4. Re-run `BREAK_CYCLE=N` until clean (or ≥95% pass / 0 FAIL).
5. Promote 3–5 prior failures into `tests/eval_production.py`.
6. Full gate:

```bash
python3 tests/run_quality_harness.py
```

Latest completed hostile pass: **cycle 6** (12 cases). Next unused cycle id: **7**.

---

## Docs change checklist

```bash
python3 scripts/audit_ingest_sources.py
python3 src/ingest_qdrant.py
# stop API first if upserting into the same Qdrant path
python3 src/build_kb.py
python3 tests/run_quality_harness.py
```

---

## What “good enough to demo again” means

- Harness exit 0 (production clean, break bar, readiness streak ≥ 2)
- New demo misses already captured as cases (or queued for the same day)
- You can honestly say: documented vs inferred, and gaps instead of fake certainty

---

## Parked (not the weekly loop)

- Private `RUNTIME_DATA_URL` CI gate
- Docs-site widget embed / production host ([DEPLOY_CHECKLIST.md](DEPLOY_CHECKLIST.md))
- Bigger model / fine-tuning / graph DB
