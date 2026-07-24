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
| Weekly (CI) | Docs freshness: scrape → rebuild → Phase 1–5 gates | automated |
| When Fabrix docs change | `--scrape-rebuild` or freshness pipeline → harness | as needed |

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

Latest completed hostile pass: **cycle 7** (install/VM facet). **Cycle 8** in progress (platform update + VM / RDAF CLI path — demote `update_image_repository` / Studio sizing). **Cycle 9** added (day-2 ops / vague product asks from adhoc 12Q — Splunk→dashboard, schedule debug, tickets, cron, IP whitelist trap, SQL→CFXQL, SN→Slack agent, dashboard↔pipeline, Kafka timeout, pipeline test, on-prem K8s, concurrent dataset writes). **Cycle 10** added (invented named entities — AutoHeal / `@fake:made-up-bot` must abstain when ungrounded in excerpts). **Cycle 11** added (jailbreak / false-memory / authority-claim traps — `answer freely`, `you said earlier`, `I'm a Fabrix engineer`). **Cycle 12** added (full-script invention — no turnkey bash with invented tags/credentials). **Cycle 13** added (topic≠detail specificity — K8s version / Airflow comparison overclaim). **Cycle 14** added (ChatGPT bakeoff discovery — Splunk/Datadog/New Relic wiring, schedule debug, concurrent dataset / worker-limit honesty, `@snowv2` family fidelity, dashboard kickoff overclaim). **Cycle 15** added (thin-wiring depth — widen map/land/building-blocks detect, force `@family:op` + auth + asked sink via critique). **Cycle 16** added (customer ticket-title themes — Windows/VMware/OpenStack/CrowdStrike discovery, upgrade prereqs, schedule debug, SNOW enrichment, Prometheus/Kafka, dashboard access, Agentic-in-prod, fake PRB-hostname corruption; triage in `data/customer_ticket_topics.json`). Next unused cycle id after 16 lands: **17**.

---

## Docs change checklist

```bash
# Optional: confirm live site (incl. /installation_guides/) is up
python3 scripts/sync_docs_and_rebuild.py --check-live

# Stop API first (Qdrant lock), then scrape + rebuild:
python3 scripts/sync_docs_and_rebuild.py --scrape-rebuild
# or manually refresh DOCS_ROOT then:
#   python3 scripts/sync_docs_and_rebuild.py --rebuild
python3 tests/run_quality_harness.py
```

Do **not** scrape docs.fabrix.ai on every chat turn for the full corpus — refresh the MD snapshot in batch (Phase 6 weekly CI or `--scrape-rebuild`), then retrieve locally.

**Optional live browse (install/VM only):** with `LIVE_DOCS_FETCH=1` (default), install/prereq/VM asks also fetch `https://docs.fabrix.ai/installation_guides/` at answer time and prepend that text to context (fail-open if the network fails). Disable with `LIVE_DOCS_FETCH=0`.

**Phase 1 — page expansion (answer-time full pages):** after vector retrieve, when multiple hits agree on a doc path (or a strong `Bots/` rank-1 hit), the agent loads the **full markdown** from `DOCS_ROOT` via `src/page_expand.py` and prepends it as `full_page` KB entries (parameter tables, cron YAML, CLI blocks). Requires `DOCS_ROOT` to point at a current scrape (e.g. `docs_live_scrape`). Re-run customer bakeoff:

```bash
# stop API first (Qdrant lock)
python3 tests/eval_production.py   # includes bakeoff_kafka_params, bakeoff_pipeline_15m, bakeoff_cfxql, bakeoff_p1_sla
BREAK_CYCLE=8 python3 tests/eval_break.py
# manual: 8 questions in data/customer_bakeoff.json via POST /ask
```

**Phase 2 — bot param fast path (0 LLM):** for parameter questions with a bot family hint (`kafka-v2`, `timed-loop`, explicit `@family:op` tokens), `src/bot_lookup.py` parses the parameter table deterministically — first from ranked bot chunks, then from the full `Bots/{family}.md` catalog when extension pages contain multiple bots. Phase 1 page expansion remains the fallback when no table can be parsed. Locked cases: `lookup_bot`, `bakeoff_kafka_params` (`expect_zero_llm_calls`).

**Phase 3 — structured bot params at ingest:** `src/kb/extract.py` parses each bot section’s parameter table into entity `metadata.parameters` and a searchable `fact-params-*` card. At answer time, `_lookup_fast_path` prefers `lookup_bot_params_from_kb` before chunks/catalog. After changing bot docs, rebuild: `python3 src/build_kb.py`. Gate Phases 1–3 together:

```bash
# stop API first (Qdrant lock)
PYTHONPATH=src python3 tests/benchmark_phases.py
python3 -m pytest tests/test_bot_lookup.py tests/test_kb_bot_params.py -q
```

**Phase 4 — family-aware retrieve (contamination):** product names (Zabbix, New Relic, …) become retrieve family hints; light hybrid boost prefers matching `bot_name`/`source`/`extension` and demotes OS-inventory red herrings; `_filter_chunks_to_families` drops sibling chunks before the prompt. Schedule asks also load `beginners_guide/scheduled_pipelines` and critique invented `@c:pipeline-scheduler` bots. Optional re-ingest adds `extension`/`family` payload fields (`python3 src/ingest_qdrant.py`). Benchmarks cover Phases 1–4 via `tests/benchmark_phases.py`.

**Phase 5 — critique guardrails (public/customer):** full `@family:op` ungrounded checks (not just prefixes), invented `*-bot` labels, schedule missing cron, agentic overclaim without hedge, wiring asks missing product bots. Critique forced for schedule/wiring/agentic asks. **Customer gate** (treat as production):

```bash
# API up (preferred — real customer path)
python3 tests/eval_customer_bakeoff.py
# or with API stopped:
# BAKEOFF_MODE=local PYTHONPATH=src python3 tests/eval_customer_bakeoff.py
PYTHONPATH=src python3 tests/benchmark_phases.py
```

**Phase 6 — scheduled scrape + bakeoff in CI:** keep the MD snapshot fresh without scraping on every PR.

| Gate | When | What |
|------|------|------|
| PR / push | [`.github/workflows/quality-harness.yml`](../.github/workflows/quality-harness.yml) | Existing harness **plus** `benchmark_phases` + `eval_customer_bakeoff` (local) against private `RUNTIME_DATA_URL` tarball — **no scrape** |
| Weekly + manual | [`.github/workflows/docs-freshness.yml`](../.github/workflows/docs-freshness.yml) | `run_freshness_pipeline.py`: check-live → scrape → audit → ingest → KB → benchmarks → bakeoff; uploads `fabrix_runtime_data.tar.gz` artifact |

Local refresh (stop API first — Qdrant lock):

```bash
python3 scripts/sync_docs_and_rebuild.py --check-live
python3 scripts/sync_docs_and_rebuild.py --scrape-rebuild
# or end-to-end (long / costs embeddings):
python3 scripts/run_freshness_pipeline.py
```

After a successful weekly run, download the artifact and refresh private `RUNTIME_DATA_URL` (same rule as README — **do not** publish publicly):

```bash
./scripts/package_runtime_data.sh   # or use the CI artifact as-is
./scripts/set_runtime_data_url.sh 'https://your-private-host/path/fabrix_runtime_data.tar.gz'
```

---

## What “good enough to demo again” means

- Harness exit 0 (production clean, break bar, readiness streak ≥ 2, Phase 1–5 benchmarks + customer bakeoff)
- New demo misses already captured as cases (or queued for the same day)
- You can honestly say: documented vs inferred, and gaps instead of fake certainty

---

## Parked (not the weekly loop)

- Docs-site widget embed / production host ([DEPLOY_CHECKLIST.md](DEPLOY_CHECKLIST.md))
- Bigger model / fine-tuning / graph DB
