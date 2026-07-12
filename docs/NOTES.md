# Fabrix Docs Agent: project notes

RAG agent that answers questions about [Fabrix.ai](https://docs.fabrix.ai) documentation: bot catalog, CFXQL reference, and platform guides. Pipeline: chunk → embed → store in Qdrant → retrieve → generate with gpt-4o-mini.

For setup and commands see [README.md](../README.md). This file is the project story: what we built, what we learned, and where we are now.

---

## Current status (2026-07-11)

| | |
|---|---|
| **Pipeline** | `ingest_qdrant.py` → OpenRouter MiniLM → local Qdrant + **structured KB** (`build_kb.py` → `data/kb/`) → `agent.py` (merged scope+facets + KB-first + optional critique) → gpt-4o-mini |
| **Corpus** | **6,387 chunks** from **543** MD files (214 bots + 328 narrative + CFXQL + 3 root pages) |
| **KB** | Topics/entities/facts/procedures under `data/kb/`; agent retrieves KB first, chunks as fallback |
| **Answer contract** | Unified `answer` + `examples` + `gaps` + `sources` + `used_inference` + optional `timing`; inference disclosed via footer |
| **Production eval** | **17/17 PASS** (ops + break regressions) |
| **Break eval** | **14/14 PASS** adversarial battery |
| **Readiness gate** | Latest: **10/10 PASS**, p50 ≈ **11.8s**, p95 ≈ **14.3s**, avg **1.8** LLM calls (budget 45s). **Do not embed on docs.fabrix.ai until GREEN twice in a row.** |
| **Embedding choice** | Stay on **OpenRouter MiniLM + Qdrant** (stakeholder sign-off after fastembed shootout) |
| **Phase** | Quality loop (latency + polish + readiness); docs site embed **blocked** until readiness gate holds |

**Latency note:** related/ops path merges scope+facet planning into one LLM call, skips critique when a local draft check passes, and retries only on abstain / missing related inference. CLI and `/ask` expose `timing` (`scope_ms`, `retrieve_ms`, `generate_ms`, `critique_ms`, `total_ms`, `llm_calls`).

**Before re-ingest:** run `python3 scripts/audit_ingest_sources.py` (543 files must stay within public docs export).

**KB build / deploy:** `python3 src/build_kb.py` after ingest. For VM deploy, copy `data/qdrant_db/` **and** `data/kb/` (or rebuild on server). Stop the API before Qdrant upserts (single-process file lock).

**Technical inference smoke:** ask multi-doc Fabrix composition questions (ServiceNow → pstream, Toolset vs Persona, K8s inventory → dashboard dataset). Expect a unified **path-first** answer (`**Documented Fabrix path**` header + numbered steps) + `used_inference` / disclosure. Run `python3 tests/eval_production.py` and `python3 tests/eval_readiness.py`.

Latest eval output (gitignored): `tests/eval_*_results.txt`.

---

## What this project is

Fabrix.ai docs describe hundreds of pipeline **bots** (automation building blocks), a query language called **CFXQL**, and narrative guides (installation, architecture, AI fabric, etc.). Users ask natural-language questions; this agent retrieves relevant doc chunks and generates grounded answers.

**Stakeholder goal:** find the right embedding + retrieval setup for cost and accuracy on this doc set. CFX already uses Qdrant; we started there.

**How we measure progress:** hand-built eval cases in [`tests/eval_set.py`](../tests/eval_set.py) cover bot lookups, CFXQL comparisons, multi-part questions, narrative guides, negative (hallucination) cases, and KB scope/inference/examples/gaps. Scripts score automatically:

- [`tests/run_eval_baseline.py`](../tests/run_eval_baseline.py): retrieval only (source hit + facts in top-k chunks)
- [`tests/run_eval_generation.py`](../tests/run_eval_generation.py): oracle full pipeline (manual category + filter hints per case)
- [`tests/run_eval_agent.py`](../tests/run_eval_agent.py): real-user proxy via `agent.answer()` (no category or filter hints)
- [`tests/run_eval_kb.py`](../tests/run_eval_kb.py): scope / empty sources / examples / gaps / labeled inference
---

## Decisions made

| Decision | Rationale |
|----------|-----------|
| Classic RAG with Qdrant | Standard chunk → embed → retrieve → generate; Qdrant is org standard |
| MiniLM-L6-v2 via OpenRouter for ingest | 10/10 retrieval on 12-case eval; fast API; no local fastembed model hit 10/10 |
| gpt-4o-mini for generation | Stable, good enough for grounded Q&A; OpenRouter free models were unreliable (404/429) |
| Markdown-header chunking for CFXQL + narrative | Size-based splitting broke CFXQL comparison answers |
| One chunk per bot (`##` headers) | 214 bot files → 1,834 chunks; validated 0 errors |
| Category filters + bot-name re-rank at query time | Fixed lookup/multi_part retrieval when vector search ranked wrong bot |
| Lookup chunk pruning + category prompts | Fixed LLM citing sibling bot params despite correct retrieval |
| Public MD export as doc source | 543 files from `docs.fabrix.ai` export tree; not a web scraper |
| Eval drift fix for `negative_01` | Swapped password-reset question (now answerable from corpus) for subscription-cancel question |

**Fastembed shootout (local CPU, in-memory):** arctic-embed-xs 9/10 ~19 min; bge-base 9/10 ~96 min; production MiniLM 10/10. Best local fallback if API unavailable: arctic-embed-xs. See [Appendix: timeline](#appendix-timeline) for details.

---

## Roadmap (4 phases)

| Phase | Goal | Status |
|-------|------|--------|
| 1 | Bot catalog + CFXQL, basic retrieve-and-answer pipeline | **Done** |
| 2 | Citations + mock test questions for accuracy | **Done** (12-case eval harness, automated scoring) |
| 3 | Improve chunking for pipeline/narrative docs | Not started |
| 4 | Agent routes by product area, multi-part handling, docs site integration | **In progress** (agent routing, API, widget done; docs site embed pending) |

Later ideas: doc generation, screenshot verification, graph DB for file relationships, version hashing for doc updates.

---

## Eval summary (12 cases)

| Category | Cases | What it tests |
|----------|-------|---------------|
| lookup | 5 | Bot parameter / behavior questions |
| comparison | 2 | Full vs Restricted CFXQL |
| multi_part | 1 | Bot behavior + CFXQL type in one answer |
| guide | 3 | `beginners_guide` narrative |
| install | 1 | `installation_guides` |
| ai_fabric | 1 | LLM pooling purpose |
| pipeline | 1 | `Pipelines` sample pipeline |
| datasource | 2 | `Datasource_Integrations` (ServiceNow, Kubernetes) |
| extensions | 1 | `Extensions` (agentic_ai) |
| releases | 1 | `rda_releases` (cfxOIA) |
| negative | 2 | Must abstain, not hallucinate |

**Latest scores (2026-06-24):**

| Layer | Result |
|-------|--------|
| Retrieval (10 scored) | 10 PASS, 0 PARTIAL, 0 FAIL; 2 negative SKIP |
| Generation (12 scored) | 11 PASS, 1 PARTIAL (`multi_part_01`), 0 FAIL |

| Case | Source | Retrieval | Generation |
|------|--------|-----------|------------|
| lookup_01, lookup_02 | bot catalog | PASS | PASS |
| comparison_01, comparison_02 | CFXQL | PASS | PASS |
| multi_part_01 | exec.md | PASS | PARTIAL (variance) |
| guide_01–03 | beginners_guide | PASS | PASS |
| install_01 | data_retention.md | PASS | PASS |
| ai_01 | llm_pooling.md | PASS | PASS |
| negative_01, negative_02 | (none) | SKIP | PASS |

---

## Agent build -- 2026-06-24

Stakeholder decisions (locked):

- Embeddings: OpenRouter MiniLM + local Qdrant, no further shootouts
- Next deliverable: docs.fabrix.ai integration via agent + API + widget

Architecture added:

- [`src/doc_urls.py`](../src/doc_urls.py): shared chunk to public URL mapping
- [`src/agent.py`](../src/agent.py): deterministic query router (`plan_query`) + orchestrator (`answer`) + answer judge
- [`src/api.py`](../src/api.py): FastAPI `POST /ask` + `GET /health`, shared QdrantClient lifespan, CORS, optional API key
- [`widget/ask-widget.js`](../widget/ask-widget.js) + [`ask-widget.css`](../widget/ask-widget.css): vanilla JS embeddable widget
- [`docs/DOCS_SITE_INTEGRATION.md`](DOCS_SITE_INTEGRATION.md): handoff doc for docs site team

Router rules (`plan_query`, deterministic-first):

1. Bot name hint → `type_filter=bot`, `category_hint=lookup` / `multi_part`
2. CFXQL keywords → `top_k=10`, no filter
3. Comparison keywords → `top_k=10`
4. Section keyword map → `type_filter=narrative`, `doc_section=...`
5. Negative keywords → `category_hint=negative`
6. LLM fallback (logged when fired)

Agent eval ([`run_eval_agent.py`](../tests/run_eval_agent.py), no category hints): **20 cases** covering all ingested doc sections. Latest (2026-06-26): **PASS=19, PARTIAL=1, FAIL=0**. `multi_part_01` PARTIAL is known LLM variance, not a routing issue.

Bot sample eval ([`run_eval_bot_sample.py`](../tests/run_eval_bot_sample.py), retrieval-only): random 15-bot spot check (seed 42) over **922** catalog bots. Latest: **PASS=11, FAIL=4** (generic slugs like `license` / `update` are the usual misses).

Router additions for expanded eval: `extension` → Extensions, `cfxoia` → rda_releases, `servicenow` / `kubernetes` / `integrate` → Datasource_Integrations.

Deployment: `uvicorn src.api:app --port 8080`; see [DOCS_SITE_INTEGRATION.md](DOCS_SITE_INTEGRATION.md). Production checklist: [DEPLOY_CHECKLIST.md](DEPLOY_CHECKLIST.md). Demo script: [DEMO_SCRIPT.md](DEMO_SCRIPT.md).

Blocker: docs repo access TBD for widget embed.

---

## Architecture

| Stage | Ingest | Embed | Store | Generate |
|-------|--------|-------|-------|----------|
| v0 (legacy) | `ingest.py` | TF-IDF | Chroma | stub |
| v1 | `ingest_with_real_embeddings.py` | TF-IDF → MiniLM | Qdrant | openrouter/free |
| **v2 (current)** | `ingest_qdrant.py` | MiniLM via OpenRouter | local Qdrant | gpt-4o-mini |
| remote (WIP) | `ingest_and_test_remote.py` | BGE-large (server) | Faizan's Qdrant | TBD |

### Corpus growth

| Phase | What's included | Chunks |
|-------|-----------------|--------|
| Baseline | 214 bot `.md` + `cfxql.md` | **1,847** |
| + beginners_guide, reference_guides | narrative | **2,856** |
| **Full (current)** | + 6 more folders in `DOCS_INCLUDE_DIRS` + 3 root pages via `DOCS_ROOT_FILES` | **6,387** |

All 543 export files ingested. Root pages: `index.md`, `Datasets.md`, `Formatting-Templates.md`. `split_oversized_chunks()` split 228 chunks >8k chars before embed (OpenRouter input limit).

Narrative breakdown (328 files): beginners_guide 1,002 chunks, reference_guides 7, installation_guides 1,434, Pipelines 108, Datasource_Integrations 464, ai_fabric 481, Extensions 220, rda_releases 69, root pages 50.

---

## Product / doc structure (reference)

### Bot catalog pages

Each extension page groups multiple bots. Every bot entry follows the same pattern:

- Short description of what it does
- Which CFXQL type it needs (Full or Restricted)
- Parameter list for that bot

Example from [docs.fabrix.ai/Bots/search_bots/](https://docs.fabrix.ai/Bots/search_bots/):

**Bot @c:timed-loop**: starts a looping block that waits x seconds before running again.

Parameter table: name, type, default, description. e.g. `interval`, `stop_after`, `max_iterations`.

### Bot type prefixes

| Prefix | Type | CFXQL | Role |
|--------|------|-------|------|
| `#` | Source filtering | Full | Translate query into remote data source API/DB call |
| `*` | Destination filtering | Full | Filter data already in memory/pipeline |
| `@` | API | Restricted | Extract API params that control bot behavior |

### CFXQL

**Full CFXQL**: SQL-like; filter/condition (`is`, `not in`, `!=`, AND/OR), GET clause for columns. Used on already-loaded data frames.

**Restricted CFXQL**: simpler; each param via `=`. Used by API bots.

---

## Key technical learnings

### Chunking

1. **Size-based splitting breaks CFXQL:** fragmented chunks, wrong comparison answers
2. **Header-based splits fix CFXQL:** `MarkdownHeaderTextSplitter` on `cfxql.md` → 13 chunks with correct metadata
3. **Bot catalog:** `clean_markdown.py` strips YAML/CSS/HTML; split on `##` → one chunk per bot (1,834 chunks, 0 errors)
4. **Prefix distribution:** @ 84%, * 14%, # 2%

### Embeddings

- TF-IDF works on tiny samples, fails at ~1,800+ chunks (CFXQL questions retrieve unrelated bots)
- MiniLM-L6-v2 avg rank **1.00** on CFXQL probe questions; BGE-large tied worst at **2.00** on same probe
- Production path: OpenRouter MiniLM 10/10 on full eval; local fastembed best was arctic-xs 9/10

### Retrieval fixes (why eval went from 2/5 to 10/10)

1. **Skip sample `.txt` bots** that outranked real catalog chunks
2. **Per-category `top_k`** (comparison needs 10, lookup needs 5)
3. **`FILTER_BY_CATEGORY`** in eval + query (bot vs narrative vs doc_section)
4. **`bot_name_hints()` + `rerank_by_bot_name()`** with expanded candidate pool (300) for lookup/multi_part

### Generation fixes

1. **Category-aware prompts** for multi_part and lookup
2. **`prune_lookup_chunks()`** drops sibling bots (e.g. `@exec:count-loop` when asking about count-loop)
3. **Tighter abstention prompt** for negative category
4. **Eval drift fix:** swapped `negative_01` when corpus expansion made old question answerable

---

## Tooling and audits

| Script | Purpose |
|--------|---------|
| [`scripts/audit_ingest_sources.py`](../scripts/audit_ingest_sources.py) | Pre-ingest: all sources within `DOCS_ROOT` / `BOTS_DIR`. Optional `--verify-urls` vs `docs.fabrix.ai` |
| [`scripts/audit_eval_sources.py`](../scripts/audit_eval_sources.py) | Verify eval bot names exist in `BOTS_DIR` |
| [`scripts/batch_ingest_narrative.py`](../scripts/batch_ingest_narrative.py) | Chunk-only validation for narrative folders |
| [`src/test_fastembed_eval.py`](../src/test_fastembed_eval.py) | Compare local fastembed models (no Qdrant re-ingest) |

---

## Blockers and next steps

| Item | Status |
|------|--------|
| Remote ingest via VPN (`ingest_and_test_remote.py`) | Blocked on timeouts for large files (e.g. cfxdm.md) |
| Faizan shared Qdrant + fastembed server | Available; not integrated into primary path |
| Phase 3 narrative chunking improvements | Not started |
| Phase 4 agent routing + docs site integration | In progress (embed pending) |
| bge-small 10-case fastembed rerun | Optional; not blocking |

**Calls:** Ravi & Faizan (2026-06-17): shared VPN Qdrant, Docker, fastembed. Plan was test models on shared server; later add agent that judges answer sufficiency and retries retrieval.

---

## Open questions

**Resolved**

- RAG + vector DB? Yes, Qdrant + classic RAG
- Embedding model for production? OpenRouter MiniLM-L6-v2 (fastembed shootout confirmed)
- Doc source? Public MD export (543 files), paths in `config.py`
- Framework? LangChain utilities for chunking; generation via OpenAI API

**Still open**

- Repo/org conventions for shipping this beyond prototype
- Whether to pursue remote BGE-large path vs staying local MiniLM
- Phase 3 chunking strategy for long narrative pages (installation_guides are 1,434 chunks alone)

---

## Repo structure

```
fabrix-docs-agent/
├── README.md
├── docs/
│   ├── NOTES.md           <- this file
│   ├── DOCS_SITE_INTEGRATION.md
│   ├── DEPLOY_CHECKLIST.md
│   └── DEMO_SCRIPT.md
├── widget/
│   ├── ask-widget.js
│   └── ask-widget.css
├── requirements.txt
├── .env                   <- gitignored API keys
├── data/
│   ├── raw/               -> cfxql_reference.txt only (sample bot txts skipped)
│   └── qdrant_db/         -> local vector store (gitignored)
├── src/
│   ├── config.py          -> paths, models, .env loading
│   ├── doc_urls.py        -> chunk metadata to docs.fabrix.ai URLs
│   ├── ingest_qdrant.py   -> chunk + embed + store
│   ├── query_qdrant.py    -> retrieve + generate (engine)
│   ├── agent.py           -> query router + orchestrator
│   ├── api.py             -> FastAPI HTTP layer
│   ├── ingest_and_test_remote.py
│   └── ...
├── scripts/
│   ├── audit_eval_sources.py
│   └── audit_ingest_sources.py
└── tests/
    ├── eval_set.py
    ├── eval_scoring.py
    ├── run_eval_baseline.py
    ├── run_eval_generation.py
    ├── run_eval_agent.py
    └── run_eval.py
```

---

## Appendix: timeline

Newest first. Iteration-level eval tables omitted; see [Eval summary](#eval-summary-12-cases) for current scores.

### 2026-06-24

- Agent build phases 1–6: `doc_urls`, `agent`, `api`, widget, agent eval harness
- Agent eval: **11 PASS, 1 PARTIAL, 0 FAIL** (matches oracle baseline)
- Full 12-case generation eval: **11 PASS, 1 PARTIAL, 0 FAIL**
- `install_01` and `ai_01` PASS on first generation run
- Documented in commit `2a358d4`

### 2026-06-23

- Added eval cases `install_01` (installation_guides), `ai_01` (ai_fabric)
- Retrieval baseline: **10/10** scored on expanded eval
- Fastembed shootout on 6,331 chunks ([`test_fastembed_eval.py`](../src/test_fastembed_eval.py)):

| Model | dim | PASS/10 | Time |
|-------|-----|---------|------|
| MiniLM (fastembed) | 384 | ~7/10 | ~3 min |
| snowflake-arctic-embed-xs | 384 | 9/10 | ~19 min |
| BAAI/bge-small-en-v1.5 | 384 | 8/8 (old harness) | ~47 min |
| BAAI/bge-base-en-v1.5 | 768 | 9/10 | ~96 min |
| **OpenRouter MiniLM + Qdrant** | 384 | **10/10** | n/a |

Decision: keep OpenRouter MiniLM for ingest. Log: `/tmp/fastembed_shootout_10.log`.

### 2026-06-22

- Major eval + corpus day:
  - Unified ingest, `config.py`, `.env` loading, eval baseline scripts
  - Retrieval iteration: 2/5 → 3/5 → **5/5** (filters, re-rank, per-category top_k)
  - Generation iteration: lookup pruning + prompts → **7/7** on bot-only cases
  - Corpus expansion: 8 narrative folders → **6,331 chunks**
  - Eval drift: `negative_01` swapped (password reset now in corpus)
  - Added 3 `guide` cases (beginners_guide): retrieval 8/8, generation 9/10
  - CFXQL: swapped sample txt for real `cfxql.md` (13 chunks)
  - Added `audit_eval_sources.py`, `audit_ingest_sources.py`

### 2026-06-18

- Generation switched to gpt-4o-mini
- TF-IDF retrieval failed at full scale on CFXQL comparison
- Swapped to MiniLM embeddings locally
- Remote ingest blocked on VPN/timeouts

### 2026-06-17

- Got MD docs via VPN (942 files in export)
- Built bot catalog chunking: 214 files, 1,834 chunks, 0 errors
- Tested `MarkdownHeaderTextSplitter` on cfxql.md
- Call with Faizan: remote Qdrant plan
- Embedding comparison: MiniLM beat BGE-large on local CFXQL probe
