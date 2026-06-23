# Fabrix Docs Agent: project notes and changelog

Living log of research, findings, meetings, and eval runs. For setup and commands see [README.md](../README.md).

---

## Product / doc structure

### Bot catalog pages

Each extension page groups multiple bots. Every bot entry follows the same pattern:

- Short description of what it does
- Which CFXQL type it needs (Full or Restricted)
- Parameter list for that bot

Example from [docs.fabrix.ai/Bots/search_bots/](https://docs.fabrix.ai/Bots/search_bots/):

**Bot @c:timed-loop**: starts a looping block that waits x seconds before running again.

Parameter table: name, type, default, description. e.g. `interval`, `stop_after`, `max_iterations`.

Also shows example pipelines using the bot.

### Bot type prefixes

| Prefix | Type | CFXQL | Role |
|--------|------|-------|------|
| `#` | Source filtering | Full | Translate query into remote data source API/DB call |
| `*` | Destination filtering | Full | Filter data already in memory/pipeline |
| `@` | API | Restricted | Extract API params that control bot behavior |

### CFXQL reference

**Full CFXQL**: SQL-like language for what each bot should do with data:

- Filter/condition clause: `is`, `not in`, `!=`, AND/OR grouping
- GET clause: select/rename output columns
- Used by bots working on already-loaded data frames

**Restricted CFXQL**: simpler; each param specified via `=`

---

## Original plan (4 phases)

1. Start with bot catalog + CFXQL, get a basic retrieve-and-answer pipeline working end-to-end
2. Add citations back to source pages + mock test questions for accuracy
3. Improve chunking for pipeline/narrative docs
4. Agent routes questions to correct product area, handles multi-part questions, integrate with docs site

Later ideas: generate docs, verify against screenshots, graph DB for file relationships, version hashing for doc updates.

---

## Open questions

- **RAG + vector DB?** Yes, classic RAG (chunk -> embed -> store -> retrieve -> generate)
- **Framework?** LangChain used for chunking utilities; LlamaIndex not evaluated deeply
- **Embedding model?** Mistral embed tested; **MiniLM-L6-v2** won local comparison; generation via **gpt-4o-mini** (OpenAI)
- **Doc source?** Internal MD export via VPN (942 files), not web scraper
- **Repo/org conventions?** TBD
- **Dev environment?** Python venv; Faizan's server is Docker + fastembed

Main task from stakeholders: find the right tool/framework for cost and accuracy. CFX has Qdrant, start there; test OpenRouter free models.

---

## Architecture evolution

| Stage | Ingest | Embed | Store | Generate |
|-------|--------|-------|-------|----------|
| v0 | `ingest.py` | TF-IDF | Chroma | stub |
| v1 | `ingest_with_real_embeddings.py` | TF-IDF -> MiniLM | Qdrant | openrouter/free |
| v2 (current) | `ingest_qdrant.py` | MiniLM via OpenRouter | Qdrant | gpt-4o-mini |
| remote (WIP) | `ingest_and_test_remote.py` | BGE-large (server) | Faizan's Qdrant | TBD |

Corpus (evolution):

| Phase | What's included | Chunks |
|-------|-----------------|--------|
| Baseline | 214 bot `.md` + `cfxql.md` | **1,847** (1,834 bot + 13 CFXQL) |
| Phase 1 | + `beginners_guide`, `reference_guides` | **2,856** (+1,009 narrative) |
| Full | + 6 more folders (see `DOCS_INCLUDE_DIRS`) | **6,331** (+4,484 narrative after oversized splits); **540/543** source files |

Source files: **540 ingested** of **543** in export (214 bot + 325 narrative + 1 CFXQL; skipped `Datasets.md`, `Formatting-Templates.md`, `index.md`). Narrative breakdown (325 files, pre-oversized-split): beginners_guide 1,002 chunks, reference_guides 7, installation_guides 1,434, Pipelines 108, Datasource_Integrations 464, ai_fabric 481, Extensions 220, rda_releases 69. `split_oversized_chunks()` split 225 chunks >8k chars before embed.

---

## Chunking learnings

1. **Size-based splitting breaks CFXQL** -- 11 fragmented chunks, all tagged generic `reference`; comparison answers were wrong/backwards
2. **Hand-rolled section splits fix CFXQL** -- split at `Full CFXQL` / `Restricted CFXQL` headers -> 4 clean chunks with correct `cfxql_type` metadata; flipped wrong answer to correct without changing embeddings or the LLM
3. **Markdown-aware splitting generalizes** -- `MarkdownHeaderTextSplitter` on real `cfxql.md` -> 13 chunks with `{h2, h4}` metadata, zero false positives
4. **Bot catalog** -- `clean_markdown.py` strips YAML/CSS/HTML; split on `##` -> one chunk per bot; validated 214 files, 0 errors, 1,834 chunks
5. **Prefix distribution** -- @ 84%, * 14%, # 2%
6. **Generalization options** -- markdown headers (best), plain-text heuristics (noisy), recursive splitter tuning, LLM chunking (expensive)

---

## Embedding learnings

- TF-IDF works on tiny samples, fails at scale (~1,800+ chunks) -- conceptual CFXQL questions retrieve unrelated bots
- Real embeddings (MiniLM, Mistral, Nemotron) correctly rank Full/Restricted CFXQL chunks
- Surprising result: MiniLM-L6-v2 (smallest/cheapest model) avg rank **1.00** on 3 CFXQL questions; BGE-large tied for worst at **2.00**
- Free OpenRouter models are unstable (404/429); `openrouter/free` auto-router worked reliably by routing around congested providers

---

## Eval history

### Early run (small corpus, TF-IDF, section-aware CFXQL, openrouter/free)

6/7 cases graded manually -- mostly PASS including comparison_01/02, negative cases, multi_part_01 (minor end-if omission).

### Retrieval baseline 2026-06-22 (before sample-txt skip)

PASS=2, PARTIAL=3 -- `c_extension_loop_bots.txt` sample file outranked real bot chunks.

### Retrieval baseline 2026-06-22 (after sample-txt skip, top_k=5)

PASS=2, PARTIAL=3 -- lookup_01 fixed (@c:count-loop rank 1); lookup_02 and multi_part_01 still PARTIAL (bots may be missing from markdown catalog).

### Retrieval baseline 2026-06-22 (per-category top_k: lookup=5, comparison=10, multi_part=8)

**PASS=3, PARTIAL=2, FAIL=0, SKIP=2**

| Case | Grade | Notes |
|------|-------|-------|
| lookup_01 | PASS | @c:count-loop rank 1 |
| lookup_02 | PARTIAL | @c:data-loop not in top-5 despite being in catalog |
| comparison_01 | PASS | |
| comparison_02 | PASS | Fixed by comparison top_k=10 (was PARTIAL at top_k=5) |
| multi_part_01 | PARTIAL | *exec:if-condition not in top-8; @exec:end-if rank 1 |
| negative_* | SKIP | Graded via generation eval below |

**Catalog audit (same day):** All three eval bot names exist in `BOTS_DIR` -- `@c:count-loop` and `@c:data-loop` in `control.md`, `*exec:if-condition` in `exec.md`. Lookup/multi_part PARTIAL grades are retrieval ranking issues, not missing docs.

**Ingest source audit:** [`scripts/audit_ingest_sources.py`](../scripts/audit_ingest_sources.py): pre-ingest checkpoint; every file under `load_and_chunk_all()` must resolve inside `DOCS_ROOT` / `BOTS_DIR`. Local audit PASSED (540 files). Optional `--verify-urls` HEAD-checks `docs.fabrix.ai`. Run before `ingest_qdrant.py`.

### Retrieval baseline 2026-06-22 (filters + bot-name re-rank)

**PASS=5, PARTIAL=0, FAIL=0, SKIP=2**

| Case | Grade | Fix applied |
|------|-------|-------------|
| lookup_01 | PASS | unchanged |
| lookup_02 | PASS | `type=bot` filter + re-rank on `data-loop` hint (fetch 300 candidates) |
| comparison_01 | PASS | unchanged |
| comparison_02 | PASS | unchanged |
| multi_part_01 | PASS | re-rank on `if-condition` hint (chunk was rank ~272 in vector search) |
| negative_* | SKIP | generation eval below |

**Retrieval changes:** `FILTER_BY_CATEGORY` in `eval_set.py`; `bot_name_hints()` + `rerank_by_bot_name()` + expanded candidate pool in `query_qdrant.retrieve()`.

### CFXQL markdown ingest + re-baseline 2026-06-22

- Swapped `data/raw/cfxql_reference.txt` (4 chunks) for `CFXQL_FILE` / `cfxql.md` (13 chunks) via `MarkdownHeaderTextSplitter`
- Re-ingested: **1,847 chunks**; baseline still PASS=5 PARTIAL=0

### Corpus expansion 2026-06-22

- Added `DOCS_ROOT` + `DOCS_INCLUDE_DIRS` in `config.py`; `chunk_narrative_markdown()` + `load_narrative_docs()` in `ingest_qdrant.py`
- `scripts/batch_ingest_narrative.py` -- chunk-only validation (no embed/Qdrant)
- Char-limited embed batches + oversized-chunk splitting (fixes OpenRouter 131k input limit)
- Tightened negative-category abstention prompt in `query_qdrant.build_prompt()`
- Ingested all 8 narrative folders -> **6,331 chunks** in local Qdrant

**Retrieval baseline (full corpus):** PASS=5, PARTIAL=0, SKIP=2 -- no regressions vs 1,847-chunk baseline.

**Generation eval (full corpus, before eval fix):** PASS=5, PARTIAL=1, FAIL=1

| Case | Grade | Notes |
|------|-------|-------|
| multi_part_01 | PARTIAL | 75% -- missing "already-loaded data" fact (LLM variance; same run PASS on re-test) |
| negative_01 | FAIL | Password-reset question now answerable from `dashboards_configuration.md` + install guides |

**Eval drift fix:** Replaced `negative_01` with "How do I cancel my Fabrix.ai subscription?" (billing not in corpus). Updated `negative_02` `expected_source` note (`architecture.md` is ingested but has no worker limit content).

**Generation eval (after eval fix):** PASS=7, PARTIAL=0, FAIL=0 -- multi_part_01 100% facts on re-run, confirms variance not regression.

---

## Calls / blockers

**Ravi & Faizan (Wed 6/17 evening)**

- Shared VPN-accessible Qdrant, Docker, fastembed server-side
- API: create collection, upload, search, delete
- Plan: test embedding models on shared server
- Later: agent judges answer sufficiency, reformulate questions, retry retrieval

**Thu 6/18**

- OpenAI credentials for generation (gpt-4o-mini)
- Remote ingestion blocked on VPN/timeouts for large files (cfxdm.md etc.)
- Local fallback: swap TF-IDF -> MiniLM on full corpus -- confirmed retrieval works at scale

---

## Repo structure (current)

```
fabrix-docs-agent/
├── README.md
├── docs/NOTES.md          <- this file
├── requirements.txt
├── .env                   <- gitignored API keys
├── data/
│   ├── raw/               -> cfxql_reference.txt only (sample bot txts skipped)
│   └── qdrant_db/         -> local vector store (gitignored)
├── src/
│   ├── config.py          -> paths, models, .env loading
│   ├── ingest_qdrant.py   -> chunk + embed + store
│   ├── query_qdrant.py    -> retrieve + generate
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
    └── run_eval.py
```

---

## Changelog

### Wed 2026-06-17

- Got MD docs via VPN (942 files)
- Tested MarkdownHeaderTextSplitter on cfxql.md (13 chunks)
- Built clean_markdown.py + bot catalog chunking; validated cfxdm, kafka, jira
- Batch ingested 214 bot files: 1,834 chunks, 0 errors
- Call with Faizan -- remote Qdrant plan
- Embedding comparison: MiniLM beat BGE-large locally

### Thu 2026-06-18

- Swapped generation to gpt-4o-mini
- TF-IDF retrieval failed at full scale on CFXQL comparison
- Swapped to MiniLM embeddings locally
- Remote ingest blocked on VPN/timeouts

### Fri 2026-06-22

- Safe refactor: unified ingest, config.py, eval baseline, .env loading
- Skipped duplicate sample bot `.txt` files (1,838 chunks)
- Retrieval baseline: PASS=2 PARTIAL=3 after re-ingest
- Per-category top_k -> PASS=3 PARTIAL=2; comparison_02 fixed
- Added `scripts/audit_eval_sources.py` -- all eval bots found in catalog
- Generation eval: negative cases PASS; multi_part_01 FAIL (end-if confusion)
- Housekeeping: gitignore `data/qdrant_db/`, README chunk count 1,838, `docs/NOTES.md`
- Retrieval fixes: category filters, bot-name re-rank, expanded candidate pool -> 5/5 PASS baseline
- Generation re-check: multi_part_01 PARTIAL (correct bot + Full CFXQL; omits end-if detail)
- Generation polish: multi-part prompt instructions, `eval_scoring.py`, `run_eval_generation.py`
- Lookup fix: prune sibling bots + lookup prompt -> 7/7 generation PASS
- Corpus expansion: all 8 narrative folders ingested -> 6,331 total chunks (was 1,847)
- negative_01 swapped out (password reset now in corpus) -> subscription cancel question
- negative_02 expected_source note updated
- Retrieval eval on full corpus: 5/5 PASS
- Generation eval on full corpus: 7/7 PASS
- Committed at 266d55e

---

## Eval runs

### Retrieval baseline runs

Latest automated output: `tests/eval_baseline_results.txt` (gitignored).

**2026-06-22 (per-category top_k):** PASS=3 PARTIAL=2 -- see table in Eval history above.

### Generation eval -- 2026-06-22 (after retrieval fixes)

| Case | Grade | Notes |
|------|-------|-------|
| multi_part_01 | PARTIAL | Correct bot + Full CFXQL; missing end-if requirement and in-memory query detail |
| negative_01 | PASS | Abstains correctly |
| negative_02 | PASS | Abstains correctly |

Takeaway: retrieval fixes unblocked generation on if-condition (was FAIL). Remaining gap on multi_part_01 is generation brevity, not wrong source.

### Generation eval -- 2026-06-22 (automated, after prompt fix)

**PASS=5, PARTIAL=1, FAIL=1** -- see `tests/eval_generation_results.txt`

| Case | Grade | Notes |
|------|-------|-------|
| lookup_01 | FAIL | LLM cited wrong params (`count` vs start/end/increment) despite correct retrieval |
| lookup_02 | PASS | |
| comparison_01 | PASS | |
| comparison_02 | PASS | |
| multi_part_01 | PARTIAL | 75% facts; manual run was PASS with fuller answer |
| negative_01 | PASS | |
| negative_02 | PASS | |

Changes: category-aware prompt for `multi_part` in `query_qdrant.build_prompt()`; shared scoring in `tests/eval_scoring.py`.

### Generation eval -- 2026-06-22 (lookup context pruning)

**PASS=7, PARTIAL=0, FAIL=0**

| Case | Grade | Fix |
|------|-------|-----|
| lookup_01 | PASS | `prune_lookup_chunks()` drops sibling `@exec:count-loop`; lookup prompt uses [1] only |
| lookup_02 | PASS | Companion-params prompt line |
| multi_part_01 | PASS | 100% facts (incl. already-loaded data) |

Changes: `prune_lookup_chunks()` in `run_pipeline`/`ask`; lookup-specific prompt instructions.

### Generation eval -- 2026-06-22 (full corpus, post-expansion)

**PASS=5, PARTIAL=1, FAIL=1** -- first run on 6,331-chunk index

| Case | Grade | Notes |
|------|-------|-------|
| negative_01 | FAIL | Retrieved dashboards_configuration.md password-reset steps; model answered instead of abstaining |
| multi_part_01 | PARTIAL | 75% facts |

### Generation eval -- 2026-06-22 (after eval drift fix)

**PASS=7, PARTIAL=0, FAIL=0**

| Case | Grade | Change |
|------|-------|--------|
| negative_01 | PASS | Question swapped to "How do I cancel my Fabrix.ai subscription?" |
| negative_02 | PASS | expected_source note only |
| multi_part_01 | PASS | 100% facts -- confirms prior PARTIAL was LLM variance |

### Eval expansion -- 2026-06-22 (beginners_guide)

Added 3 `guide` cases to `eval_set.py` with filter `type: narrative` + `doc_section: beginners_guide`, `top_k=5`.

| Case | Source | Retrieval | Generation |
|------|--------|-----------|------------|
| guide_01 | data_ingestion.md (Event Gateway endpoints) | PASS | PASS |
| guide_02 | architecture.md (design principle) | PASS | PASS |
| guide_03 | persistent_streams.md (NATS/Kafka) | PASS | PASS |

**Retrieval:** PASS=8/8 scored (7 original + 3 guide; 2 negative SKIP). **Generation:** PASS=9, PARTIAL=1 (`multi_part_01` variance), FAIL=0 (10 cases).

### Eval expansion -- 2026-06-23 (installation_guides + ai_fabric)

Added 2 narrative cases with per-section filters (`install`, `ai_fabric`), `top_k=5`.

| Case | Source | Retrieval |
|------|--------|-----------|
| install_01 | data_retention.md (daily backup schedule) | PASS |
| ai_01 | llm_pooling.md (LLM pool purpose) | PASS |

**Retrieval baseline (OpenRouter MiniLM + Qdrant):** PASS=**10/10** scored, SKIP=2 (12 cases total).

### Fastembed local comparison -- 2026-06-22 / 2026-06-23

Compared fastembed models on full 6,331-chunk index via [`src/test_fastembed_eval.py`](../src/test_fastembed_eval.py) (in-memory retrieval, same filters/top_k as eval baseline). **10 scored cases** as of 2026-06-23 (includes `install_01`, `ai_01`).

| Model | dim | PASS/10 | Time | Notes |
|-------|-----|---------|------|-------|
| sentence-transformers/all-MiniLM-L6-v2 | 384 | ~7/10 | ~3 min | Prior run (8-case harness); missed `comparison_02` |
| snowflake/snowflake-arctic-embed-xs | 384 | **9/10** | **~19 min** | Miss `comparison_01`; fixed `comparison_02`; new cases PASS |
| **BAAI/bge-small-en-v1.5** | 384 | 8/8 (old) | ~47 min | 10-case rerun optional |
| BAAI/bge-base-en-v1.5 | 768 | **9/10** | **~96 min** | Miss `install_01` PARTIAL (50% facts); fixed `comparison_01` |
| OpenRouter MiniLM + Qdrant (production) | 384 | **10/10** | n/a | Current ingest path |

MiniLM miss (prior): `comparison_02` PARTIAL. arctic miss: `comparison_01` PARTIAL. bge-base miss: `install_01` PARTIAL (source hit, 50% facts). bge-small: 8/8 on old 8-case harness.

**Recommendation:** Keep **OpenRouter MiniLM + Qdrant** for ingest (10/10, fast API). No local fastembed model reached 10/10. Best local tradeoff if needed: **arctic-embed-xs** (9/10, ~19 min) over bge-base (9/10, ~96 min). Log: `/tmp/fastembed_shootout_10.log`.