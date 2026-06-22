# Fabrix Docs Agent — project notes and changelog

Living log of research, findings, meetings, and eval runs. For setup and commands see [README.md](../README.md).

---

## Product / doc structure

### Bot catalog pages

Each extension page groups multiple bots. Every bot entry follows the same pattern:

- Short description of what it does
- Which CFXQL type it needs (Full or Restricted)
- Parameter list for that bot

Example from [docs.fabrix.ai/Bots/search_bots/](https://docs.fabrix.ai/Bots/search_bots/):

**Bot @c:timed-loop** — starts a looping block that waits x seconds before running again.

Parameter table: name, type, default, description — e.g. `interval`, `stop_after`, `max_iterations`.

Also shows example pipelines using the bot.

### Bot type prefixes

| Prefix | Type | CFXQL | Role |
|--------|------|-------|------|
| `#` | Source filtering | Full | Translate query into remote data source API/DB call |
| `*` | Destination filtering | Full | Filter data already in memory/pipeline |
| `@` | API | Restricted | Extract API params that control bot behavior |

### CFXQL reference

**Full CFXQL** — SQL-like language for what each bot does with data:

- Filter/condition clause: `is`, `not in`, `!=`, AND/OR grouping
- GET clause: select/rename output columns
- Used by bots working on already-loaded data frames

**Restricted CFXQL** — simpler; each param specified via `=`

---

## Original plan (4 phases)

1. Start with bot catalog + CFXQL — basic retrieve-and-answer pipeline end-to-end
2. Add citations to source pages + mock test questions for accuracy
3. Improve chunking for pipeline/narrative docs
4. Agent routes questions to product area; handles multi-part questions; integrate with docs site

Later ideas: generate docs, verify against screenshots, graph DB for file relationships, version hashing for doc updates.

---

## Open questions

- **RAG + vector DB?** Yes — current approach is classic RAG (chunk → embed → store → retrieve → generate)
- **Framework?** LangChain used for chunking utilities; LlamaIndex not evaluated deeply
- **Embedding model?** Mistral embed tested; **MiniLM-L6-v2** won local comparison; generation via **gpt-4o-mini** (OpenAI)
- **Doc source?** Internal MD export via VPN (942 files), not web scraper — ask for documentation files, convert to MD
- **Repo/org conventions?** TBD
- **Dev environment?** Python venv; Faizan's server is Docker + fastembed

Main task from stakeholders: **find the right tool/framework for cost and accuracy** — CFX has Qdrant, start there; test OpenRouter free models.

---

## Architecture evolution

| Stage | Ingest | Embed | Store | Generate |
|-------|--------|-------|-------|----------|
| v0 | `ingest.py` | TF-IDF | Chroma | stub |
| v1 | `ingest_with_real_embeddings.py` | TF-IDF → MiniLM | Qdrant | openrouter/free |
| v2 (current) | `ingest_qdrant.py` | MiniLM via OpenRouter | Qdrant | gpt-4o-mini |
| remote (WIP) | `ingest_and_test_remote.py` | BGE-large (server) | Faizan's Qdrant | TBD |

Corpus: 214 bot `.md` files → ~1,834 bot chunks + 13 CFXQL chunks (`cfxql.md`) = **~1,847 total**.

---

## Chunking learnings

1. **Size-based splitting breaks CFXQL** — 11 fragmented chunks, all tagged generic `reference`; comparison answers were wrong/backwards
2. **Hand-rolled section splits fix CFXQL** — split at `Full CFXQL` / `Restricted CFXQL` headers → 4 clean chunks with correct `cfxql_type` metadata; flipped wrong answer to correct **without changing embeddings or LLM**
3. **Markdown-aware splitting generalizes** — `MarkdownHeaderTextSplitter` on real `cfxql.md` → 13 chunks with `{h2, h4}` metadata, zero false positives
4. **Bot catalog** — `clean_markdown.py` strips YAML/CSS/HTML; split on `##` → one chunk per bot; validated 214 files, 0 errors, 1,834 chunks
5. **Prefix distribution** — @ 84%, * 14%, # 2%
6. **Generalization options** — markdown headers (best), plain-text heuristics (noisy), recursive splitter tuning, LLM chunking (expensive)

---

## Embedding learnings

- TF-IDF works on tiny samples, **fails at scale** (~1,800+ chunks) — conceptual CFXQL questions retrieve unrelated bots
- Real embeddings (MiniLM, Mistral, Nemotron) correctly rank Full/Restricted CFXQL chunks
- **Surprise:** MiniLM-L6-v2 (smallest/cheapest) avg rank **1.00** on 3 CFXQL questions; BGE-large tied worst at **2.00**
- Free OpenRouter models unstable (404/429); `openrouter/free` auto-router worked for early tests

---

## Eval history

### Early run (small corpus, TF-IDF, section-aware CFXQL, openrouter/free)

6/7 cases graded manually — mostly PASS including comparison_01/02, negative cases, multi_part_01 (minor end-if omission).

### Retrieval baseline 2026-06-22 (before sample-txt skip)

PASS=2, PARTIAL=3 — `c_extension_loop_bots.txt` sample file outranked real bot chunks.

### Retrieval baseline 2026-06-22 (after sample-txt skip, top_k=5)

PASS=2, PARTIAL=3 — lookup_01 fixed (@c:count-loop rank 1); lookup_02 and multi_part_01 still PARTIAL (bots may be missing from markdown catalog).

### Retrieval baseline 2026-06-22 (per-category top_k: lookup=5, comparison=10, multi_part=8)

**PASS=3, PARTIAL=2, FAIL=0, SKIP=2**

| Case | Grade | Notes |
|------|-------|-------|
| lookup_01 | PASS | @c:count-loop rank 1 |
| lookup_02 | PARTIAL | @c:data-loop not in top-5 despite being in catalog |
| comparison_01 | PASS | |
| comparison_02 | **PASS** | Fixed by comparison top_k=10 (was PARTIAL at top_k=5) |
| multi_part_01 | PARTIAL | *exec:if-condition not in top-8; @exec:end-if rank 1 |
| negative_* | SKIP | Graded via generation eval below |

**Catalog audit (same day):** All three eval bot names exist in `BOTS_DIR` — `@c:count-loop` and `@c:data-loop` in `control.md`, `*exec:if-condition` in `exec.md`. Lookup/multi_part PARTIAL grades are retrieval ranking issues, not missing docs.

### Retrieval baseline 2026-06-22 (filters + bot-name re-rank)

**PASS=5, PARTIAL=0, FAIL=0, SKIP=2**

| Case | Grade | Fix applied |
|------|-------|-------------|
| lookup_01 | PASS | unchanged |
| lookup_02 | **PASS** | `type=bot` filter + re-rank on `data-loop` hint (fetch 300 candidates) |
| comparison_01 | PASS | unchanged |
| comparison_02 | PASS | unchanged |
| multi_part_01 | **PASS** | re-rank on `if-condition` hint (chunk was rank ~272 in vector search) |
| negative_* | SKIP | generation eval below |

**Retrieval changes:** `FILTER_BY_CATEGORY` in `eval_set.py`; `bot_name_hints()` + `rerank_by_bot_name()` + expanded candidate pool in `query_qdrant.retrieve()`.

### CFXQL markdown ingest + re-baseline 2026-06-22

- Swapped `data/raw/cfxql_reference.txt` (4 chunks) for `CFXQL_FILE` / `cfxql.md` (13 chunks) via `MarkdownHeaderTextSplitter`
- Re-ingested: **1847 chunks**; baseline still **PASS=5 PARTIAL=0**

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
- Local fallback: swap TF-IDF → MiniLM on full corpus — confirmed retrieval works at scale

---

## Repo structure (current)

```
fabrix-docs-agent/
├── README.md
├── docs/NOTES.md          ← this file
├── requirements.txt
├── .env                   ← gitignored API keys
├── data/
│   ├── raw/               → cfxql_reference.txt only (sample bot txts skipped)
│   └── qdrant_db/         → local vector store (gitignored)
├── src/
│   ├── config.py          → paths, models, .env loading
│   ├── ingest_qdrant.py   → chunk + embed + store
│   ├── query_qdrant.py    → retrieve + generate
│   ├── ingest_and_test_remote.py
│   └── ...
├── scripts/
│   └── audit_eval_sources.py
└── tests/
    ├── eval_set.py
    ├── run_eval_baseline.py
    └── run_eval.py
```

---

## Changelog

### Wed 2026-06-17

- Got MD docs via VPN (942 files)
- Tested MarkdownHeaderTextSplitter on cfxql.md (13 chunks)
- Built clean_markdown.py + bot catalog chunking; validated cfxdm, kafka, jira
- Batch ingested 214 bot files: 1,834 chunks, 0 errors
- Call with Faizan — remote Qdrant plan
- Embedding comparison: MiniLM beat BGE-large locally

### Thu 2026-06-18

- Swapped generation to gpt-4o-mini
- TF-IDF retrieval failed at full scale on CFXQL comparison
- Swapped to MiniLM embeddings locally
- Remote ingest blocked on VPN/timeouts

### Fri 2026-06-22

- Safe refactor: unified ingest, config.py, eval baseline, .env loading
- Skipped duplicate sample bot `.txt` files (1838 chunks)
- Retrieval baseline: PASS=2 PARTIAL=3 after re-ingest
- Per-category top_k → PASS=3 PARTIAL=2; comparison_02 fixed
- Added `scripts/audit_eval_sources.py` — all eval bots found in catalog
- Generation eval: negative cases PASS; multi_part_01 FAIL (end-if confusion)
- Housekeeping: gitignore `data/qdrant_db/`, README chunk count 1838, `docs/NOTES.md`
- Retrieval fixes: category filters, bot-name re-rank, expanded candidate pool → **5/5 PASS** baseline
- Generation re-check: multi_part_01 PARTIAL (correct bot + Full CFXQL; omits end-if detail)

---

## Eval runs

### Retrieval baseline runs

Latest automated output: `tests/eval_baseline_results.txt` (gitignored).

**2026-06-22 (per-category top_k):** PASS=3 PARTIAL=2 — see table in Eval history above.

### Generation eval — 2026-06-22 (after retrieval fixes)

| Case | Grade | Notes |
|------|-------|-------|
| multi_part_01 | **PARTIAL** | Correct bot + Full CFXQL; missing end-if requirement and in-memory query detail |
| negative_01 | **PASS** | Abstains correctly |
| negative_02 | **PASS** | Abstains correctly |

**Takeaway:** Retrieval fixes unblocked generation on if-condition (was FAIL). Remaining gap on multi_part_01 is generation brevity, not wrong source.
