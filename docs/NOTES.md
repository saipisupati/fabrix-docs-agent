# Technical notes

Architecture and evaluation notes for the Fabrix docs agent. For setup and commands see [README.md](../README.md).

---

## Pipeline

| Stage | Component |
|-------|-----------|
| Ingest | `ingest_qdrant.py` → OpenRouter MiniLM → local Qdrant (`data/qdrant_db/`) |
| KB | `build_kb.py` → `data/kb/kb.json` + embeddings |
| Agent | `agent.py` → merged scope+facets, KB-first retrieve, optional critique → gpt-4o-mini |
| API | `api.py` → `POST /ask`, `GET /health` |

**Corpus:** 543 public MD files → ~6,387 chunks (214 bots + 328 narrative + CFXQL + 3 root pages).

**Answer contract:** unified `answer` + `examples` + `gaps` + `sources` + `used_inference` + optional `timing`. Path-first shape: `**Documented Fabrix path**` + numbered steps + `Next (inferred)` for gaps.

---

## Quality gate

Run `python3 tests/run_quality_harness.py` (stop API first). See [QUALITY_LOOP.md](QUALITY_LOOP.md).

| Suite | Purpose |
|-------|---------|
| `eval_production.py` | Production-style ops battery (22 cases) |
| `eval_break.py` | Adversarial battery (27 cases) |
| `eval_readiness.py` | PASS rate + p95 latency (14-case subset) |

Latest harness bar: production 100%, break ≥95%/0 FAIL, readiness GREEN ×2.

---

## Key decisions

| Decision | Rationale |
|----------|-----------|
| Qdrant + classic RAG | Standard chunk → embed → retrieve → generate |
| MiniLM via OpenRouter | Strong retrieval on eval set; fast API |
| gpt-4o-mini | Stable grounded generation |
| Header-based CFXQL chunking | Size-based splitting broke comparison answers |
| One chunk per bot (`##` headers) | 214 bot files → 1,834 chunks |
| Public MD export as source | Audited via `scripts/audit_ingest_sources.py` |
| KB-first + chunk fallback | Structured facts/procedures before raw chunks |
| Integration-family fidelity | Generic bot/source filtering by named products |

---

## Architecture paths

| Path | Ingest | Embed | Store |
|------|--------|-------|-------|
| **Local (primary)** | `ingest_qdrant.py` | MiniLM via OpenRouter | local Qdrant |
| Remote (optional) | `ingest_and_test_remote.py` | BGE-large (server) | hosted Qdrant via `REMOTE_BASE_URL` |
| Legacy | `ingest.py` / `query.py` | TF-IDF | Chroma |

---

## Technical learnings

**Chunking:** header-based splits for CFXQL and narrative; bot catalog split on `##` (one chunk per bot).

**Retrieval:** category filters, bot-name re-rank, lookup chunk pruning fixed early eval misses.

**Agent:** merged scope+facet LLM call; skip critique when draft checks pass; anti-abstain when docs retrieved; synthesis latch for compare/recommend asks.

**Deploy:** copy `data/qdrant_db/` and `data/kb/` or rebuild on server. Local Qdrant allows one process at a time.

---

## Eval scripts

| Script | Layer |
|--------|-------|
| `run_eval_baseline.py` | Retrieval only |
| `run_eval_generation.py` | Full pipeline (oracle hints) |
| `run_eval_agent.py` | Agent path (no hints) |
| `run_eval_kb.py` | Scope / inference / examples / gaps |
| `run_quality_harness.py` | Raised bar (all suites + streak) |

Case definitions: `eval_set.py`, `eval_production.py`, `eval_break.py`.
