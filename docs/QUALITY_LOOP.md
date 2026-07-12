# Quality loop (step-by-step)

Closed-loop process to improve the Fabrix docs agent until a **raised bar** holds. Fixes must be **generic** (families, fidelity, honesty, retrieve bias), never question-specific branches in `src/agent.py`.

## Raised bar

| Suite | Requirement |
|-------|-------------|
| `tests/eval_production.py` | 100% PASS (0 FAIL, 0 PARTIAL) |
| `tests/eval_break.py` (full cycle 1+2) | ≥95% PASS and **0 FAIL** |
| `tests/eval_readiness.py` | GREEN (pass ≥95%, p95 ≤ 45s) **twice in a row** |

Harness exit **0** only when all three hold (including readiness streak ≥ 2).

## Step 0: Prep

```bash
cd fabrix-docs-agent
source venv/bin/activate
set -a && source .env && set +a
```

- Stop anything on `:8080` (local Qdrant allows one process at a time):
  ```bash
  kill $(lsof -tiTCP:8080 -sTCP:LISTEN) 2>/dev/null
  ```
- Confirm `data/qdrant_db/` and `data/kb/kb.json` exist.

## Step 1: Run harness

```bash
python3 tests/run_quality_harness.py
```

Reads/writes (gitignored):

- `tests/quality_harness_digest.md`: suite table, FAIL/PARTIAL cases, suggested fix categories, verdict
- `tests/readiness_streak.json`: consecutive GREEN count

Note `readiness_streak=N/2` in the digest footer.

## Step 2: Triage (if exit 1)

1. Open the digest.
2. Group FAILs by tag (`contamination`, `overclaim`, `wrong_facet`, …).
3. Pick **one** fix category per round (avoid shotgun patches).

| Tag / attack | Generic fix area |
|--------------|------------------|
| contamination | `INTEGRATION_FAMILIES` / bot fidelity / source filter |
| overclaim | honesty in prompt + critique; Next (inferred) / gaps |
| wrong_facet | retrieve seeds / facet plan / path-first shape |
| abstain_fail | scope latch / topic seeds / anti-abstain |
| thin_wiring | wiring shape + critique sink check |
| latency (readiness RED on p95) | skip clean critique / fewer LLM calls |

## Step 3: Implement + re-run

- Edit `src/agent.py` (or rebuild KB/ingest only if retrieve is empty).
- Re-run Step 1.
- If readiness alone goes RED once, streak resets; you need two consecutive GREENs again.

## Step 4: Lock regressions

When a new failure appears:

1. Add a case to `eval_break.py` / `eval_production.py` with `need_any` / `forbid_*`.
2. Do **not** special-case that question in the agent.
3. Re-run the harness.

## Step 5: Done

Harness exit 0 + `readiness_streak≥2` → ready for production deploy decision. Restart the API when finished:

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8080
```
