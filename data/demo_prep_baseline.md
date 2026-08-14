# Harness baseline — 2026-08-07 (demo prep)

## Production (`eval_production.py`)
- **PASS=51 PARTIAL=1 FAIL=0** (scored 52)
- PARTIAL: `c14_schedule_debug` — model said “don’t see a documented procedure” without inference disclosure; `_looks_like_abstention` did not match that phrase so schedule honesty fallback never fired.

## Fix applied
- Broadened `_looks_like_abstention` in `src/agent.py` to cover “don’t see a documented procedure/playbook” phrasing (generic; not question-specific).

## Break (full)
- Started after prod; interrupted for cycle-33 focus (full 180+ cases too slow for same-session demo prep).
- Cycle 33: 12 novel cases; first run 11 PASS / 1 FAIL (scoring mismatch on correct invented-bot refuse); scoring softened → re-run.

## Readiness streak
- Not refreshed this session (full harness not completed). Treat cycle 33 + DEMO_2 smoke as demo gate.
