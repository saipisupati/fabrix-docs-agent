# Demo script

~12 min walkthrough of the Fabrix docs agent. Use the chat UI (API on :8080 + static server on :5173) or CLI.

**Before you start**

```bash
cd fabrix-docs-agent
source venv/bin/activate
uvicorn src.api:app --port 8080   # terminal 1
python3 -m http.server 5173 --directory chat   # terminal 2
# open http://localhost:5173/
```

Talking point: *We separate documented path vs inference, and call out gaps instead of faking certainty.*

For **Demo 2** (what improved since v1), use [DEMO_2.md](DEMO_2.md).

---

## 1. Open (1–2 min)

- RAG agent over the public Fabrix documentation corpus
- 543 source files → ~6,400 chunks in Qdrant **plus** a structured KB (`data/kb/`)
- Gated by production + break + readiness evals (`tests/run_quality_harness.py`)

### Since the first demo (bullets)

**KB shift**
- Moved from chunk-only retrieval to **KB-first** (structured topics/facts/procedures), with chunk fallback
- Same public docs; cleaner ops paths and examples

**Agent improvements**
- Stay on the named product (family fidelity — no cross-contamination)
- Honesty: inferred labeling + **What the docs don’t cover**
- Abstain on SLA / compliance / secrets / quotes
- Slang/typos (SN, PD, SNOW); no invented bot names
- Automated break/fix loop so demos don’t regress

---

## 2. Live questions (8 min)

Point at the chat UI: answer bubble, then **Examples → Sources → What the docs don’t cover**.

**Bot lookup (fast, grounded)**

```
What parameters does the timed-loop bot take?
```

- Lists parameters from the bot catalog
- Often `llm_calls=0` on the fast path

**CFXQL compare**

```
What is the difference between Full and Restricted CFXQL?
```

- Full vs Restricted; may show inference disclosure and a gap if examples are thin

**Ops synthesis (path + inferred + gaps)**

```
How do I get SN ticketing into a Fabrix stream for downstream bots?
```

- Documented path with ServiceNow-related bots/sources
- Look for **Next (inferred)** / disclosure and **What the docs don’t cover**

**Compare**

```
When should an engineer use a persistent stream instead of a regular dataset in RDA Fabric?
```

**Stay on topic (contamination)**

```
How do I add Zabbix as a datasource? Our collector also runs on a Linux host if that matters.
```

- Zabbix path; should not drift into linux-inventory / ServiceNow

**Honesty / abstain**

```
What is Fabrix's contractual P1 support SLA response time?
```

- Abstains; empty sources (billing/SLA are out of public docs)

**Gaps money shot (capability overclaim)**

```
Use Fabio Copilot to rewrite my Zabbix pipeline in production for me end-to-end.
```

- Mentions Fabio + Zabbix from docs where possible
- **What the docs don’t cover** should say the end-to-end rewrite is not documented

---

## 3. Accuracy proof (2 min, optional)

Stop API first (Qdrant file lock):

```bash
python3 tests/run_quality_harness.py
```

- Expect exit 0: production 22/22, break ≥95% / 0 FAIL (38 cases with cycle 3), readiness GREEN ×2

---

## 4. After a miss (habit)

See [QUALITY_LOOP.md](QUALITY_LOOP.md): add an eval case → one generic agent fix → re-run harness. Never hardcode a single question in `src/agent.py`.

---

## 5. Widget embed (optional, 2 min)

- Show `widget/ask-widget.js` + `widget/ask-widget.css`
- Show [DOCS_SITE_INTEGRATION.md](DOCS_SITE_INTEGRATION.md)
- API holds keys and Qdrant; nothing secret in the browser

---

## Skip during demo

- Obscure or made-up bot names
- Running eval + API at the same time (local Qdrant file lock)
- CI / `RUNTIME_DATA_URL` wiring (local harness is the gate)

## If something breaks

- Fall back to timed-loop or CFXQL question
- If Qdrant lock error: stop API/eval, retry CLI only
