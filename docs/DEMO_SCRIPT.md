# Demo script

~12 min walkthrough of the Fabrix docs agent. Use the chat UI (API on :8080 + static server on :5173) or CLI.

**Before you start**

```bash
cd fabrix-docs-agent
source venv/bin/activate
uvicorn src.api:app --port 8080   # terminal 1
python3 -m http.server 5173 --directory chat   # terminal 2
```

---

## 1. Open (1 min)

- RAG agent over the public Fabrix documentation corpus
- 543 source files, ~6,400 chunks, answers cite documentation sources
- Gated by production + break + readiness evals (see `tests/run_quality_harness.py`)

---

## 2. Live questions (7 min)

**Bot lookup**

```bash
python3 src/agent.py "What parameters does the count loop bot take?"
```

- Should list `interval`, `start`, `end`, `increment`
- Fast path: `llm_calls=0`

**CFXQL**

```bash
python3 src/agent.py "What is the difference between Full and Restricted CFXQL?"
```

**Ops synthesis**

```bash
python3 src/agent.py "How do I get SN ticketing into a Fabrix stream for downstream bots?"
```

**Compare**

```bash
python3 src/agent.py "When should an engineer use a persistent stream instead of a regular dataset in RDA Fabric?"
```

**Honesty / abstain**

```bash
python3 src/agent.py "What is Fabrix's contractual P1 support SLA response time?"
python3 src/agent.py "How do I cancel my Fabrix.ai subscription?"
```

- Should abstain; must not invent SLA or billing flows

---

## 3. Accuracy proof (2 min, optional)

Stop API first (Qdrant file lock):

```bash
python3 tests/run_quality_harness.py
```

- Production 22/22, break 27/27, readiness GREEN ×2 when bar is met

---

## 4. Widget embed (2 min)

- Show `widget/ask-widget.js` + `widget/ask-widget.css`
- Show [DOCS_SITE_INTEGRATION.md](DOCS_SITE_INTEGRATION.md)
- API holds keys and Qdrant; nothing secret in the browser

---

## Skip during demo

- Obscure or made-up bot names
- Running eval + API at the same time (local Qdrant file lock)

## If something breaks

- Fall back to count-loop or CFXQL question
- If Qdrant lock error: stop API/eval, retry CLI only
