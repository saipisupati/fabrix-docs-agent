# Demo script (loose)

~12 min run-of-show for Dheeraj. Use CLI, not the widget, unless you've already verified API + Qdrant locally.

**Before you start**

```bash
cd fabrix-docs-agent
source venv/bin/activate   # if you use venv
```

---

## 1. Open (1 min)

- "Built a docs Q&A agent over the full public Fabrix corpus"
- "543 source files, ~6,400 chunks, answers cite docs.fabrix.ai"
- "Tested on 20 hand-picked questions, last run was 20/20 pass"

---

## 2. Live questions (7 min)

Prefer the **chat UI** (API on :8080 + `python3 -m http.server 5173 --directory chat`) so they see elapsed time, Cancel, sources, and gaps. CLI works too.

**Bot lookup (easy win)**

```bash
python3 src/agent.py "What parameters does the count loop bot take?"
```

- Should mention `name`, `start`, `end`, `increment`
- Source should land on the count-loop bot page
- CLI footer shows `[timing] … llm_calls=0` on the fast path

**CFXQL depth**

```bash
python3 src/agent.py "What is the difference between Full and Restricted CFXQL?"
```

- Full: more operators, Result Format / GET clause
- Restricted: basically `=` and `AND`, no Result Format

**Ops synthesis (path-first)**

```bash
python3 src/agent.py "How do I get SN ticketing into a Fabrix stream for downstream bots?"
```

- ServiceNow → stream/pstream path; may show inference footer
- Numbered steps should be real steps (not “1. Documented Fabrix path”)

**Day-2 compare**

```bash
python3 src/agent.py "When should an engineer use a persistent stream instead of a regular dataset in RDA Fabric?"
```

**Honesty check (important)**

```bash
python3 src/agent.py "What is Fabrix's contractual P1 support SLA response time?"
```

- Should abstain / out of scope — must NOT invent an SLA number

Also fine:

```bash
python3 src/agent.py "How do I cancel my Fabrix.ai subscription?"
```

- Should say it couldn't find this in the docs
- Should NOT invent a cancellation flow

---

## 3. Optional extras (if he's engaged)

```bash
python3 src/agent.py "What does the if-condition bot do, and what kind of CFXQL does it expect?"
```

```bash
python3 src/agent.py "What example datasets are available for pipelines?"
```

```bash
python3 src/agent.py "What is a primary design principle of RDA Fabric architecture?"
```

---

## 4. Accuracy proof (2 min, optional)

If he asks "how do you know it's accurate?":

```bash
# Stop API first (Qdrant lock)
python3 tests/eval_production.py
python3 tests/eval_readiness.py
```

- Production: 17 ops + break regressions
- Readiness: PASS rate + p95 latency gate (must be GREEN before any docs embed)
- Or mention break battery: `python3 tests/eval_break.py`

Or just say: "We gate on production + readiness evals; we are not embedding until readiness is green twice."

---

## 5. What goes on the site (2 min)

- Show `widget/ask-widget.js` + `widget/ask-widget.css`
- Show `docs/DOCS_SITE_INTEGRATION.md` (two lines of HTML to embed)
- "Widget talks to our API, API holds the keys and Qdrant, nothing secret in the browser"
- "Not live yet, need API host + docs repo access"

---

## 6. Close, ask him (2 min)

- "Does accuracy look good enough for v1?"
- "Any questions you'd want in the eval set?"
- "Where should we host the API?"
- "Can we get docs repo access to embed the widget?"

**One-liner to end on:**

> "Accuracy and latency are gated locally. We are not putting this on the docs site until the readiness gate is green twice in a row — then we need hosting and a widget slot."

---

## Skip during demo

- Obscure or made-up bot names
- Very specific dataset slugs (router can misroute)
- Remote/VPN path unless you're sure it's up
- Running eval + API at the same time (local Qdrant file lock)

## If something breaks

- "Let me try a simpler phrasing" → fall back to count-loop or CFXQL question
- If Qdrant lock error: stop API/eval, retry CLI only
