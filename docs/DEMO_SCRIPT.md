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

Run each from project root. After each answer, point at the **answer** and the **source links** at the bottom.

**Bot lookup (easy win)**

```bash
python3 src/agent.py "What parameters does the count loop bot take?"
```

- Should mention `name`, `start`, `end`, `increment`
- Source should land on the count-loop bot page

**CFXQL depth**

```bash
python3 src/agent.py "What is the difference between Full and Restricted CFXQL?"
```

- Full: more operators, Result Format / GET clause
- Restricted: basically `=` and `AND`, no Result Format

**Platform guide**

```bash
python3 src/agent.py "What endpoint types does the RDA Event Gateway support for data ingestion?"
```

- Should hit syslog, http, filebeat, etc.

**Integration**

```bash
python3 src/agent.py "What ServiceNow modules does Fabrix AIOps integrate with?"
```

- Ticketing, CMDB, that kind of thing

**Honesty check (important)**

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
python3 tests/run_eval_agent.py
```

- 20 cases: bots, guides, install, pipelines, integrations, negative/hallucination
- Don't run live unless you've run it recently (takes a few min)

Or just say: "We have a fixed eval set in `tests/eval_set.py`, last full agent eval was 20/20."

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

> "Corpus is complete, accuracy is tested, widget and API are ready. We just need hosting and a slot in the docs site."

---

## Skip during demo

- Obscure or made-up bot names
- Very specific dataset slugs (router can misroute)
- Remote/VPN path unless you're sure it's up
- Running eval + API at the same time (local Qdrant file lock)

## If something breaks

- "Let me try a simpler phrasing" → fall back to count-loop or CFXQL question
- If Qdrant lock error: stop API/eval, retry CLI only
