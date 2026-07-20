# Demo 2 — what got better

- **Who it’s for:** show what’s improved since v1 (not a full product tour)
- **Time:** ~10–12 min
- **UI:** chat at http://localhost:5173/ (API on :8080)
- **Headline:** The agent stays on topic, says when docs are thin, and doesn’t invent answers — and we test that on purpose.

---

## Before you start

```bash
cd fabrix-docs-agent
source venv/bin/activate
uvicorn src.api:app --port 8080          # terminal 1
python3 -m http.server 5173 --directory chat   # terminal 2
# open http://localhost:5173/
```

- Check `/health` → `{"status":"ok"}`
- Don’t run the quality tests while the API is up (they fight over the same local database)

---

## Opening (30–60 sec)

- First version: search the public Fabrix docs and answer — useful, but could mix products and sound too sure
- Since then: a cleaner knowledge base, clearer “what’s documented vs guessed,” and automated stress tests
- Line to say: *Here’s the documented path; here’s what we inferred; here’s what the docs don’t cover*

---

## What changed since demo 1 (say this)

### Cleaner knowledge base

- **Before:** Answers came mostly from raw doc snippets (find similar text)
- **Now:** We also build a structured knowledge base (`data/kb/`) — topics, facts, steps, examples from the same public docs
- Agent looks at the knowledge base first, then falls back to snippets if needed
- That means clearer answers (named bots, steps, examples) instead of only “nearby paragraphs”
- Same public Fabrix docs (~543 files); rebuild the knowledge base after docs change with `build_kb.py`

### Better agent behavior

- **Stay on topic:** a Zabbix question doesn’t wander into ServiceNow or Linux inventory
- **Honesty:** marks **inferred** tips; shows **What the docs don’t cover**
- **Refuse junk:** won’t invent SLAs, compliance answers, quotes, or passwords
- **Slang / typos:** SN, PD, SNOW, and common misspellings still land on the right product
- **No made-up bots:** bot names must show up in the docs we retrieved
- **Answer shape:** documented path → steps → optional “next (inferred)” tips

### How we keep it from slipping

- Everyday test set + hard “break” questions (meant to trip it up)
- When something fails: add a test → one general fix (not a one-off for that question) → re-run tests
- How-to: `QUALITY_LOOP.md`, `CONTINUOUS_QUALITY.md`

---

## Then vs now (one breath)

- **v1:** Snippet search only; could mix products; sounded sure when docs were thin; mostly manual checks
- **Now:** Knowledge base first + snippets; stays on topic; marks guesses and gaps; refuses traps; automated stress tests

---

## Live questions (show the improvements)

- After each answer, point at: **Examples → Sources → What the docs don’t cover**
- **Still works for basics**
  - `What parameters does the timed-loop bot take?`
  - Quick, grounded bot answer
- **Ops path + honesty**
  - `How do I get SN ticketing into a Fabrix stream for downstream bots?`
  - Show documented path, sources, gaps / inferred line
- **Stay on topic (big win)**
  - `How do I add Zabbix as a datasource? Our collector also runs on a Linux host if that matters.`
  - Zabbix only; Linux is a distraction
- **Refuse junk (trust)**
  - `What is Fabrix's contractual P1 support SLA response time?`
  - Says it doesn’t know; no fake sources
- **Gaps money shot**
  - `Use Fabio Copilot to rewrite my Zabbix pipeline in production for me end-to-end.`
  - Answers what docs support; gaps say full rewrite isn’t documented
- **Optional slang (if time)**
  - `PD alerts into Fabrix then into a dataset for dashboards — walk me through it.`
  - PD means PagerDuty

---

## Closing (30 sec)

- Quality is a loop: real miss → test case → one general fix → re-run
- We don’t hardcode answers for specific questions
- Local tests are green; putting this on the public docs site / full CI is a later decision

---

## Evidence (don’t run live unless asked)

- Everyday tests: 34 cases
- Hard break tests: cycles 1–6
- Latest: 34/34 everyday; cycle 6 break 12/12
- Playbooks: `docs/QUALITY_LOOP.md`, `docs/CONTINUOUS_QUALITY.md`

---

## Skip today

- CI / private data URL setup
- Embedding on the public docs site / production hosting
- Obscure or made-up bot names
- Running tests and the API at the same time

## If something breaks

- Fall back to timed-loop or CFXQL compare
- If the local database is locked: stop the API or the tests, then retry
