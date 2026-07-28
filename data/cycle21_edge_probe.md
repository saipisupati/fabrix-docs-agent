# Cycle 21 edge probe — agent vs docs-grounded (Claude)

**Baseline:** `e0ec565` (cycle 20). Raw agent: `data/cycle21_edge_probe.json`  
**Method:** Local agent answers + grounded answers from live `docs.fabrix.ai` (no prior chat context).

---

## Q1 — two branches write same dataset, who wins?

**Agent**
> Last write wins / last branch execution overwrites. Design carefully; use separate datasets. Gaps: merge strategies not documented. *(inferred)*

**Claude (docs-grounded)**
> Public docs do **not** state a branch-level “who wins” rule for two pipeline branches writing the same dataset. Related: concurrent/multi-proc pipeline execution exists (`@exec:run-pipeline-multi-proc`, etc.), and datasets are saved via bots like `@dm:save`, but conflict/merge semantics for dual writers are not specified. Honest answer: **not documented** — don’t invent last-write-wins.

---

## Q2 — dashboard two tenants side by side?

**Agent**
> Docs don’t explicitly support side-by-side multi-tenant dashboards. Suggests bots → pipelines → datasets → dashboard as a possible approach; combining into one view not outlined. *(inferred)*

**Claude (docs-grounded)**
> No public doc found that says a dashboard can show two tenants side-by-side. Dashboards bind to datasets/pstreams; multi-tenant UI semantics aren’t spelled out. Honest: **not specified**. Don’t invent a multi-tenant combine workflow.

---

## Q3 — max dataset size before performance degrades?

**Agent**
> I couldn't find that in the documentation.

**Claude (docs-grounded)**
> No general “max dataset size before degrade” number in public docs. Playground has *pipeline* limits (e.g. 100 bots, 1 min), not a production dataset size threshold. Honest: **not documented** (match).

---

## Q4 — bots parallel by default?

**Agent**
> **Documented Fabrix path:** sequential by default; parallel needs special config / parallelization bots. Gaps: exact config not specified. *(inferred path label)*

**Claude (docs-grounded)**
> A normal pipeline is a **sequence of bots**. Explicit concurrency is documented via exec bots such as `@exec:run-pipeline-multi-proc` / `run-pipeline-by-row-multi-proc` (`num_procs` 0–50). So: **in-pipeline steps are ordered; parallel is opt-in via specific bots**, not a vague “configure parallelism.” Prefer naming those bots over a generic Documented path.

---

## Q5 — delete pstream also delete datasets?

**Agent**
> Confident **no** — datasets independent; manage manually. Gap: cascade not documented in detail. *(inferred)*

**Claude (docs-grounded)**
> Pstream delete docs say: delete config only, or optionally **“Delete all the data as well”** (OpenSearch index for that pstream). That is about **pstream/index data**, not “datasets that reference the pstream.” Cascade to object-store datasets is **not clearly documented**. Better: describe the documented delete options; say dataset cascade is unspecified — don’t assert a hard no.

---

## Q6 — version-control pipelines in Git?

**Agent**
> No explicit Git integration; toolsets/prompts have versions; suggest manual export to Git. *(inferred)*

**Claude (docs-grounded)**
> Studio has **pipeline versions / Version History / Publish**. SDK examples treat pipelines as `.rda` files runnable via `rdac`. **No documented first-class Git sync** for portal pipelines. Honest: built-in versioning yes; Git = bring-your-own / file export, not a product Git feature.

---

## Q7 — Toolset vs Extension?

**Agent**
> Toolset = YAML MCP tools for agents (strong). Extension = vague “broader module/integration.” *(Extension half fuzzy)*

**Claude (docs-grounded)**
> **Toolset** (agentic): YAML grouping of MCP tools (runPipeline, streamQuery, RESTAPI, …) — what a Copilot/agent can do.  
> **Extension** (classic RDA): installable integration package (e.g. `splunk`, `kafka-v2`, `aws_v2`) that ships **bots** (`@family:op`) and credentials.  
> Different layers: agent capabilities vs bot catalog packages.

---

## Q8 — staging/sandbox separate from prod?

**Agent**
> I couldn't find that in the documentation.

**Claude (docs-grounded)**
> No documented “staging env twin of production.” Adjacent concepts: **RDA Playground** (limited learning sandbox) and **Staging Area based ingestion** (ingestion feature, not an env). Honest: no product staging environment page; mention Playground if teaching/sandbox is the intent; don’t invent DIY staging as required.

---

## Q9 — how know bot is deprecated?

**Agent**
> Check Bot Catalog / bot pages / release notes. *(reasonable but generic)*

**Claude (docs-grounded)**
> Extension list pages mark **Deprecated** with replacement (e.g. `aws` → `aws_v2`, `azure` → `azure_v2`). So: look for **Deprecated** on the extension page / catalog, plus recommended `*_v2` successor. Stronger than generic “check release notes.”

---

## Q10 — two bots same name in different pipelines?

**Agent**
> I couldn't find that in the documentation.

**Claude (docs-grounded)**
> Catalog bots are **shared by name** (`@family:op`) — the same bot is normally reused across many pipelines. SDK: bot `name` must be **unique within a bot package**. So clarify the ask: reusing a catalog bot in two pipelines = yes/normal; defining two custom bots with the same name in one package = no.

---

## Quick reference (for your judging)

| # | Theme | Agent tendency |
|---|--------|----------------|
| 1 | Concurrent write semantics | Invented last-write-wins |
| 2 | Multi-tenant dashboard | Soft hedge + invented path |
| 3 | Numeric limit | Abstain |
| 4 | Parallelism | Soft sequential claim; missed named multi-proc bots |
| 5 | Delete cascade | Hard no while docs underspecify |
| 6 | Git | Mostly honest hedge |
| 7 | Concepts | Toolset good; Extension vague |
| 8 | Staging env | Abstain |
| 9 | Deprecation | Generic path; missed Deprecated banners |
| 10 | Naming | Abstain (could clarify catalog reuse) |
