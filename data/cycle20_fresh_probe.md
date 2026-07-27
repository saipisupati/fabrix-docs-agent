# Cycle 20 discovery — fresh 10Q agent vs docs-grounded comparison

**Baseline:** `5f3959b` on `origin/main` (cycle 19 landed). Working tree clean.
**Agent answers:** `data/cycle20_fresh_probe.json`
**Method:** New customer-shaped asks (not in c17–c19 locks). Agent local RAG vs grounded reading of `docs.fabrix.ai` (web + known scrape topics).

## Detector note

`_is_stepwise_procedure_ask` was **False for all 10**. Cycle 19 only fires when a how-cue **and** an action cue (rollback/audit/revert/…) co-occur. Clone / pause / configure / export / migrate how-tos therefore skip both the prompt guardrail and the pre-gen honesty gate — same failure shape as pre–cycle-19, in new clothes.

## Scorecard

| id | Agent | Grounded docs | Verdict |
|----|-------|---------------|---------|
| q1_clone_pipeline | **Documented Fabrix path** + invented `@files:loadfile` “clone” workflow | Pipe builder docs: **Clone** clones a *bot row* in the builder; pipelines are adapted via **edit / new version / Publish**, not a copy-bot | **FAIL — invented procedure** |
| q2_pstream_retention | UI path + `retention_days` (+ optional purge filter); invents an `rdac` CLI example under Next | `retention_days` is documented on Persistent Streams; UI path plausible; CLI command not documented | **PASS** (mild inferred CLI overreach) |
| q3_pause_schedules_maint | **Documented Fabrix path**: Evict Job + tweak/`disable` cron for maintenance | Evict stuck jobs + cron schedules are documented separately; **no** “pause all schedules for maintenance” procedure | **FAIL — invented procedure** (stitched adjacent facts) |
| q4_custom_dynamic_bot | Dynamic bot / Mako / `rda_requests` / `@dm:dynamic-bot` | Matches Dynamic Bots guide | **PASS** |
| q5_cfxvault_update_params | `columns*` only, 0-ish LLM fast path | Matches live `@cfxvault:update-credential` table | **PASS** |
| q6_saml_sso_login | Admin → Authentication Servers → SSO Details → MetaData URL | Matches `rdaf_portal` §10 SSO / SAML | **PASS** |
| q7_webhook_no_pipeline | Confident **No** + invented webhook→pipeline narrative | Docs show webhooks into **OIA alert ingest**, not “invoke arbitrary bot without a pipeline”; absence ≠ hard no | **WEAK — overclaim** |
| q8_migrate_vault_creds | Abstain | No migrate-UAT→prod vault procedure found | **PASS** |
| q9_export_dashboard_xlsx | `@files:datasets-to-xlsx` pipeline path | Bot is real (dataset→xlsx sheets); **dashboard/report** export is a different facet (release notes: pivot/tabular XLSX) | **PARTIAL — wrong facet** |
| q10_multitenant_isolation | Abstain | No public multi-tenant org-isolation playbook found | **PASS** |

**Tally:** 5 PASS · 1 PARTIAL · 1 WEAK · **2 FAIL** (both invented-procedure)

## Dominant new signal (cycle 20 candidate class)

**Invented-procedure still escapes when the how-to verb is outside the cycle-19 action-cue list.**

- q1 / q3 are the same class as bot-rollback / audit-trail: confident numbered path, narrow gap, real adjacent bots/UI mashed into an undocumented workflow.
- Cycle 19’s action-cue gate was correct for false-trigger risk, but coverage is too narrow for the real customer how-to distribution (clone, pause, migrate, export, configure, promote, …).

## Suggested cycle-20 direction (do not implement yet)

Generic broadening of procedural grounding — e.g. treat stepwise how-cues (`how do i` / `what's the process` / `walk me through` / `steps to`) as the primary trigger, and rely on excerpt-procedure matching (not a fixed verb allowlist) to decide honesty vs full path. Keep install/wiring false-trigger checks in unit tests.

Candidate eval locks:

1. `c20_clone_pipeline_invented` — forbid `@files:loadfile` as clone mechanism; need honesty or UI versioning/Publish language
2. `c20_pause_schedules_maint` — forbid confident maintenance playbook; need “no documented pause-all procedure” (or clearly labeled adjacent: cron / evict)
3. Optional: `c20_webhook_invoke_bot` — forbid hard “No, you cannot”; require hedge when docs don’t state capability
