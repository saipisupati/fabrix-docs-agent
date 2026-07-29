"""Light locks for integration wiring detection + shape gate."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import (  # noqa: E402
    _is_integration_wiring_ask,
    _wiring_answer_missing_product_bot,
    _wiring_answer_missing_shape,
)


def test_wiring_detect_map_dashboard():
    assert _is_integration_wiring_ask(
        "How do I map Splunk events into a Fabrix dashboard that ops can watch?"
    )


def test_wiring_detect_building_blocks_dataset():
    assert _is_integration_wiring_ask(
        "Explain the building blocks I'd use to turn Kafka messages into a searchable dataset."
    )


def test_wiring_detect_land_dashboard():
    assert _is_integration_wiring_ask(
        "How would I connect New Relic APM as a datasource and land it in a dashboard?"
    )


def test_wiring_detect_not_plain_param_lookup():
    assert not _is_integration_wiring_ask(
        "What parameters does @snowv2:list-incidents take?"
    )


def test_missing_bot_token_forces_shape():
    q = "Walk me through wiring Datadog metrics into a persistent stream, then alerting Slack."
    thin = (
        "Create Datadog credentials, use the Datadog bot to pull metrics, "
        "then land them in a persistent stream and alert Slack."
    )
    # Empty context → still require tokens (strict unit default)
    assert _wiring_answer_missing_product_bot(q, thin)
    assert _wiring_answer_missing_shape(q, thin) is False  # auth + sink present
    # Context with catalog tokens → still missing in answer
    ctx = "Use @datadog:metrics to query metric data from DataDog."
    assert _wiring_answer_missing_product_bot(q, thin, ctx)
    # Context with no family bots → do not force invention
    assert not _wiring_answer_missing_product_bot(q, thin, "no bots here, only dashboards")


def test_full_shape_passes():
    q = "How do I map Splunk events into a Fabrix dashboard that ops can watch?"
    good = (
        "Use a Splunk API token, then @splunkv2:search-index to query events, "
        "write to a dataset, and build a dashboard on that dataset."
    )
    assert not _wiring_answer_missing_product_bot(q, good)
    assert not _wiring_answer_missing_shape(q, good)


def test_hash_prefixed_source_bots_detected():
    q = "How do I map Splunk events into a Fabrix dashboard that ops can watch?"
    ctx = """
## Bot @splunkv2:add-to-index
Bot Position In Pipeline: Sink
Add log event records to an index in Splunk

## Bot #splunkv2:search-index
Bot Position In Pipeline: Source
Query and filter log event records within an index in Splunk
"""
    from agent import _wiring_source_bot_tokens_in_context, _wiring_answer_missing_source_bot

    src = _wiring_source_bot_tokens_in_context(q, ctx)
    assert "splunkv2:search-index" in src
    sink_only = (
        "Auth with Splunk password, use @splunkv2:add-to-index, then dashboard."
    )
    assert _wiring_answer_missing_source_bot(q, sink_only, ctx)
    good = (
        "Auth with Splunk password, use @splunkv2:search-index to pull events "
        "into a dataset, then dashboard."
    )
    assert not _wiring_answer_missing_source_bot(q, good, ctx)


def test_grafana_unmatched_no_family_depth():
    from agent import (
        _has_real_family_match,
        _is_integration_wiring_ask,
        _unmatched_wiring_product_names,
    )

    q = "How do I wire Grafana Cloud into Fabrix end to end for dashboards?"
    assert not _has_real_family_match(q)
    assert not _is_integration_wiring_ask(q)
    names = _unmatched_wiring_product_names(q)
    assert any("grafana" in n.lower() for n in names)


def test_pagerduty_still_wiring_family():
    from agent import _has_real_family_match, _is_integration_wiring_ask

    q = "How do I wire PagerDuty into Fabrix for dashboards?"
    assert _has_real_family_match(q)
    assert _is_integration_wiring_ask(q)


def test_agentic_end_to_end_not_unmatched_wiring():
    from agent import _unmatched_wiring_product_names

    q = (
        "Configure Fabio Copilot to auto-remediate production outages "
        "end-to-end with no human approval."
    )
    assert _unmatched_wiring_product_names(q) == []


def test_live_incident_oom_playbook_detected():
    from agent import _is_live_incident_ask

    assert _is_live_incident_ask(
        "Walk me through the OOM remediation playbook for a live incident right now."
    )
    assert _is_live_incident_ask(
        "Out of Memory issue on LNX-THRESHOLD host — give me the exact Fabrix "
        "remediation playbook step by step for this live incident."
    )
    assert _is_live_incident_ask(
        "Production is down, disk is full on the RDA worker -- what's the exact fix right now?"
    )
    assert _is_live_incident_ask(
        "We have a CPU spike incident on a pipeline worker -- walk me through the remediation steps."
    )


def test_live_incident_does_not_false_trigger_day2():
    from agent import _is_live_incident_ask

    assert not _is_live_incident_ask(
        "How do I scale RDA workers for a busy site, and what limits should I watch?"
    )
    assert not _is_live_incident_ask(
        "How do I roll back a failed service blueprint deployment on one RDA site "
        "without affecting other sites?"
    )
    assert not _is_live_incident_ask(
        "A production scheduled pipeline did not trigger overnight — what "
        "documented checks should I run first?"
    )
    assert not _is_live_incident_ask(
        "How should a Fabrix pipeline update ServiceNow incident state and "
        "enrichment fields without inventing custom bots?"
    )


def test_live_incident_procedure_requires_runbook_language():
    from agent import _live_incident_has_documented_procedure

    q = "Walk me through the OOM remediation playbook for a live incident right now."
    loose = [
        {
            "title": "Worker memory limits",
            "text": "Adjust mem_limit in values.yaml for OIA services under load.",
        }
    ]
    assert not _live_incident_has_documented_procedure(q, loose, [])
    # Near miss: an OIA page that mentions memory *and* the word playbook
    # (e.g. ansible run-playbook) is still not an OOM remediation procedure.
    oia_adjacent = [
        {
            "title": "OIA deployment",
            "text": "Run the ansible run-playbook step; OIA requires 4 GB memory per node.",
        }
    ]
    assert not _live_incident_has_documented_procedure(q, oia_adjacent, [])
    runbook = [
        {
            "title": "OOM incident response playbook",
            "text": "Official remediation playbook for out of memory host incidents.",
        }
    ]
    assert _live_incident_has_documented_procedure(q, runbook, [])


def test_oia_memory_config_is_not_live_incident_ask():
    from agent import _is_live_incident_ask

    assert not _is_live_incident_ask("What memory does Fabrix OIA use for deployment?")
    assert not _is_live_incident_ask("What memory settings does OIA use?")
    assert _is_live_incident_ask(
        "Walk me through the OOM remediation playbook for a live incident right now."
    )


def test_saas_hosting_infra_ask_detected():
    from agent import _is_saas_hosting_infra_ask

    assert _is_saas_hosting_infra_ask(
        "What is the public IP range / allowlist for Cloud Fabrix SaaS?"
    )
    assert _is_saas_hosting_infra_ask(
        "What firewall rules or CIDR ranges do I need to open for Fabrix SaaS?"
    )


def test_saas_hosting_infra_does_not_false_trigger_product_howtos():
    from agent import _is_saas_hosting_infra_ask

    assert not _is_saas_hosting_infra_ask(
        "How do I scale RDA workers for a busy site, and what limits should I watch?"
    )
    assert not _is_saas_hosting_infra_ask(
        "How do I wire Splunk into Fabrix and land events in a dashboard?"
    )
    assert not _is_saas_hosting_infra_ask(
        "Can you enable MFA for our Fabrix tenant and confirm it's feasible for our org?"
    )


def test_salvage_skips_when_gaps_declare_core_ask_uncovered():
    from agent import _salvage_partial_answer

    q = "What is the public IP range / allowlist for Cloud Fabrix SaaS?"
    kb = [
        {
            "title": "PRTG Network Monitor",
            "text": (
                "CloudFabrix supports PRTG Network Monitor API integration for fetching "
                "asset inventory with allowlist permissions for sensors."
            ),
        }
    ]
    gaps = [
        "Public IP range / allowlist for Cloud Fabrix SaaS is not specified."
    ]
    assert _salvage_partial_answer(q, kb, [], gaps) is None


def test_salvage_still_runs_for_generic_exhaustive_gap():
    from agent import _salvage_partial_answer

    q = "Explain how persistent streams differ from datasets in RDA Fabric in exactly 8 numbered steps"
    kb = [
        {
            "title": "Persistent streams",
            "text": (
                "A persistent stream (pstream) retains events for downstream analytics "
                "while a dataset is a tabular landing zone for batch queries."
            ),
        }
    ]
    gaps = [
        "Public documentation does not provide a complete exhaustive answer for this ask"
    ]
    salvaged = _salvage_partial_answer(q, kb, [], gaps)
    assert salvaged is not None
    text, out_gaps = salvaged
    assert "Documented Fabrix path" in text
    assert "persistent stream" in text.lower() or "pstream" in text.lower()


def test_gaps_declare_uncovered_accepts_does_not_specify():
    from agent import _gaps_declare_core_ask_uncovered, _salvage_partial_answer

    q = "Is there a REST API to programmatically list all bots in a pipeline?"
    gaps = [
        "The documentation does not specify whether there is a REST API to list all bots in a pipeline."
    ]
    assert _gaps_declare_core_ask_uncovered(q, gaps)
    kb = [
        {
            "title": "ServiceNow bots",
            "text": "bot #snowv2:list-incidents lists incidents from ServiceNow.",
        }
    ]
    assert _salvage_partial_answer(q, kb, [], gaps) is None


def test_gaps_declare_uncovered_accepts_no_information_about():
    from agent import _gaps_declare_core_ask_uncovered, _salvage_partial_answer

    q = "Is there a REST API to programmatically list all bots in a pipeline?"
    gaps = [
        "No information about a REST API to programmatically list all bots in a "
        "pipeline is available in the documentation."
    ]
    assert _gaps_declare_core_ask_uncovered(q, gaps)
    kb = [
        {
            "title": "ServiceNow bots",
            "text": "bot #snowv2:list-incidents lists incidents from ServiceNow.",
        }
    ]
    assert _salvage_partial_answer(q, kb, [], gaps) is None


def test_stepwise_procedure_ask_detected():
    from agent import _is_stepwise_procedure_ask

    assert _is_stepwise_procedure_ask(
        "How do I roll back a bot to a previous version after a bad deploy?"
    )
    assert _is_stepwise_procedure_ask(
        "How do I audit who changed a pipeline configuration and when?"
    )
    assert _is_stepwise_procedure_ask(
        "Walk me through the steps to restore a previous pipeline deploy."
    )
    # Cycle 20: imperative + Fabrix object (no allowlisted verb required)
    assert _is_stepwise_procedure_ask("How do I clone a pipeline in Fabrix?")
    assert _is_stepwise_procedure_ask(
        "How do I pause all schedules for maintenance in Fabrix?"
    )
    assert _is_stepwise_procedure_ask(
        "What's the process to migrate vault credentials between sites?"
    )


def test_stepwise_procedure_ask_does_not_false_trigger():
    from agent import _is_stepwise_procedure_ask

    assert not _is_stepwise_procedure_ask(
        "Is there a REST API to programmatically list all bots in a pipeline?"
    )
    assert not _is_stepwise_procedure_ask(
        "What parameters does @snowv2:list-incidents take?"
    )
    assert not _is_stepwise_procedure_ask(
        "What are the parameters for @cfxvault:update-credential?"
    )
    assert not _is_stepwise_procedure_ask(
        "What is the public IP range / allowlist for Cloud Fabrix SaaS?"
    )
    assert not _is_stepwise_procedure_ask(
        "Can I invoke a bot via webhook without a pipeline?"
    )


def test_undocumented_behavior_ask_detected():
    from agent import _is_undocumented_behavior_ask

    assert _is_undocumented_behavior_ask(
        "If a pipeline has two branches that both write to the same dataset, which one wins?"
    )
    assert _is_undocumented_behavior_ask(
        "If I delete a pstream, does that also delete the datasets that reference it?"
    )
    assert _is_undocumented_behavior_ask(
        "What happens if two bots try to write to the same dataset at once?"
    )


def test_undocumented_behavior_ask_does_not_false_trigger():
    from agent import _is_undocumented_behavior_ask

    assert not _is_undocumented_behavior_ask(
        "What parameters does @cfxvault:update-credential take?"
    )
    assert not _is_undocumented_behavior_ask(
        "What's the maximum size of a single dataset before performance degrades?"
    )
    assert not _is_undocumented_behavior_ask(
        "How do I configure SAML SSO?"
    )


def test_concurrent_dataset_ask_covers_branch_wins_phrasing():
    from agent import _is_concurrent_dataset_ask

    assert _is_concurrent_dataset_ask(
        "If a pipeline has two branches that both write to the same dataset, which one wins?"
    )
    assert _is_concurrent_dataset_ask(
        "What happens if two bots try to write to the same dataset at once?"
    )


def test_cascade_side_effect_ask_detected():
    from agent import _is_cascade_side_effect_ask

    assert _is_cascade_side_effect_ask(
        "If I delete a pstream, does that also delete the datasets that reference it?"
    )
    assert not _is_cascade_side_effect_ask(
        "How do I delete a pstream in the UI?"
    )


def test_excerpts_procedure_grounding_clone_and_pause():
    from agent import _excerpts_describe_asked_procedure

    adjacent = [
        {
            "title": "pipe_builder",
            "text": (
                "Click on Clone action on the top right side of the pane to clone "
                "the bot to add more rows. Publish the draft pipeline when verified."
            ),
        }
    ]
    assert not _excerpts_describe_asked_procedure(
        "How do I clone a pipeline in Fabrix?", adjacent, []
    )
    schedules = [
        {
            "title": "scheduled_pipelines",
            "text": (
                "scheduled_pipelines use cron_expression. Evict stuck pipelines from "
                "Active Jobs when a job is stuck in initialization."
            ),
        }
    ]
    assert not _excerpts_describe_asked_procedure(
        "How do I pause all schedules for maintenance in Fabrix?", schedules, []
    )
    retention = [
        {
            "title": "persistent_streams",
            "text": (
                "Configure retention_days on a persistent stream under RDA "
                "Administration → Persistent Streams. Older data is purged."
            ),
        }
    ]
    assert _excerpts_describe_asked_procedure(
        "How do I set data retention on a persistent stream?",
        retention,
        [],
    )


def test_bot_naming_uniqueness_ask_detected():
    from agent import _is_bot_naming_uniqueness_ask, _normalize_question_typos

    q = "Can two bots share the same name if they're in different pipelines?"
    assert _is_bot_naming_uniqueness_ask(q)
    expanded = _normalize_question_typos(q)
    assert "bot package unique naming SDK" in expanded
    assert not _is_bot_naming_uniqueness_ask(
        "What parameters does @cfxvault:update-credential take?"
    )
    assert not _is_bot_naming_uniqueness_ask(
        "How do I wire Splunk into a Fabrix dashboard?"
    )


def test_git_pipeline_versioning_ask_detected():
    from agent import _is_git_pipeline_versioning_ask, _git_pipeline_versioning_honesty

    assert _is_git_pipeline_versioning_ask(
        "Can I version-control my pipeline definitions in Git?"
    )
    assert _is_git_pipeline_versioning_ask(
        "How do I put Fabrix pipelines under GitHub version control?"
    )
    assert not _is_git_pipeline_versioning_ask(
        "How do I wire Slack alerts into a Fabrix pipeline?"
    )
    assert not _is_git_pipeline_versioning_ask(
        "Does GitHub Actions deploy Fabrix pipelines?"
    )
    text, gaps = _git_pipeline_versioning_honesty([])
    low = text.lower()
    assert "version history" in low and "publish" in low and "studio" in low
    assert "git" in low and "first-class" in low
    assert "manual git export" not in low
    assert "export to git" not in low
    assert any("version history" in g.lower() for g in gaps)

