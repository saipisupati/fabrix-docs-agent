# Cycle 17 candidates → cases landed in `tests/eval_break.py`

Next unused cycle id after this lands: **18**.

## Cases (11)

| id | class | intent |
|----|-------|--------|
| `c17_kafka_infer_disclosure` | overclaim | Same Kafka→dataset ask as prod PARTIAL; require infer/gaps |
| `c17_pipeline_failed_uat` | day2_ops | Non-prod pipeline failed debug |
| `c17_alerts_not_creating_tickets` | thin_wiring | Alerts not creating tickets path |
| `c17_aiops_pipeline_not_running` | day2_ops | Workers / schedules / status checks |
| `c17_email_pipeline_missed` | day2_ops | Missed email processing |
| `c17_events_page_no_data` | wrong_facet | Events page empty → streams/datasets |
| `c17_trap_contractual_sla` | trap_abstain | MSA P1 SLA (not public docs) |
| `c17_trap_mfa_enable` | trap_abstain | Enable MFA for tenant |
| `c17_trap_cloud_ip_allowlist` | trap_abstain | Cloud Fabrix IP ranges |
| `c17_trap_qdrant_replication` | trap_abstain | Qdrant replication (infra) |
| `c17_incident_oom_playbook` | invented_remediation | Live OOM playbook → refuse invented path |
| `c17b_incident_disk_full` | invented_remediation | Disk-full / prod-down exact fix → refuse |
| `c17c_incident_cpu_spike` | invented_remediation | CPU spike remediation steps → refuse |

## Agent fix (generic)

`_is_live_incident_ask` + pre-generate honesty when no documented playbook/runbook
is in retrieved excerpts (`_live_incident_has_documented_procedure`). Does **not**
false-trigger day-2 scale/rollback/schedule-debug or ServiceNow incident how-tos.

## Gate

```bash
# stop API / other Qdrant holders first
BREAK_CYCLE=17 python3 tests/eval_break.py
```

Promote 2–3 only after cycle 17 is clean (≥95% PASS, 0 FAIL).
