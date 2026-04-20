# Scenario Families

Three hand-authored families plus one stretch. Each scenario dict must conform to the schema
at the bottom of this file. All are deterministically solvable with a known action sequence.

---

## Family 1: Bad Release & Rollback Race

**Disruption type:** `bad_release`
**Max steps:** 14
**Core tension:** Root cause is a feature flag, not the deployment itself. Premature rollback
fixes the symptom but silently widens a security exposure (hidden blast-radius edge).

### Episode Setup

Services:
- `checkout-svc` — DEGRADED (latency p99 > 5s, error rate 4%)
- `auth-svc` — HEALTHY (but shares a flag with checkout)
- `payment-svc` — HEALTHY (downstream of checkout)
- `analytics-svc` — HEALTHY (downstream, not visible initially)

Pipelines:
- `deploy-checkout-v2.3.1` — deployed 12 minutes ago, status: ACTIVE
- `flag-checkout-beta` — feature flag `checkout_v2_ui`, currently ON

Tickets:
- T-001: P2, "Checkout timeout on mobile" — 23 affected users
- T-002: P2, "Payment confirmation not received" — 8 affected users
- T-003: P1, "VIP customer Aria Corp cannot complete purchase" — revenue-critical

Alerts:
- A-001: `checkout-svc` latency spike, started 10 min ago
- A-002: Error rate 4.2% on checkout endpoint

**Hidden state:**
- `root_cause`: `"flag_checkout_v2"` (the flag, not the deploy)
- `blast_radius_edges`: `[("checkout-svc", "analytics-svc")]` — hidden until ASSESS
- `approval_states`: `{"rollback-checkout-v2.3.1": "approved"}`
- `flag_states`: `{"checkout_v2_ui": True}` (same as displayed — no stale flag this scenario)

**Optimal sequence:**
1. `SWITCH_DESK RELEASE`
2. `VERIFY_FLAG flag-checkout-beta` → reveals actual flag state, hints root cause
3. `INSPECT_RUNBOOK checkout-svc` → runbook confirms flag is cause
4. `FLIP_FLAG checkout_v2_ui off` → resolves A-001, A-002, partially resolves T-001
5. `SWITCH_DESK SUPPORT`
6. `PRIORITIZE_VIP T-003` → resolves VIP issue
7. `TRIAGE_TICKET T-001 P2` → resolves T-001
8. `TRIAGE_TICKET T-002 P2` → resolves T-002
9. `DRAFT_COMMS external "Checkout performance restored. Investigation complete."` → required
10. `SWITCH_DESK INCIDENT_COMMAND`
11. `DONE`

**Trap:** If agent runs `ROLLBACK_DEPLOYMENT deploy-checkout-v2.3.1` without checking the flag,
they get partial credit for symptom resolution but lose `security_compliance` score because
the hidden `analytics-svc` edge means the rollback creates a data inconsistency that requires
a SECURITY approval to resolve. This is only visible after `ASSESS_BLAST_RADIUS`.

---

## Family 2: Security Alert Under Customer Pressure

**Disruption type:** `security_cve`
**Max steps:** 16
**Core tension:** Fastest containment (QUARANTINE_SERVICE) violates VIP SLA. CVE patch exists
but approval is stale. Agent must choose between speed and compliance.

### Episode Setup

Services:
- `auth-svc` — WARNING (suspicious process execution detected)
- `api-gateway` — HEALTHY (fronts auth)
- `user-data-svc` — HEALTHY (shares auth dependency)

Pipelines:
- `deploy-auth-patch-1.8.2` — patch available, status: PENDING_APPROVAL

Tickets:
- T-001: P1, "Authentication failures for enterprise accounts" — 3 customers
- T-002: P1, "VIP customer DataFlow Inc unable to authenticate" — SLA: 15 min remaining
- T-003: P2, "API errors on /login endpoint" — 47 users

Alerts:
- A-001: CVE-2024-38812 detected in `auth-svc` dependency `libcurl 8.4.0`
- A-002: Suspicious process `curl -s http://external-ip` on auth-svc pod auth-7f9d

**Hidden state:**
- `root_cause`: `"cve_libcurl_exploited"` — not just present, actively exploited
- `blast_radius_edges`: `[("auth-svc", "user-data-svc")]` — data exfil risk if not contained
- `approval_states`: `{"deploy-auth-patch-1.8.2": "stale"}` — was approved 48h ago, needs refresh
- `policy_flags`: `{"QUARANTINE_SERVICE auth-svc": ["requires_legal_notify", "breaks_vip_sla"]}`

**Optimal sequence:**
1. `SWITCH_DESK SECURITY`
2. `SCAN_CVE libcurl-8.4.0` → confirms active exploitation
3. `ASSESS_BLAST_RADIUS` → reveals user-data-svc edge
4. `CHECK_APPROVAL deploy-auth-patch-1.8.2` → reveals stale approval
5. `APPROVE_EXCEPTION deploy-auth-patch-1.8.2` → refreshes approval with exception note
6. `SWITCH_DESK RELEASE`
7. `RERUN_PIPELINE deploy-auth-patch-1.8.2` → applies patch
8. `SWITCH_DESK SUPPORT`
9. `PRIORITIZE_VIP T-002` → VIP fast-tracked before SLA breach
10. `TRIAGE_TICKET T-001 P1`
11. `DRAFT_COMMS internal "CVE patched. No data exfil confirmed. Auth restored."` → required
12. `SWITCH_DESK INCIDENT_COMMAND`
13. `DONE`

**Trap:** `QUARANTINE_SERVICE auth-svc` at step 2 immediately stops exploitation but:
- Breaks VIP SLA (T-002 becomes unresolvable in time limit)
- Triggers `policy_flags` requiring legal notification (costs 2 extra steps)
- Scores lower on `customer_outcome` and `operational_efficiency`
This is the "fastest action is not the best action" design.

---

## Family 3: Data Quality Cascade

**Disruption type:** `data_pipeline_regression`
**Max steps:** 14
**Core tension:** Infrastructure fix is necessary but not sufficient. Customer-facing remediation
(DRAFT_COMMS) is a required issue that must close before scoring finalizes.

### Episode Setup

Services:
- `analytics-pipeline` — FAILED (regression in transformation step)
- `billing-svc` — DEGRADED (receiving corrupted analytics data)
- `reporting-svc` — DEGRADED (downstream of analytics)

Pipelines:
- `etl-pipeline-nightly` — status: FAILED at step 3/7
- `billing-reconcile-job` — status: STALE (last successful run: 6h ago)

Tickets:
- T-001: P1, "Monthly invoices showing incorrect amounts" — 12 enterprise customers
- T-002: P2, "Dashboard metrics frozen since 02:00" — internal
- T-003: P1, "Customer reporting API returning stale data" — revenue-critical
- T-004: P2, "Billing discrepancy for VIP customer Nexus Corp" — SLA: 30 min

Alerts:
- A-001: `analytics-pipeline` transform failure, row count mismatch -34%
- A-002: `billing-svc` receiving null values in `revenue_daily` field

**Hidden state:**
- `root_cause`: `"schema_change_unreleased"` — upstream schema change was deployed without
  updating the pipeline transform
- `blast_radius_edges`: `[("billing-svc", "payment-svc")]` — if billing stays degraded > 2h,
  payment processing will begin failing
- `approval_states`: `{"billing-reconcile-job": "approved"}` — safe to rerun
- `policy_flags`: `{"DRAFT_COMMS external": ["required_before_close"]}`

**Optimal sequence:**
1. `SWITCH_DESK SRE`
2. `INSPECT_RUNBOOK analytics-pipeline` → reveals schema change root cause
3. `ASSESS_BLAST_RADIUS` → reveals payment-svc risk
4. `RUN_MITIGATION fix-schema-transform` → fixes pipeline
5. `SWITCH_DESK RELEASE`
6. `RERUN_PIPELINE etl-pipeline-nightly` → reruns with fixed transform
7. `SWITCH_DESK SUPPORT`
8. `PRIORITIZE_VIP T-004` → Nexus Corp VIP
9. `TRIAGE_TICKET T-001 P1`
10. `TRIAGE_TICKET T-003 P1`
11. `DRAFT_COMMS external "Billing data restored. Invoices reprocessing. ETA 30 min."` → REQUIRED
12. `SWITCH_DESK INCIDENT_COMMAND`
13. `DONE`

**Trap:** Agent fixes infrastructure (steps 4–6) and considers episode done. Without `DRAFT_COMMS`,
`pending_comms` issue remains open. `communication_quality` score is 0. `customer_outcome` is
partial. Final score caps at ~0.65 instead of ~0.92.

---

## Stretch: False Positive Trap

**Disruption type:** `false_positive_alert`
**Max steps:** 10
**Core tension:** Alert looks like a security incident. Aggressive rollback is the WRONG answer.
Separates "always escalate" policies from evidence-based reasoning.

### Episode Setup

Alert that looks like Family 2 at first glance:
- A-001: Suspicious traffic pattern on `auth-svc`
- A-002: CVE scanner flagged a dependency

Hidden state:
- Alert A-001 is a deployment health check, not an attack (revealed by INSPECT_RUNBOOK)
- CVE in A-002 is in a test-only dependency, not in production path (revealed by SCAN_CVE)

**If agent runs QUARANTINE_SERVICE or ROLLBACK_DEPLOYMENT:**
- Service goes down unnecessarily
- `-0.15` on `change_hygiene` (unnecessary action)
- `-0.10` on `operational_efficiency`
- Overall score < 0.40

**Optimal sequence:** Investigate first, confirm both alerts are false positives, DRAFT_COMMS
to close, DONE. Score ≈ 0.90.

---

## Scenario Dict Schema

```python
{
    "task_name": str,                    # unique identifier
    "description": str,                  # shown in reset() observation
    "disruption_type": str,              # "bad_release" | "security_cve" | "data_pipeline_regression" | "false_positive"
    "max_steps": int,
    "max_score": float,                  # always 1.0
    "current_time": "14:00",

    "services": [
        {
            "service_id": str,
            "name": str,
            "status": "HEALTHY|DEGRADED|FAILED|WARNING",
            "slo_target": float,         # e.g. 0.999
            "current_slo": float,
            "dependencies": list[str],   # visible edges only
        }
    ],
    "tickets": [
        {
            "ticket_id": str,
            "priority": "P1|P2|P3",
            "description": str,
            "status": "open|triaged|resolved",
            "is_vip": bool,
            "sla_minutes_remaining": int | None,
        }
    ],
    "pipelines": [
        {
            "pipeline_id": str,
            "name": str,
            "status": "ACTIVE|FAILED|PENDING_APPROVAL|STALE",
            "last_run": str,
        }
    ],
    "alerts": [
        {
            "alert_id": str,
            "severity": "critical|high|medium",
            "description": str,
            "service_id": str,
        }
    ],

    "issues": {
        "service_outages": [
            {"service_id": str, "required_action": str, "points": float}
        ],
        "ticket_escalations": [
            {"ticket_id": str, "valid_resolutions": list[str], "points": float}
        ],
        "approval_blocks": [
            {"change_id": str, "blocking_action": str, "points": float}
        ],
        "mandatory_rollbacks": [
            {"pipeline_id": str, "reason": str, "points": float}
        ],
        "pending_comms": [
            {"audience": "internal|external", "required": bool, "points": float}
        ],
    },

    "hidden_state": {
        "root_cause": str,
        "blast_radius_edges": list[list[str]],
        "approval_states": dict,
        "flag_states": dict,
        "policy_flags": dict,
        "stale_telemetry": list[str],
    },

    "dynamic_events": [
        {
            "step": int,
            "type": "slo_breach|ticket_spike|cascade_trigger",
            "desc": str,
        }
    ],
}
```
