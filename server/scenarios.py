"""
OpsTwin Recovery Arena -- Hand-Authored Scenarios
===================================================
Three core families + one stretch (False Positive Trap).

Every scenario dict conforms to the schema at the bottom of docs/scenarios.md.
All issue points per scenario sum to 1.0 (normalized), so an agent that
resolves every issue via valid actions earns a cumulative episode reward
close to 1.0 before any inspection bonuses or efficiency bonuses.
"""
from typing import Any, Dict, List


# --- Helpers to keep scenario dicts readable -----------------------

def _svc(sid: str, status: str, slo: float = 0.999, curr_slo: float = 0.999,
         deps: List[str] = None) -> Dict:
    return dict(
        service_id=sid, name=sid, status=status,
        slo_target=slo, current_slo=curr_slo,
        dependencies=deps or [],
    )


def _ticket(tid: str, priority: str, description: str,
            is_vip: bool = False, sla_min: int = None) -> Dict:
    return dict(
        ticket_id=tid, priority=priority, description=description,
        status="open", is_vip=is_vip,
        sla_minutes_remaining=sla_min,
    )


def _pipe(pid: str, name: str, status: str, last_run: str = "unknown") -> Dict:
    return dict(
        pipeline_id=pid, name=name, status=status, last_run=last_run,
    )


def _alert(aid: str, severity: str, description: str, service_id: str) -> Dict:
    return dict(
        alert_id=aid, severity=severity,
        description=description, service_id=service_id,
    )


# =================================================================
# FAMILY 1: Bad Release & Rollback Race
# =================================================================

SCENARIO_BAD_RELEASE = dict(
    task_name="bad_release",
    disruption_type="bad_release",
    description=(
        "Checkout service degraded 10 min after deploy-checkout-v2.3.1. "
        "Error rate 4.2%, latency p99 > 5s. 31 customers affected. "
        "VIP account Aria Corp cannot complete purchase. "
        "Root cause may be the feature flag, not the deploy itself."
    ),
    max_steps=14,
    current_time="14:00",
    incident_severity=3,

    services=[
        _svc("checkout-svc", "DEGRADED", 0.999, 0.940,
             deps=["auth-svc", "payment-svc"]),
        _svc("auth-svc", "HEALTHY", 0.999, 0.999),
        _svc("payment-svc", "HEALTHY", 0.999, 0.998,
             deps=["checkout-svc"]),
        _svc("analytics-svc", "HEALTHY", 0.995, 0.995),
    ],
    pipelines=[
        _pipe("deploy-checkout-v2.3.1", "Deploy checkout v2.3.1",
              "ACTIVE", "14:48 12min ago"),
        _pipe("flag-checkout-beta", "Flag: checkout_v2_ui",
              "ACTIVE", "flag on since 13:30"),
    ],
    tickets=[
        _ticket("T-001", "P2", "Checkout timeout on mobile (23 users)"),
        _ticket("T-002", "P2", "Payment confirmation missing (8 users)"),
        _ticket("T-003", "P1", "VIP Aria Corp cannot complete purchase",
                is_vip=True, sla_min=30),
    ],
    alerts=[
        _alert("A-001", "high",
               "checkout-svc latency spike, started 10 min ago",
               "checkout-svc"),
        _alert("A-002", "high",
               "Error rate 4.2% on /checkout endpoint",
               "checkout-svc"),
    ],

    issues={
        "service_outages": [
            {"service_id": "checkout-svc",
             "required_action": "FLIP_FLAG checkout_v2_ui off",
             "valid_actions": ["FLIP_FLAG checkout_v2_ui off"],
             "points": 0.30},
        ],
        "ticket_escalations": [
            {"ticket_id": "T-001",
             "valid_resolutions": ["TRIAGE_TICKET T-001 P2"],
             "points": 0.10},
            {"ticket_id": "T-002",
             "valid_resolutions": ["TRIAGE_TICKET T-002 P2"],
             "points": 0.10},
            {"ticket_id": "T-003",
             "valid_resolutions": ["PRIORITIZE_VIP T-003"],
             "points": 0.20},
        ],
        "approval_blocks": [],
        "mandatory_rollbacks": [],
        "pending_comms": [
            {"audience": "external", "required": True, "points": 0.20,
             "reason": "Customers affected -- external comms required"},
        ],
        "alerts_to_clear": [
            {"alert_id": "A-001", "points": 0.05},
            {"alert_id": "A-002", "points": 0.05},
        ],
    },

    hidden_state={
        "root_cause": "flag_checkout_v2_ui",
        "blast_radius_edges": [["checkout-svc", "analytics-svc"]],
        "approval_states": {"deploy-checkout-v2.3.1": "approved"},
        "flag_states": {"checkout_v2_ui": True},
        "policy_flags": {
            "ROLLBACK_DEPLOYMENT deploy-checkout-v2.3.1": ["blast_radius_assessed"],
        },
        "stale_telemetry": [],
    },

    dynamic_events=[
        {"step": 6, "type": "ticket_spike",
         "desc": "5 more checkout tickets arrived. Error rate still climbing."},
    ],

    max_score=1.0,
)


# =================================================================
# FAMILY 2: Security Alert Under Customer Pressure
# =================================================================

SCENARIO_SECURITY_CVE = dict(
    task_name="security_cve",
    disruption_type="security_cve",
    description=(
        "CVE-2024-38812 detected in auth-svc dependency libcurl 8.4.0. "
        "Suspicious process spotted on pod auth-7f9d. Enterprise accounts "
        "cannot authenticate. VIP DataFlow Inc has 15 min before SLA breach. "
        "Patch is available but approval is stale. Quarantining is fast but "
        "trips policy flags and breaks the VIP SLA."
    ),
    max_steps=16,
    current_time="11:00",
    incident_severity=4,

    services=[
        _svc("auth-svc", "WARNING", 0.999, 0.998),
        _svc("api-gateway", "HEALTHY", 0.999, 0.999,
             deps=["auth-svc"]),
        _svc("user-data-svc", "HEALTHY", 0.999, 0.999,
             deps=["auth-svc"]),
    ],
    pipelines=[
        _pipe("deploy-auth-patch-1.8.2", "Auth patch 1.8.2",
              "PENDING_APPROVAL", "approved 48h ago"),
    ],
    tickets=[
        _ticket("T-001", "P1", "Auth failures for enterprise accounts (3 customers)"),
        _ticket("T-002", "P1", "VIP DataFlow Inc cannot authenticate",
                is_vip=True, sla_min=15),
        _ticket("T-003", "P2", "API errors on /login endpoint (47 users)"),
    ],
    alerts=[
        _alert("A-001", "critical",
               "CVE-2024-38812 in auth-svc dependency libcurl 8.4.0",
               "auth-svc"),
        _alert("A-002", "high",
               "Suspicious process 'curl -s http://external-ip' on pod auth-7f9d",
               "auth-svc"),
    ],

    issues={
        "service_outages": [
            {"service_id": "auth-svc",
             "required_action": "RERUN_PIPELINE deploy-auth-patch-1.8.2",
             "valid_actions": ["RERUN_PIPELINE deploy-auth-patch-1.8.2"],
             "points": 0.25},
        ],
        "ticket_escalations": [
            {"ticket_id": "T-001",
             "valid_resolutions": ["TRIAGE_TICKET T-001 P1"],
             "points": 0.10},
            {"ticket_id": "T-002",
             "valid_resolutions": ["PRIORITIZE_VIP T-002"],
             "points": 0.20},
            {"ticket_id": "T-003",
             "valid_resolutions": ["TRIAGE_TICKET T-003 P2"],
             "points": 0.10},
        ],
        "approval_blocks": [
            {"change_id": "deploy-auth-patch-1.8.2",
             "blocking_action": "APPROVE_EXCEPTION deploy-auth-patch-1.8.2",
             "points": 0.15},
        ],
        "mandatory_rollbacks": [],
        "pending_comms": [
            {"audience": "internal", "required": True, "points": 0.15,
             "reason": "Security incident -- internal comms required"},
        ],
        "alerts_to_clear": [
            {"alert_id": "A-001", "points": 0.025},
            {"alert_id": "A-002", "points": 0.025},
        ],
    },

    hidden_state={
        "root_cause": "cve_libcurl_exploited",
        "blast_radius_edges": [["auth-svc", "user-data-svc"]],
        "approval_states": {"deploy-auth-patch-1.8.2": "stale"},
        "flag_states": {},
        "policy_flags": {
            "QUARANTINE_SERVICE auth-svc": ["requires_legal_notify", "breaks_vip_sla"],
        },
        "stale_telemetry": [],
    },

    dynamic_events=[
        {"step": 8, "type": "slo_breach",
         "desc": "Enterprise auth failures ticking up. +2 customers affected."},
    ],

    max_score=1.0,
)


# =================================================================
# FAMILY 3: Data Quality Cascade
# =================================================================

SCENARIO_DATA_PIPELINE = dict(
    task_name="data_pipeline_regression",
    disruption_type="data_pipeline_regression",
    description=(
        "ETL pipeline failed at step 3 of 7. analytics-pipeline returning -34% "
        "row count. billing-svc receiving null values. Monthly invoices wrong "
        "for 12 enterprise customers. VIP Nexus Corp has 30 min on SLA. "
        "Root cause: upstream schema change deployed without transform update."
    ),
    max_steps=14,
    current_time="08:30",
    incident_severity=3,

    services=[
        _svc("analytics-pipeline", "FAILED", 0.995, 0.820),
        _svc("billing-svc", "DEGRADED", 0.999, 0.975,
             deps=["analytics-pipeline"]),
        _svc("reporting-svc", "DEGRADED", 0.995, 0.960,
             deps=["analytics-pipeline"]),
        _svc("payment-svc", "HEALTHY", 0.999, 0.999),
    ],
    pipelines=[
        _pipe("etl-pipeline-nightly", "ETL nightly",
              "FAILED", "failed at step 3 of 7"),
        _pipe("billing-reconcile-job", "Billing reconcile",
              "STALE", "last success 6h ago"),
    ],
    tickets=[
        _ticket("T-001", "P1", "Monthly invoices showing wrong amounts (12 enterprise customers)"),
        _ticket("T-002", "P2", "Dashboard metrics frozen since 02:00 (internal)"),
        _ticket("T-003", "P1", "Customer reporting API returning stale data"),
        _ticket("T-004", "P2", "Billing discrepancy for VIP Nexus Corp",
                is_vip=True, sla_min=30),
    ],
    alerts=[
        _alert("A-001", "critical",
               "analytics-pipeline transform failure, row count mismatch -34%",
               "analytics-pipeline"),
        _alert("A-002", "high",
               "billing-svc receiving null in revenue_daily field",
               "billing-svc"),
    ],

    issues={
        "service_outages": [
            {"service_id": "analytics-pipeline",
             "required_action": "RUN_MITIGATION fix-schema-transform",
             "valid_actions": ["RUN_MITIGATION fix-schema-transform"],
             "points": 0.20},
            {"service_id": "billing-svc",
             "required_action": "RERUN_PIPELINE etl-pipeline-nightly",
             "valid_actions": ["RERUN_PIPELINE etl-pipeline-nightly"],
             "points": 0.15},
        ],
        "ticket_escalations": [
            {"ticket_id": "T-001",
             "valid_resolutions": ["TRIAGE_TICKET T-001 P1"],
             "points": 0.10},
            {"ticket_id": "T-003",
             "valid_resolutions": ["TRIAGE_TICKET T-003 P1"],
             "points": 0.10},
            {"ticket_id": "T-004",
             "valid_resolutions": ["PRIORITIZE_VIP T-004"],
             "points": 0.15},
        ],
        "approval_blocks": [],
        "mandatory_rollbacks": [],
        "pending_comms": [
            {"audience": "external", "required": True, "points": 0.25,
             "reason": "Customer-facing data corruption -- external comms required"},
        ],
        "alerts_to_clear": [
            {"alert_id": "A-001", "points": 0.025},
            {"alert_id": "A-002", "points": 0.025},
        ],
    },

    hidden_state={
        "root_cause": "schema_change_unreleased_analytics",
        "blast_radius_edges": [["billing-svc", "payment-svc"]],
        "approval_states": {"billing-reconcile-job": "approved"},
        "flag_states": {},
        "policy_flags": {},
        "stale_telemetry": ["billing_reconcile_last_success"],
    },

    dynamic_events=[
        {"step": 7, "type": "slo_breach",
         "desc": "3 more customer reports landed. Reporting-svc SLO dipped."},
    ],

    max_score=1.0,
)


# =================================================================
# STRETCH: False Positive Trap
# =================================================================

SCENARIO_FALSE_POSITIVE = dict(
    task_name="false_positive",
    disruption_type="false_positive",
    description=(
        "Suspicious traffic on auth-svc, CVE flagged on a dependency. "
        "Surface reading says 'security incident'. But the traffic might be "
        "a health check and the CVE might be in a test-only path. "
        "Investigate before you act."
    ),
    max_steps=10,
    current_time="10:00",
    incident_severity=2,

    services=[
        _svc("auth-svc", "WARNING", 0.999, 0.998),
        _svc("api-gateway", "HEALTHY", 0.999, 0.999, deps=["auth-svc"]),
    ],
    pipelines=[
        _pipe("deploy-auth-v1.9.0", "Auth v1.9.0", "ACTIVE", "deployed 3h ago"),
    ],
    tickets=[],  # no customer impact -- part of the trap
    alerts=[
        _alert("A-001", "high",
               "Suspicious traffic pattern on auth-svc (looks like scan)",
               "auth-svc"),
        _alert("A-002", "medium",
               "CVE scanner flagged dependency libfoo-0.3",
               "auth-svc"),
    ],

    issues={
        "service_outages": [],
        "ticket_escalations": [],
        "approval_blocks": [],
        "mandatory_rollbacks": [],
        "pending_comms": [
            {"audience": "internal", "required": True, "points": 0.25,
             "reason": "False positives closed, notify team."},
        ],
        "alerts_to_clear": [
            {"alert_id": "A-001", "points": 0.375,
             "requires_inspection": ["runbook:auth-svc"]},
            {"alert_id": "A-002", "points": 0.375,
             "requires_inspection": ["runbook:auth-svc"]},
        ],
    },

    hidden_state={
        "root_cause": "false_positive_auth_healthcheck",
        "blast_radius_edges": [],
        "approval_states": {},
        "flag_states": {},
        "policy_flags": {
            "QUARANTINE_SERVICE auth-svc": ["requires_legal_notify"],
            "ROLLBACK_DEPLOYMENT deploy-auth-v1.9.0": ["blast_radius_assessed"],
        },
        "stale_telemetry": [],
    },

    dynamic_events=[],

    max_score=1.0,
)


# --- Registry ----------------------------------------------------

SCENARIOS = {
    "bad_release": SCENARIO_BAD_RELEASE,
    "security_cve": SCENARIO_SECURITY_CVE,
    "data_pipeline_regression": SCENARIO_DATA_PIPELINE,
    "false_positive": SCENARIO_FALSE_POSITIVE,
}
ALL_TASK_NAMES = list(SCENARIOS.keys())
