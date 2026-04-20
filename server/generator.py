"""
Procedural Scenario Generator
===============================
Generates seed-deterministic, solvable scenarios in three families:

  bad_release          - feature flag is root cause, rollback is a trap
  security_cve         - stale-approved patch vs fast-but-harmful quarantine
  data_pipeline        - schema change requires mitigation + rerun + comms

Each generated scenario conforms to the same shape as the hand-authored
SCENARIOS in server/scenarios.py. Issue points sum to 1.0.

Used by: evaluate.py's held-out eval (seeds not in training), train.py's
curriculum (optional), and the research story ("1000+ scenarios from 3
families").
"""
import random
from typing import Dict, List

AIRLINES = ["checkout", "auth", "billing", "reporting", "analytics", "notification"]
DEPS = ["libcurl", "libssl", "libfoo", "libxml", "libzstd"]
CUSTOMERS = ["Aria Corp", "DataFlow Inc", "Nexus Corp", "Orbit Labs", "Meridian Systems"]


def _distribute_points(counts: Dict[str, int], total: float = 1.0) -> Dict[str, List[float]]:
    """Spread `total` across the N issue slots, summing exactly to `total`.

    counts: {"service_outages": 1, "ticket_escalations": 3, ...}
    Returns: same keys mapped to a list of per-item point values.
    """
    # Weight per category (roughly mirrors handcrafted scenarios)
    category_weight = {
        "service_outages": 0.35,
        "ticket_escalations": 0.25,
        "approval_blocks": 0.10,
        "mandatory_rollbacks": 0.05,
        "pending_comms": 0.20,
        "alerts_to_clear": 0.05,
    }
    # Normalize only over categories that have at least one issue
    active = {k: category_weight[k] for k, n in counts.items() if n > 0}
    total_w = sum(active.values()) or 1.0
    result = {}
    for cat, n in counts.items():
        if n == 0:
            result[cat] = []
            continue
        share = active[cat] / total_w * total
        per_item = round(share / n, 4)
        pts = [per_item] * n
        pts[-1] = round(share - sum(pts[:-1]), 4)  # correct rounding
        result[cat] = pts
    return result


# --- Family 1: bad_release ----------------------------------------

def generate_bad_release(seed: int, difficulty: str = "medium") -> Dict:
    rng = random.Random(seed)
    svc = rng.choice(["checkout", "billing", "search", "cart"])
    svc_id = f"{svc}-svc"
    flag_id = f"{svc}_v2_ui"
    deploy_id = f"deploy-{svc}-v{rng.randint(2, 4)}.{rng.randint(0, 9)}.{rng.randint(0, 5)}"
    vip = rng.choice(CUSTOMERS)

    n_tickets = {"easy": 2, "medium": 3, "hard": 4}[difficulty]
    max_steps = {"easy": 12, "medium": 14, "hard": 16}[difficulty]

    tickets = []
    for i in range(n_tickets):
        is_vip = (i == 0)
        tickets.append(dict(
            ticket_id=f"T-{i+1:03d}",
            priority="P1" if is_vip else "P2",
            description=f"{svc} issue ({rng.randint(5, 40)} users)",
            status="open",
            is_vip=is_vip,
            sla_minutes_remaining=30 if is_vip else None,
        ))

    services = [
        dict(service_id=svc_id, name=svc_id, status="DEGRADED",
             slo_target=0.999, current_slo=0.94, dependencies=[]),
        dict(service_id="auth-svc", name="auth-svc", status="HEALTHY",
             slo_target=0.999, current_slo=0.999, dependencies=[]),
        dict(service_id=f"{rng.choice(['payment', 'analytics'])}-svc",
             name="downstream-svc", status="HEALTHY",
             slo_target=0.999, current_slo=0.999, dependencies=[svc_id]),
    ]
    pipelines = [
        dict(pipeline_id=deploy_id, name=f"Deploy {svc}",
             status="ACTIVE", last_run="recent"),
    ]
    alerts = [
        dict(alert_id="A-001", severity="high",
             description=f"{svc_id} latency spike",
             service_id=svc_id),
    ]

    counts = {
        "service_outages": 1,
        "ticket_escalations": n_tickets,
        "approval_blocks": 0,
        "mandatory_rollbacks": 0,
        "pending_comms": 1,
        "alerts_to_clear": 1,
    }
    pts = _distribute_points(counts)

    issues = {
        "service_outages": [{
            "service_id": svc_id,
            "required_action": f"FLIP_FLAG {flag_id} off",
            "valid_actions": [f"FLIP_FLAG {flag_id} off"],
            "points": pts["service_outages"][0],
        }],
        "ticket_escalations": [
            {"ticket_id": t["ticket_id"],
             "valid_resolutions": [f"PRIORITIZE_VIP {t['ticket_id']}"]
                                   if t["is_vip"]
                                   else [f"TRIAGE_TICKET {t['ticket_id']} {t['priority']}"],
             "points": pts["ticket_escalations"][i]}
            for i, t in enumerate(tickets)
        ],
        "approval_blocks": [],
        "mandatory_rollbacks": [],
        "pending_comms": [{
            "audience": "external", "required": True,
            "points": pts["pending_comms"][0],
        }],
        "alerts_to_clear": [{
            "alert_id": "A-001",
            "points": pts["alerts_to_clear"][0],
        }],
    }

    return dict(
        task_name=f"gen_bad_release_s{seed}_{difficulty}",
        disruption_type="bad_release",
        description=(
            f"{svc_id} degraded 10 min after {deploy_id}. "
            f"VIP {vip} affected. Root cause suspected on flag {flag_id}."
        ),
        max_steps=max_steps,
        current_time=f"{rng.randint(9, 18):02d}:00",
        incident_severity=3,
        services=services,
        pipelines=pipelines,
        tickets=tickets,
        alerts=alerts,
        issues=issues,
        hidden_state=dict(
            root_cause=f"flag_{flag_id}",
            blast_radius_edges=[[svc_id, "analytics-svc"]],
            approval_states={deploy_id: "approved"},
            flag_states={flag_id: True},
            policy_flags={f"ROLLBACK_DEPLOYMENT {deploy_id}": ["blast_radius_assessed"]},
            stale_telemetry=[],
        ),
        dynamic_events=[],
        max_score=1.0,
    )


# --- Family 2: security_cve ---------------------------------------

def generate_security_cve(seed: int, difficulty: str = "medium") -> Dict:
    rng = random.Random(seed)
    svc_id = "auth-svc"
    dep = rng.choice(DEPS)
    dep_version = f"{rng.randint(7, 9)}.{rng.randint(0, 5)}.{rng.randint(0, 9)}"
    patch_id = f"deploy-auth-patch-{rng.randint(1, 2)}.{rng.randint(8, 9)}.{rng.randint(0, 5)}"
    vip = rng.choice(CUSTOMERS)

    n_tickets = {"easy": 2, "medium": 3, "hard": 4}[difficulty]
    max_steps = {"easy": 14, "medium": 16, "hard": 18}[difficulty]

    tickets = []
    for i in range(n_tickets):
        is_vip = (i == 0)
        tickets.append(dict(
            ticket_id=f"T-{i+1:03d}",
            priority="P1",
            description=f"Auth failures for {'VIP ' + vip if is_vip else 'enterprise accounts'}",
            status="open",
            is_vip=is_vip,
            sla_minutes_remaining=15 if is_vip else None,
        ))

    services = [
        dict(service_id=svc_id, status="WARNING",
             slo_target=0.999, current_slo=0.998, dependencies=[]),
        dict(service_id="api-gateway", status="HEALTHY",
             slo_target=0.999, current_slo=0.999, dependencies=[svc_id]),
        dict(service_id="user-data-svc", status="HEALTHY",
             slo_target=0.999, current_slo=0.999, dependencies=[svc_id]),
    ]
    pipelines = [dict(pipeline_id=patch_id, name=f"Auth patch",
                      status="PENDING_APPROVAL", last_run="48h ago approved")]
    alerts = [
        dict(alert_id="A-001", severity="critical",
             description=f"CVE in {dep}-{dep_version}", service_id=svc_id),
        dict(alert_id="A-002", severity="high",
             description="Suspicious process on auth pod", service_id=svc_id),
    ]

    counts = {
        "service_outages": 1,
        "ticket_escalations": n_tickets,
        "approval_blocks": 1,
        "mandatory_rollbacks": 0,
        "pending_comms": 1,
        "alerts_to_clear": 2,
    }
    pts = _distribute_points(counts)

    issues = {
        "service_outages": [{
            "service_id": svc_id,
            "required_action": f"RERUN_PIPELINE {patch_id}",
            "valid_actions": [f"RERUN_PIPELINE {patch_id}"],
            "points": pts["service_outages"][0],
        }],
        "ticket_escalations": [
            {"ticket_id": t["ticket_id"],
             "valid_resolutions": [f"PRIORITIZE_VIP {t['ticket_id']}"]
                                   if t["is_vip"]
                                   else [f"TRIAGE_TICKET {t['ticket_id']} {t['priority']}"],
             "points": pts["ticket_escalations"][i]}
            for i, t in enumerate(tickets)
        ],
        "approval_blocks": [{
            "change_id": patch_id,
            "blocking_action": f"APPROVE_EXCEPTION {patch_id}",
            "points": pts["approval_blocks"][0],
        }],
        "mandatory_rollbacks": [],
        "pending_comms": [{
            "audience": "internal", "required": True,
            "points": pts["pending_comms"][0],
        }],
        "alerts_to_clear": [
            {"alert_id": "A-001", "points": pts["alerts_to_clear"][0]},
            {"alert_id": "A-002", "points": pts["alerts_to_clear"][1]},
        ],
    }

    return dict(
        task_name=f"gen_security_cve_s{seed}_{difficulty}",
        disruption_type="security_cve",
        description=(
            f"CVE detected in auth-svc dependency {dep}-{dep_version}. "
            f"Patch available but approval stale. VIP {vip} under SLA pressure."
        ),
        max_steps=max_steps,
        current_time=f"{rng.randint(9, 18):02d}:00",
        incident_severity=4,
        services=services,
        pipelines=pipelines,
        tickets=tickets,
        alerts=alerts,
        issues=issues,
        hidden_state=dict(
            root_cause=f"cve_{dep}_exploited",
            blast_radius_edges=[[svc_id, "user-data-svc"]],
            approval_states={patch_id: "stale"},
            flag_states={},
            policy_flags={
                f"QUARANTINE_SERVICE {svc_id}": ["requires_legal_notify", "breaks_vip_sla"],
            },
            stale_telemetry=[],
        ),
        dynamic_events=[],
        max_score=1.0,
    )


# --- Family 3: data_pipeline --------------------------------------

def generate_data_pipeline(seed: int, difficulty: str = "medium") -> Dict:
    rng = random.Random(seed)
    pipeline_id = f"etl-{rng.choice(['nightly', 'hourly', 'realtime'])}-{rng.randint(1, 9)}"
    vip = rng.choice(CUSTOMERS)

    n_tickets = {"easy": 3, "medium": 4, "hard": 5}[difficulty]
    max_steps = {"easy": 12, "medium": 14, "hard": 16}[difficulty]

    tickets = []
    for i in range(n_tickets):
        is_vip = (i == 0)
        p = "P1" if i < 2 else "P2"
        tickets.append(dict(
            ticket_id=f"T-{i+1:03d}",
            priority=p,
            description=f"Data issue: "
                        f"{'VIP ' + vip + ' billing' if is_vip else 'customer report'}",
            status="open",
            is_vip=is_vip,
            sla_minutes_remaining=30 if is_vip else None,
        ))

    services = [
        dict(service_id="analytics-pipeline", status="FAILED",
             slo_target=0.995, current_slo=0.82, dependencies=[]),
        dict(service_id="billing-svc", status="DEGRADED",
             slo_target=0.999, current_slo=0.975,
             dependencies=["analytics-pipeline"]),
        dict(service_id="reporting-svc", status="DEGRADED",
             slo_target=0.995, current_slo=0.96,
             dependencies=["analytics-pipeline"]),
    ]
    pipelines = [dict(pipeline_id=pipeline_id, name="ETL",
                      status="FAILED", last_run="failed step 3")]
    alerts = [
        dict(alert_id="A-001", severity="critical",
             description="analytics-pipeline row count mismatch",
             service_id="analytics-pipeline"),
    ]

    counts = {
        "service_outages": 2,
        "ticket_escalations": n_tickets,
        "approval_blocks": 0,
        "mandatory_rollbacks": 0,
        "pending_comms": 1,
        "alerts_to_clear": 1,
    }
    pts = _distribute_points(counts)

    issues = {
        "service_outages": [
            {"service_id": "analytics-pipeline",
             "valid_actions": ["RUN_MITIGATION fix-schema-transform"],
             "points": pts["service_outages"][0]},
            {"service_id": "billing-svc",
             "valid_actions": [f"RERUN_PIPELINE {pipeline_id}"],
             "points": pts["service_outages"][1]},
        ],
        "ticket_escalations": [
            {"ticket_id": t["ticket_id"],
             "valid_resolutions": [f"PRIORITIZE_VIP {t['ticket_id']}"]
                                   if t["is_vip"]
                                   else [f"TRIAGE_TICKET {t['ticket_id']} {t['priority']}"],
             "points": pts["ticket_escalations"][i]}
            for i, t in enumerate(tickets)
        ],
        "approval_blocks": [],
        "mandatory_rollbacks": [],
        "pending_comms": [{
            "audience": "external", "required": True,
            "points": pts["pending_comms"][0],
        }],
        "alerts_to_clear": [{
            "alert_id": "A-001",
            "points": pts["alerts_to_clear"][0],
        }],
    }

    return dict(
        task_name=f"gen_data_pipeline_s{seed}_{difficulty}",
        disruption_type="data_pipeline_regression",
        description=(
            f"ETL {pipeline_id} failed. Billing data corrupted. "
            f"VIP {vip} invoices wrong."
        ),
        max_steps=max_steps,
        current_time=f"{rng.randint(2, 9):02d}:00",
        incident_severity=3,
        services=services,
        pipelines=pipelines,
        tickets=tickets,
        alerts=alerts,
        issues=issues,
        hidden_state=dict(
            root_cause="schema_change_unreleased_analytics",
            blast_radius_edges=[["billing-svc", "payment-svc"]],
            approval_states={pipeline_id: "approved"},
            flag_states={},
            policy_flags={},
            stale_telemetry=["billing_reconcile_last_success"],
        ),
        dynamic_events=[],
        max_score=1.0,
    )


# --- Dispatcher ---------------------------------------------------

GENERATED_FAMILIES = ["bad_release", "security_cve", "data_pipeline"]

FAMILY_FNS = {
    "bad_release":   generate_bad_release,
    "security_cve":  generate_security_cve,
    "data_pipeline": generate_data_pipeline,
}


def generate_scenario(family: str, seed: int = 42,
                      difficulty: str = "medium") -> Dict:
    if family not in FAMILY_FNS:
        raise ValueError(f"Unknown family {family!r}. Valid: {GENERATED_FAMILIES}")
    return FAMILY_FNS[family](seed=seed, difficulty=difficulty)
