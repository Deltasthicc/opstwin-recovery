"""
Multi-Objective Scoring Engine
================================
Six pure scoring functions + aggregator. No LLM judges, no state mutation.

Dimensions (weights sum to 1.0):
  service_recovery       (0.35) - fraction of service outages resolved * blast-radius modifier
  customer_outcome       (0.20) - priority-weighted ticket resolution + VIP-first bonus
  security_compliance    (0.15) - policy adherence, CVE handling, safe disclosures
  change_hygiene         (0.10) - no thrashing, no unnecessary rollbacks
  communication_quality  (0.10) - required comms sent, timeliness
  operational_efficiency (0.10) - resolved issues per step, penalize noise

The final scalar is the weighted sum, clamped to [0.01, 0.99]. This is the
score reported on the observation; it is NOT the per-step reward (those come
from issue-resolution points and inspection bonuses in environment.py).
"""
from typing import Dict, Iterable, List, Optional, Set, Tuple


WEIGHTS = {
    "service_recovery": 0.35,
    "customer_outcome": 0.20,
    "security_compliance": 0.15,
    "change_hygiene": 0.10,
    "communication_quality": 0.10,
    "operational_efficiency": 0.10,
}


# --- Dimension 1: service_recovery ---------------------------------

def compute_service_recovery(resolved_outages: int,
                              total_outages: int,
                              blast_radius_assessed: bool,
                              blast_radius_contained: bool) -> float:
    """
    Fraction of service outages resolved, scaled by a blast-radius modifier.

    If the agent never ran ASSESS_BLAST_RADIUS, modifier = 0.85 (uncertainty
    penalty). If assessed but hidden edges remain unexplored, 0.92. Fully
    contained = 1.0.
    """
    if total_outages == 0:
        return 1.0
    base = min(resolved_outages / total_outages, 1.0)
    if not blast_radius_assessed:
        modifier = 0.85
    elif not blast_radius_contained:
        modifier = 0.92
    else:
        modifier = 1.0
    return min(max(base * modifier, 0.0), 1.0)


# --- Dimension 2: customer_outcome ---------------------------------

PRIORITY_WEIGHTS = {"P1": 3.0, "P2": 1.5, "P3": 1.0}


def compute_customer_outcome(tickets: Dict[str, dict],
                              resolved_ticket_ids: Set[str]) -> float:
    """
    Priority-weighted fraction of tickets resolved. VIP tickets count double.
    Bonus if ALL VIP tickets are resolved.
    """
    if not tickets:
        return 1.0

    total_weight = 0.0
    served_weight = 0.0
    vip_tickets: Set[str] = set()

    for tid, t in tickets.items():
        w = PRIORITY_WEIGHTS.get(t.get("priority", "P3"), 1.0)
        if t.get("is_vip"):
            w *= 2.0
            vip_tickets.add(tid)
        # If SLA has expired without resolution, weight drops to zero
        sla = t.get("sla_minutes_remaining")
        if sla is not None and sla <= 0 and tid not in resolved_ticket_ids:
            w = 0.0
        total_weight += w
        if tid in resolved_ticket_ids:
            served_weight += w

    if total_weight == 0:
        return 1.0
    raw = served_weight / total_weight
    if vip_tickets and vip_tickets.issubset(resolved_ticket_ids):
        raw = min(raw + 0.10, 1.0)
    return min(max(raw, 0.0), 1.0)


# --- Dimension 3: security_compliance -----------------------------

def compute_security_compliance(audit_trail: List[Dict],
                                 policy_flags: Dict[str, List[str]],
                                 policy_satisfied: Set[str],
                                 unresolved_cves: int,
                                 unsafe_disclosures: int) -> float:
    """
    Start from 1.0, subtract penalties for:
      - Policy-flagged actions taken without preconditions satisfied
      - Unresolved critical CVEs
      - Unsafe external disclosures

    `policy_satisfied` is the set of policy keys the agent has cleared via
    APPROVE_EXCEPTION, INSPECT_RUNBOOK, or similar acknowledge-style actions.
    """
    score = 1.0

    # Policy violations: action matched a flagged pattern but precondition not met
    for entry in audit_trail:
        action = entry.get("action", "")
        action_upper = action.upper().strip()
        for pattern, required in policy_flags.items():
            if pattern.upper() in action_upper:
                # required is a list of policy tokens that must be in policy_satisfied
                missing = [r for r in required if r not in policy_satisfied]
                if missing:
                    score -= 0.15
                    break  # one violation per action

    score -= 0.10 * max(unresolved_cves, 0)
    score -= 0.20 * max(unsafe_disclosures, 0)

    return min(max(score, 0.0), 1.0)


# --- Dimension 4: change_hygiene ----------------------------------

def compute_change_hygiene(audit_trail: List[Dict],
                            necessary_rollbacks: Set[str],
                            executed_rollbacks: Set[str],
                            pipeline_reruns_on_healthy: int) -> float:
    """
    "Don't panic" dimension. Penalize:
      - Rollbacks that weren't in the necessary set (unnecessary rollback)
      - Duplicate commands (same action, same arg, twice)
      - Pipeline reruns on already-healthy pipelines
    """
    score = 1.0

    unnecessary = executed_rollbacks - necessary_rollbacks
    score -= 0.20 * len(unnecessary)

    # Duplicate action detection (exact string match)
    seen: Dict[str, int] = {}
    duplicates = 0
    for entry in audit_trail:
        a = entry.get("action", "").strip().upper()
        if not a:
            continue
        if a.startswith("REQUEST_INFO") or a.startswith("READ_MESSAGES"):
            continue  # info queries don't count as thrashing
        seen[a] = seen.get(a, 0) + 1
        if seen[a] > 1:
            duplicates += 1
    score -= 0.05 * duplicates

    score -= 0.10 * max(pipeline_reruns_on_healthy, 0)

    return min(max(score, 0.0), 1.0)


# --- Dimension 5: communication_quality ---------------------------

def compute_communication_quality(pending_comms_required: List[Dict],
                                   resolved_comms_keys: Set[str],
                                   draft_comms_log: List[Dict],
                                   max_steps: int) -> float:
    """
    Fraction of required comms sent, with a timeliness bonus if any required
    comm was sent within the first 60% of the episode.

    pending_comms_required: list of issue dicts from scenario["issues"]["pending_comms"]
    resolved_comms_keys: set of audience keys ("external", "internal") already closed
    draft_comms_log: list of {"audience": str, "message": str, "step": int}
    """
    required = [c for c in pending_comms_required if c.get("required", True)]
    if not required:
        return 1.0

    closed = sum(1 for c in required
                 if c.get("audience") in resolved_comms_keys)
    base = closed / len(required)

    timely = any(
        entry.get("step", max_steps) <= 0.6 * max(max_steps, 1)
        for entry in draft_comms_log
    )
    bonus = 0.10 if timely and closed > 0 else 0.0

    return min(base + bonus, 1.0)


# --- Dimension 6: operational_efficiency --------------------------

def compute_operational_efficiency(steps_used: int,
                                    max_steps: int,
                                    total_issues: int,
                                    resolved_issues: int,
                                    unnecessary_actions: int) -> float:
    """
    Resolved issues per step used, normalized against theoretical best,
    then minus per-unnecessary-action penalty.
    """
    if resolved_issues == 0 or max_steps == 0:
        return 0.0
    efficiency = resolved_issues / max(steps_used, 1)
    max_efficiency = total_issues / max(max_steps, 1)
    base = min(efficiency / max_efficiency, 1.0) if max_efficiency > 0 else 0.0
    base -= 0.03 * max(unnecessary_actions, 0)
    return min(max(base, 0.0), 1.0)


# --- Aggregator ---------------------------------------------------

def compute_multi_objective_scores(
        resolved_outages: int,
        total_outages: int,
        blast_radius_assessed: bool,
        blast_radius_contained: bool,
        tickets: Dict[str, dict],
        resolved_ticket_ids: Set[str],
        audit_trail: List[Dict],
        policy_flags: Dict[str, List[str]],
        policy_satisfied: Set[str],
        unresolved_cves: int,
        unsafe_disclosures: int,
        necessary_rollbacks: Set[str],
        executed_rollbacks: Set[str],
        pipeline_reruns_on_healthy: int,
        pending_comms_required: List[Dict],
        resolved_comms_keys: Set[str],
        draft_comms_log: List[Dict],
        steps_used: int,
        max_steps: int,
        total_issues: int,
        resolved_issues: int,
        unnecessary_actions: int,
) -> Dict[str, float]:
    """Compute all 6 dimensions + the weighted final scalar."""
    scores = {
        "service_recovery": compute_service_recovery(
            resolved_outages, total_outages,
            blast_radius_assessed, blast_radius_contained,
        ),
        "customer_outcome": compute_customer_outcome(
            tickets, resolved_ticket_ids,
        ),
        "security_compliance": compute_security_compliance(
            audit_trail, policy_flags, policy_satisfied,
            unresolved_cves, unsafe_disclosures,
        ),
        "change_hygiene": compute_change_hygiene(
            audit_trail, necessary_rollbacks, executed_rollbacks,
            pipeline_reruns_on_healthy,
        ),
        "communication_quality": compute_communication_quality(
            pending_comms_required, resolved_comms_keys,
            draft_comms_log, max_steps,
        ),
        "operational_efficiency": compute_operational_efficiency(
            steps_used, max_steps, total_issues,
            resolved_issues, unnecessary_actions,
        ),
    }

    final = sum(scores[k] * WEIGHTS[k] for k in scores)

    # Inaction cap. If there is real work to resolve and the agent resolved
    # none of it, the episode score is capped at 0.30 regardless of how
    # many dimensions happened to have "nothing to do" in this scenario.
    # Without this, a scenario like false_positive (few issues total) would
    # score ~0.72 on pure inaction because three dimensions return 1.0 on
    # empty work. The cap ensures "doing nothing" is never a success.
    if total_issues > 0 and resolved_issues == 0:
        final = min(final, 0.30)

    scores["weighted_final"] = round(min(max(final, 0.01), 0.99), 4)
    return {k: round(v, 4) for k, v in scores.items()}
