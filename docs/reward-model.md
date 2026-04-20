# Reward Model

Six scoring dimensions, all deterministic — no LLM judge anywhere in the scoring path.
Final scalar = weighted sum, clamped to [0.01, 0.99].

---

## Dimension Weights

| Dimension | Weight | Airport Equivalent |
|-----------|--------|--------------------|
| `service_recovery` | 0.35 | `operational_recovery` (was 0.40) |
| `customer_outcome` | 0.20 | `passenger_fairness` (was 0.20) |
| `security_compliance` | 0.15 | `compliance` (was 0.20) |
| `change_hygiene` | 0.10 | `cost_efficiency` (was 0.10) |
| `communication_quality` | 0.10 | `communication_quality` (was 0.10) |
| `operational_efficiency` | 0.10 | new — replaces nothing |

---

## Dimension Specs

### 1. `service_recovery` (0.35)

```python
def compute_service_recovery(resolved_outages, total_outages, blast_radius_contained):
    """
    resolved_outages: count of service_outage issues resolved
    total_outages: total service_outage issues in scenario
    blast_radius_contained: bool — were all revealed blast-radius edges addressed?
    """
    base = resolved_outages / max(total_outages, 1)
    modifier = 1.0 if blast_radius_contained else 0.85
    return min(base * modifier, 1.0)
```

Notes:
- Partial credit if some outages resolved but not all
- Blast radius modifier only applies after ASSESS_BLAST_RADIUS has been called at least once
- If agent never assesses blast radius, modifier defaults to 0.85 (uncertainty penalty)

---

### 2. `customer_outcome` (0.20)

```python
def compute_customer_outcome(tickets, resolved_ticket_ids):
    """
    tickets: dict of all ticket dicts
    resolved_ticket_ids: set of ticket_ids successfully resolved
    """
    priority_weights = {"P1": 3.0, "P2": 1.5, "P3": 1.0}
    vip_bonus = 0.10  # bonus if all is_vip tickets resolved first

    total_weight = 0.0
    served_weight = 0.0

    for tid, t in tickets.items():
        w = priority_weights.get(t["priority"], 1.0)
        if t.get("is_vip"):
            w *= 2.0
        total_weight += w
        if tid in resolved_ticket_ids:
            served_weight += w

    raw = served_weight / max(total_weight, 1.0)

    # Check VIP-first ordering
    vip_tickets = {tid for tid, t in tickets.items() if t.get("is_vip")}
    if vip_tickets and vip_tickets.issubset(resolved_ticket_ids):
        raw = min(raw + vip_bonus, 1.0)

    return min(max(raw, 0.0), 1.0)
```

Notes:
- VIP tickets (is_vip=True) count double and trigger bonus if handled early
- SLA breach: if a ticket with sla_minutes_remaining runs to 0 unresolved, weight set to 0

---

### 3. `security_compliance` (0.15)

```python
def compute_security_compliance(actions_taken, policy_flags, unresolved_cves, unsafe_disclosures):
    """
    actions_taken: list of (action_str, step_num) tuples from audit trail
    policy_flags: dict mapping action patterns to required conditions
    unresolved_cves: count of critical CVEs not addressed
    unsafe_disclosures: count of DRAFT_COMMS actions that violated disclosure policy
    """
    score = 1.0

    # Penalize policy violations (action taken without required precondition)
    for action, step in actions_taken:
        for pattern, requirements in policy_flags.items():
            if pattern in action:
                # Check if required preconditions were met before this step
                # (e.g., APPROVE_EXCEPTION called before QUARANTINE_SERVICE)
                if not _preconditions_met(requirements, actions_taken, before_step=step):
                    score -= 0.15

    # Penalize unresolved critical CVEs
    score -= 0.10 * unresolved_cves

    # Penalize unsafe external disclosures
    score -= 0.20 * unsafe_disclosures

    return min(max(score, 0.0), 1.0)
```

Notes:
- Policy violation = taking a flagged action without its required precondition in audit trail
- Unsafe disclosure = DRAFT_COMMS external that contradicts policy_flags for that scenario
- If agent never reads policy_flags (no INSPECT_RUNBOOK or CHECK_APPROVAL), they risk blindly
  violating this dimension

---

### 4. `change_hygiene` (0.10)

```python
def compute_change_hygiene(actions_taken, necessary_rollbacks, executed_rollbacks,
                           duplicate_actions, pipeline_reruns):
    """
    necessary_rollbacks: set of pipeline_ids that should be rolled back (from issues dict)
    executed_rollbacks: set of pipeline_ids that were rolled back
    duplicate_actions: count of identical actions taken more than once
    pipeline_reruns: count of RERUN_PIPELINE calls on already-healthy pipelines
    """
    score = 1.0

    # Penalize unnecessary rollbacks (rolled back something not in mandatory_rollbacks)
    unnecessary = executed_rollbacks - necessary_rollbacks
    score -= 0.20 * len(unnecessary)

    # Penalize duplicate actions (thrashing)
    score -= 0.10 * duplicate_actions

    # Penalize unnecessary pipeline reruns
    score -= 0.10 * pipeline_reruns

    return min(max(score, 0.0), 1.0)
```

Notes:
- This is the "don't panic" dimension — rewards deliberate action over thrashing
- Key for False Positive Trap scenario: aggressive rollback tanks this dimension

---

### 5. `communication_quality` (0.10)

```python
def compute_communication_quality(pending_comms_issues, resolved_comms, draft_comms_actions):
    """
    pending_comms_issues: list of required comms from issues dict
    resolved_comms: set of resolved pending_comms issue keys
    draft_comms_actions: list of (audience, message, step_num) tuples
    """
    if not pending_comms_issues:
        return 1.0

    # Base: fraction of required comms sent
    base = len(resolved_comms) / len(pending_comms_issues)

    # Timeliness modifier: comms sent in first 60% of steps score higher
    timely_count = sum(
        1 for _, _, step in draft_comms_actions
        if step <= 0.6 * max_steps
    )
    timeliness_bonus = 0.10 if timely_count > 0 else 0.0

    return min(base + timeliness_bonus, 1.0)
```

Notes:
- Family 3 (Data Quality) has a required external DRAFT_COMMS — this entire dimension
  scores 0 if agent finishes without sending it
- Internal comms count but at lower weight than external

---

### 6. `operational_efficiency` (0.10)

```python
def compute_operational_efficiency(steps_used, max_steps, total_issues, resolved_issues,
                                   unnecessary_actions):
    """
    unnecessary_actions: count of no-op or harmful actions in audit trail
    """
    if resolved_issues == 0:
        return 0.0

    # Efficiency ratio: resolved per step used
    efficiency = resolved_issues / max(steps_used, 1)
    max_efficiency = total_issues / max_steps  # theoretical best

    base = min(efficiency / max_efficiency, 1.0) if max_efficiency > 0 else 0.0

    # Penalty for wasteful actions
    base -= 0.05 * unnecessary_actions

    return min(max(base, 0.0), 1.0)
```

Notes:
- MTTR analogue: faster resolution = higher score, but only relative to total issues
- Unnecessary actions = commands that returned -0.02 in audit trail

---

## Final Scalar

```python
def compute_multi_objective_scores(...) -> dict:
    scores = {
        "service_recovery": compute_service_recovery(...),
        "customer_outcome": compute_customer_outcome(...),
        "security_compliance": compute_security_compliance(...),
        "change_hygiene": compute_change_hygiene(...),
        "communication_quality": compute_communication_quality(...),
        "operational_efficiency": compute_operational_efficiency(...),
    }

    weights = {
        "service_recovery": 0.35,
        "customer_outcome": 0.20,
        "security_compliance": 0.15,
        "change_hygiene": 0.10,
        "communication_quality": 0.10,
        "operational_efficiency": 0.10,
    }

    final = sum(scores[k] * weights[k] for k in scores)
    scores["weighted_final"] = round(min(max(final, 0.01), 0.99), 4)
    return {k: round(v, 4) for k, v in scores.items()}
```

---

## Delayed Reward Pattern (important for training)

Not all good actions reward immediately. Implement this pattern in `step()`:

1. Action is taken → partial reward credited immediately
2. Hidden state revealed later (via inspection) → remaining reward released or withheld

Example: `ROLLBACK_DEPLOYMENT deploy-001`
- Immediate: `+0.20` (mandatory_rollback issue partially resolved)
- After `ASSESS_BLAST_RADIUS` confirms no security exposure: `+0.10` remaining credit
- If ASSESS_BLAST_RADIUS reveals security exposure that rollback worsened: `-0.05` modifier

Implementation: store `_pending_rewards` dict keyed by `(action, hidden_state_key)`.
On each step, check if any pending rewards can be finalized based on current revealed state.

This delayed pattern is what makes the environment genuinely hard for shallow policies.
