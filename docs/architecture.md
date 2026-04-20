# Architecture Spec

This document covers: Desks, Hidden State, Service Graph, and the full action list.
All components are transplants of airport-ops-recovery primitives with domain changes.

---

## 1. Environment Loop (`server/environment.py`)

Direct port of `AirportRecoveryEnvironment`. Changes:

| Airport | OpsTwin |
|---------|---------|
| `_flights` | `_services` |
| `_gates` | `_pipelines` |
| `_crew` | `_alerts` |
| `_passengers` | `_tickets` |
| `_issues["gate_conflicts"]` | `_issues["service_outages"]` |
| `_issues["passenger_rebookings"]` | `_issues["ticket_escalations"]` |
| `_issues["crew_swaps"]` | `_issues["approval_blocks"]` |
| `_issues["cancellations_needed"]` | `_issues["mandatory_rollbacks"]` |
| `_issues["held_connections"]` | `_issues["pending_comms"]` |

Keep unchanged: `reset()/step()/_obs()/_exec()/_load()`, `MINUTES_PER_STEP = 5`,
`_clock_minutes`, `_done`, `_audit_trail`, `_fired_events`, `_dynamic_events`,
`GATE_COOLDOWN_STEPS` (rename to `PIPELINE_COOLDOWN_STEPS`).

The `_obs()` method should return the current desk's filtered view, not full state.
Only `INCIDENT_COMMAND` desk sees the full summary.

---

## 2. Desks (`server/desks.py`)

Replaces `roles.py`. Five desks with hard command gating. Same pattern:
`is_command_allowed(cmd)` returns `(bool, reason)`.

### Desk Definitions

**INCIDENT_COMMAND**
- Commands: `REQUEST_INFO`, `ESCALATE_TO_IC`, `DONE`, `ASSESS_BLAST_RADIUS`,
  `SWITCH_DESK`, `SEND_MESSAGE`, `READ_MESSAGES`
- Visible: all-desks summary, SLO status, blast radius, full audit trail
- Role: strategic oversight, episode termination, blast radius assessment

**SRE**
- Commands: `RESTART_SERVICE`, `ISOLATE_SERVICE`, `ROLLBACK_DEPLOYMENT`,
  `RUN_MITIGATION`, `REQUEST_INFO`, `INSPECT_RUNBOOK`, `REQUEST_FORECAST`
- Visible: telemetry, service graph (partial), pipeline status
- Role: infrastructure recovery actions

**SECURITY**
- Commands: `QUARANTINE_SERVICE`, `BLOCK_ROLLOUT`, `APPROVE_EXCEPTION`,
  `SCAN_CVE`, `REQUEST_INFO`, `INSPECT_RUNBOOK`
- Visible: security alerts, CVE list, approval queue
- Role: containment, compliance gating

**SUPPORT**
- Commands: `TRIAGE_TICKET`, `MERGE_TICKETS`, `DRAFT_COMMS`, `PRIORITIZE_VIP`,
  `REQUEST_INFO`
- Visible: ticket queue, customer SLAs, VIP flags
- Role: customer-facing resolution, communication drafting

**RELEASE**
- Commands: `RERUN_PIPELINE`, `CANCEL_PIPELINE`, `FLIP_FLAG`, `PAUSE_ROLLOUT`,
  `REQUEST_INFO`, `VERIFY_FLAG`, `CHECK_APPROVAL`
- Visible: CI/CD state, feature flag values, deployment history
- Role: change management, rollout control

### Universal Commands (all desks)

`SWITCH_DESK`, `SEND_MESSAGE`, `READ_MESSAGES`, `INSPECT_RUNBOOK`

### Implementation Notes

- Role switch reward: `+0.01` for first switch to each new desk (same as airport)
- Penalty for wrong-desk commands: `-0.02` with helpful message
- `DeskCoordinator.reset()` clears active desk, message queue, desks_used set
- `is_active` property: False until first `SWITCH_DESK` call

---

## 3. Hidden State (`server/hidden_state.py`)

Replaces `visibility.py` + `UncertaintyLayer`. **This is the signature primitive — do not cut.**

### Hidden Variables Per Episode

Each episode initializes these latent values. They are set in `_load()` but NOT exposed in `_obs()`:

| Key | Type | What it represents |
|-----|------|-------------------|
| `root_cause` | str | actual trigger (e.g., `"flag_checkout_v2"`, not just `"latency"`) |
| `blast_radius_edges` | list[tuple] | hidden service dependency edges not in initial graph |
| `approval_states` | dict[str, str] | `"approved"/"pending"/"stale"` per change_id |
| `flag_states` | dict[str, bool] | actual flag values (may differ from agent's initial view) |
| `policy_flags` | dict[str, list] | actions requiring sign-off before execution |
| `stale_telemetry` | set[str] | metric keys whose displayed value is N minutes delayed |

### Inspection Actions → Revelation

Each inspection action reveals one hidden variable if it hasn't been revealed yet.
New revelation: `+0.02` reward (same as airport's inspect pattern).
Already-revealed: `+0.0` reward, but info is returned again.

| Action | Reveals |
|--------|---------|
| `INSPECT_RUNBOOK <service_id>` | root_cause hint for that service |
| `CHECK_APPROVAL <change_id>` | actual approval_state for that change |
| `VERIFY_FLAG <flag_id>` | actual flag value |
| `REQUEST_FORECAST` | which telemetry keys are stale |
| `ASSESS_BLAST_RADIUS` | one hidden blast_radius_edge per call |

### Delayed Reward Pattern

Some actions score fully ONLY after hidden state is confirmed. Example:
- `ROLLBACK_DEPLOYMENT dep_001` gives partial reward immediately
- If hidden state shows `blast_radius_edges` includes a security service, full reward
  is withheld until `ASSESS_BLAST_RADIUS` confirms the agent checked for this
- This forces genuine investigation rather than greedy action

---

## 4. Service Dependency Graph (`server/graph.py`)

Replaces `NetworkTracker`. Directed graph of service dependencies.

### Visible on Episode Start

A partial topology: 5–8 services, with the edges that are explicitly declared in the scenario.
Format returned by `REQUEST_INFO graph`:
```
Services: auth-svc [DEGRADED], checkout-svc [HEALTHY], payment-svc [HEALTHY] ...
Visible dependencies:
  checkout-svc → auth-svc (auth calls)
  payment-svc → checkout-svc (payment flow)
Hidden: ? edges not yet assessed
```

### Alert Types

| Alert | Meaning |
|-------|---------|
| `cascade_risk` | service A failure will propagate to B within N minutes |
| `shared_dependency` | A and B both depend on the failing component |
| `policy_block` | action X on A triggers mandatory policy review for B |
| `slo_breach_imminent` | service approaching SLO threshold given current trajectory |

### Hidden Edges

1–3 edges per episode are hidden at start. Revealed one per `ASSESS_BLAST_RADIUS` call.
When revealed, emit a dynamic event message in the next `step()` response.

---

## 5. Full Action Reference

```
# Desk navigation
SWITCH_DESK <INCIDENT_COMMAND|SRE|SECURITY|SUPPORT|RELEASE>
SEND_MESSAGE <desk_name> <message>
READ_MESSAGES

# SRE actions
RESTART_SERVICE <service_id>
ISOLATE_SERVICE <service_id>
ROLLBACK_DEPLOYMENT <deploy_id>
RUN_MITIGATION <mitigation_id>

# Security actions
QUARANTINE_SERVICE <service_id>
BLOCK_ROLLOUT <pipeline_id>
APPROVE_EXCEPTION <change_id>
SCAN_CVE <dependency_id>

# Support actions
TRIAGE_TICKET <ticket_id> <priority:P1|P2|P3>
MERGE_TICKETS <ticket_id_1> <ticket_id_2>
DRAFT_COMMS <audience:internal|external> <message>
PRIORITIZE_VIP <ticket_id>

# Release actions
RERUN_PIPELINE <pipeline_id>
CANCEL_PIPELINE <pipeline_id>
FLIP_FLAG <flag_id> <on|off>
PAUSE_ROLLOUT <deploy_id>

# Hidden-state inspection (small reward for new revelations)
INSPECT_RUNBOOK <service_id>
CHECK_APPROVAL <change_id>
VERIFY_FLAG <flag_id>
ASSESS_BLAST_RADIUS
REQUEST_FORECAST

# Info queries
REQUEST_INFO <services|tickets|pipelines|alerts|summary|scoring|audit|graph>

# Control
ESCALATE_TO_IC        # once per episode, strategic hint
DONE
```

### Reward Values per Action Category

| Category | Reward on correct resolution |
|----------|------------------------------|
| Service recovery (required) | `issue["points"]` (normalized, sums to ~1.0) |
| Ticket escalation resolved | `issue["points"]` |
| Approval block cleared | `issue["points"]` |
| Mandatory rollback executed | `issue["points"]` |
| Pending comms sent | `issue["points"]` |
| Hidden state revealed (new) | `+0.02` |
| Invalid command | `-0.02` |
| Harmful action (unnecessary) | `-0.05` |
| Unknown command | `-0.02` |
| Efficiency bonus (finish early) | `min(steps_remaining * 0.01, 0.10)` |

---

## 6. Models (`models.py`)

```python
class OpsAction(BaseModel):
    command: str

class OpsObservation(BaseModel):
    done: bool
    reward: float | None
    current_time: str
    incident_severity: int          # 1-5 (replaces weather_severity)
    incident_description: str
    ops_status: str                 # summary line
    services: list[dict]
    tickets: list[dict]
    pipelines: list[dict]
    alerts: list[dict]
    triage_queue: list[dict]        # unresolved tickets by priority
    graph_alerts: list[dict]        # cascade/dependency alerts
    uncertainty_alerts: list[dict]  # hidden state that was revealed
    active_desk: str
    desk_messages: list[dict]
    pending_issues_count: int
    resolved_issues_count: int
    total_issues_count: int
    message: str
    available_commands: list[str]
    multi_objective_scores: dict
    score: float
    hint: str

class OpsState(BaseModel):
    episode_id: str
    step_count: int
    task_name: str
    disruption_type: str
    total_services: int
    resolved_issues: int
    total_issues: int
    max_steps: int
```
