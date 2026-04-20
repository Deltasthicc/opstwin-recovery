"""
OpsTwin Recovery Arena -- Environment
========================================
Enterprise incident-response environment. Agent coordinates five operational
desks (INCIDENT_COMMAND / SRE / SECURITY / SUPPORT / RELEASE) to recover
service, clear tickets, and satisfy policy constraints within a step budget.

Architecture mirrors the airport-ops-recovery skeleton:
    reset()  -> _load(scenario) -> _obs(msg)
    step()   -> _exec(cmd)       -> handler methods
Handlers return (reward, message). Issue resolution adds to _resolved[key]
and credits the issue's points value.

Per-step reward is the shaped signal the RL trainer optimizes. The
multi-objective score is reported on every observation for visibility and
is the display metric, not the training target.
"""
import copy
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from openenv.core.env_server import Environment
from models import OpsAction, OpsObservation, OpsState
from server.scenarios import SCENARIOS, ALL_TASK_NAMES
from server.scoring import compute_multi_objective_scores
from server.hidden_state import HiddenStateLayer
from server.desks import DeskCoordinator
from server.graph import ServiceDependencyGraph


# --- Policy reference shown via REQUEST_INFO policies -------------

POLICY_REFERENCE = """Enterprise Incident Response Policy:
  - Quarantining a service with active customers requires legal notification.
  - Rollback of a service with hidden dependency edges requires blast radius
    assessment before execution.
  - Stale approvals (>24h) must be refreshed via APPROVE_EXCEPTION before
    the associated pipeline can be rerun.
  - External comms required for any customer-visible data corruption or
    outage lasting >10 minutes.
  - VIP customer SLAs take priority over generic P1 tickets.
  - DONE should be called by INCIDENT_COMMAND after all issues are resolved.
"""


class OpsTwinEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    AVAILABLE_COMMANDS = [
        # Desk navigation
        "SWITCH_DESK <INCIDENT_COMMAND|SRE|SECURITY|SUPPORT|RELEASE>",
        "SEND_MESSAGE <desk> <message>",
        "READ_MESSAGES",
        # SRE
        "RESTART_SERVICE <service_id>",
        "ISOLATE_SERVICE <service_id>",
        "ROLLBACK_DEPLOYMENT <deploy_id>",
        "RUN_MITIGATION <mitigation_id>",
        # Security
        "QUARANTINE_SERVICE <service_id>",
        "BLOCK_ROLLOUT <pipeline_id>",
        "APPROVE_EXCEPTION <change_id>",
        "SCAN_CVE <dependency_id>",
        # Support
        "TRIAGE_TICKET <ticket_id> <P1|P2|P3>",
        "MERGE_TICKETS <tid1> <tid2>",
        "DRAFT_COMMS <internal|external> <message>",
        "PRIORITIZE_VIP <ticket_id>",
        # Release
        "RERUN_PIPELINE <pipeline_id>",
        "CANCEL_PIPELINE <pipeline_id>",
        "FLIP_FLAG <flag_id> <on|off>",
        "PAUSE_ROLLOUT <deploy_id>",
        # Inspection (hidden state reveals)
        "INSPECT_RUNBOOK <service_id>",
        "CHECK_APPROVAL <change_id>",
        "VERIFY_FLAG <flag_id>",
        "ASSESS_BLAST_RADIUS",
        "REQUEST_FORECAST",
        # Info
        "REQUEST_INFO <services|tickets|pipelines|alerts|summary|scoring|audit|graph|uncertainty|policies|triage>",
        # Control
        "ESCALATE_TO_IC  (once per episode, strategic hint)",
        "DONE",
    ]

    MINUTES_PER_STEP = 5
    PIPELINE_COOLDOWN_STEPS = 2

    # -- INIT ---------------------------------------------------
    def __init__(self):
        self._state = OpsState()
        # Core world
        self._services: Dict[str, Dict] = {}
        self._tickets: Dict[str, Dict] = {}
        self._pipelines: Dict[str, Dict] = {}
        self._alerts: Dict[str, Dict] = {}
        # Issues and resolutions
        self._issues: Dict[str, List[Dict]] = {
            "service_outages": [], "ticket_escalations": [],
            "approval_blocks": [], "mandatory_rollbacks": [],
            "pending_comms": [], "alerts_to_clear": [],
        }
        self._resolved: Dict[str, Set[str]] = {
            "service_outages": set(), "ticket_escalations": set(),
            "approval_blocks": set(), "mandatory_rollbacks": set(),
            "pending_comms": set(), "alerts_to_clear": set(),
        }
        # Episode clock and scoring
        self._score = 0.0
        self._max_score = 1.0
        self._max_steps = 14
        self._clock_minutes = 0
        self._start_hour = 14
        self._start_min = 0
        self._description = ""
        self._done = False
        # Bookkeeping
        self._fired_events: Set[int] = set()
        self._dynamic_events: List[Dict] = []
        self._pipeline_cooldowns: Dict[str, int] = {}
        self._draft_comms_log: List[Dict] = []     # {audience, message, step}
        self._executed_rollbacks: Set[str] = set()
        self._pipeline_reruns_on_healthy: int = 0
        self._escalation_used = False
        self._incident_severity = 1
        self._hint = ""
        self._audit_trail: List[Dict] = []
        # Signals for scoring
        self._policy_satisfied: Set[str] = set()
        self._unsafe_disclosures: int = 0
        # Component layers
        self._hidden = HiddenStateLayer()
        self._desks = DeskCoordinator()
        self._graph = ServiceDependencyGraph()
        # Memory hints from previous episodes (populated by postmortem on reset)
        self._memory_hints: List[str] = []

    # -- RESET --------------------------------------------------
    def reset(self, seed: Optional[int] = None,
              episode_id: Optional[str] = None, **kwargs) -> OpsObservation:
        task_name = kwargs.get("task", ALL_TASK_NAMES[0])
        memory_hints = kwargs.get("memory_hints") or []

        # Procedural scenario: task="gen_<family>_s<seed>_<diff>" or task="generated"
        if task_name.startswith("gen_") or task_name == "generated":
            from server.generator import generate_scenario, GENERATED_FAMILIES
            family = kwargs.get("family", GENERATED_FAMILIES[0])
            gen_seed = kwargs.get("seed", seed or 42)
            difficulty = kwargs.get("difficulty", "medium")
            if task_name.startswith("gen_"):
                # Parse "gen_<family>_s<seed>_<diff>"
                parts = task_name.split("_")
                fam_map = {"bad": "bad_release", "security": "security_cve",
                           "data": "data_pipeline"}
                if len(parts) >= 4:
                    family = fam_map.get(parts[1], family)
                    try:
                        gen_seed = int(parts[-2].lstrip("s"))
                    except ValueError:
                        pass
                    difficulty = parts[-1]
            sc = generate_scenario(family, seed=gen_seed, difficulty=difficulty)
        elif task_name not in SCENARIOS:
            task_name = ALL_TASK_NAMES[0]
            sc = SCENARIOS[task_name]
        else:
            sc = SCENARIOS[task_name]
        self._load(sc, episode_id, memory_hints=memory_hints)

        intro_lines = [
            f"INCIDENT -- {sc['disruption_type'].upper().replace('_', ' ')}",
            self._description,
            "",
            f"Severity: {self._incident_severity}/5 | Steps: {self._max_steps} "
            f"| Clock: {self._fmt()}",
            f"Issues to resolve: {self._state.total_issues}",
            "",
            "Tip: Use REQUEST_INFO summary to see the big picture.",
            "Tip: ESCALATE_TO_IC gives a strategic hint (once per episode).",
        ]
        if self._memory_hints:
            intro_lines.append("")
            intro_lines.append("[MEMORY] Lessons from past incidents:")
            for hint in self._memory_hints:
                intro_lines.append(f"  - {hint}")

        return self._obs("\n".join(intro_lines))

    def _load(self, sc: Dict, eid: Optional[str] = None,
              memory_hints: Optional[List[str]] = None):
        """Initialize all episode state from a deep-copied scenario."""
        sc = copy.deepcopy(sc)
        self._services = {s["service_id"]: s for s in sc["services"]}
        self._tickets = {t["ticket_id"]: t for t in sc["tickets"]}
        self._pipelines = {p["pipeline_id"]: p for p in sc["pipelines"]}
        self._alerts = {a["alert_id"]: a for a in sc["alerts"]}
        self._issues = {
            "service_outages": sc["issues"].get("service_outages", []),
            "ticket_escalations": sc["issues"].get("ticket_escalations", []),
            "approval_blocks": sc["issues"].get("approval_blocks", []),
            "mandatory_rollbacks": sc["issues"].get("mandatory_rollbacks", []),
            "pending_comms": sc["issues"].get("pending_comms", []),
            "alerts_to_clear": sc["issues"].get("alerts_to_clear", []),
        }
        self._resolved = {k: set() for k in self._issues}
        self._score = 0.0
        self._max_score = sc.get("max_score", 1.0)
        self._max_steps = sc["max_steps"]
        self._description = sc["description"]
        self._done = False
        self._fired_events = set()
        self._dynamic_events = sc.get("dynamic_events", [])
        self._pipeline_cooldowns = {}
        self._draft_comms_log = []
        self._executed_rollbacks = set()
        self._pipeline_reruns_on_healthy = 0
        self._escalation_used = False
        self._incident_severity = sc.get("incident_severity", 2)
        self._hint = ""
        self._audit_trail = []
        self._policy_satisfied = set()
        self._unsafe_disclosures = 0
        self._memory_hints = list(memory_hints or [])

        self._hidden = HiddenStateLayer()
        self._hidden.reset(sc)
        self._desks = DeskCoordinator()
        self._desks.reset()
        self._graph = ServiceDependencyGraph()
        self._graph.reset(self._services)

        h, m = sc["current_time"].split(":")
        self._start_hour, self._start_min = int(h), int(m)
        self._clock_minutes = 0

        total = sum(len(self._issues[k]) for k in self._issues)
        self._state = OpsState(
            episode_id=eid or str(uuid.uuid4()),
            step_count=0,
            task_name=sc["task_name"],
            disruption_type=sc["disruption_type"],
            total_services=len(self._services),
            resolved_issues=0,
            total_issues=total,
            max_steps=self._max_steps,
        )

    def _fmt(self) -> str:
        t = self._start_hour * 60 + self._start_min + self._clock_minutes
        return f"{(t // 60) % 24:02d}:{t % 60:02d}"

    # -- STEP ---------------------------------------------------
    def step(self, action: OpsAction, timeout_s=None, **kwargs) -> OpsObservation:
        if self._done:
            return self._obs("Episode ended.", force_done=True)

        self._state.step_count += 1
        self._clock_minutes += self.MINUTES_PER_STEP
        self._tick_cooldowns()

        cmd = action.command.strip()
        event_msgs = self._check_events()

        reward, msg = self._exec(cmd)

        # After any action, re-check if any alerts can now clear. This gives
        # us clean false-positive handling (inspection-gated alerts clear once
        # the inspection completes) without every handler having to poke at
        # alerts individually.
        sweep_reward = self._sweep_clear_alerts()
        reward += sweep_reward
        self._score += reward

        self._audit_trail.append({
            "step": self._state.step_count,
            "action": cmd,
            "reward": round(reward, 4),
            "resolved": self._nresolved(),
        })

        if event_msgs:
            msg = "".join(f"[!] {e}\n" for e in event_msgs) + "\n" + msg

        nr = self._nresolved()
        done_all = nr >= self._state.total_issues
        done_steps = self._state.step_count >= self._max_steps
        done_cmd = cmd.upper().startswith("DONE")

        if done_all or done_steps or done_cmd:
            self._done = True
            # Efficiency bonus for finishing early with everything resolved.
            # Add the bonus to BOTH the score and the returned reward so that
            # the RL trainer (which sums per-step rewards) sees it.
            if done_all:
                bonus = min((self._max_steps - self._state.step_count) * 0.01, 0.10)
                self._score += bonus
                reward += bonus
                # Rewrite the most recent audit entry to reflect the bonus
                if self._audit_trail:
                    self._audit_trail[-1]["reward"] = round(reward, 4)
                msg += (f"\n\n[OK] All {self._state.total_issues} issues resolved in "
                        f"{self._state.step_count} steps. Bonus +{bonus:.2f}.")
            elif done_cmd:
                msg += f"\n\nAgent DONE. {nr}/{self._state.total_issues} resolved."
            else:
                msg += f"\n\n[TIME] Out of steps. {nr}/{self._state.total_issues} resolved."

        self._state.resolved_issues = nr
        return self._obs(msg, reward=reward)

    def _tick_cooldowns(self):
        expired = [pid for pid, r in self._pipeline_cooldowns.items() if r <= 1]
        for pid in expired:
            del self._pipeline_cooldowns[pid]
        for pid in self._pipeline_cooldowns:
            self._pipeline_cooldowns[pid] -= 1

    def _check_events(self) -> List[str]:
        msgs: List[str] = []
        for ev in self._dynamic_events:
            step = ev.get("step")
            if step == self._state.step_count and step not in self._fired_events:
                self._fired_events.add(step)
                msgs.append(ev.get("desc", "Event occurred."))
        return msgs

    # -- DISPATCH -----------------------------------------------
    def _exec(self, cmd: str) -> Tuple[float, str]:
        u = cmd.upper().strip()

        # Desk-navigation commands work before the desk system is active
        if u.startswith("SWITCH_DESK"):     return self._cmd_switch_desk(cmd)
        if u.startswith("SEND_MESSAGE"):    return self._cmd_send_message(cmd)
        if u.startswith("READ_MESSAGES"):   return self._cmd_read_messages()

        # Desk-based command validation (only after first SWITCH_DESK)
        allowed, reason = self._desks.is_command_allowed(u)
        if not allowed:
            return (-0.02, f"[DESK] {reason}")

        # SRE
        if u.startswith("RESTART_SERVICE"):    return self._cmd_restart_service(cmd)
        if u.startswith("ISOLATE_SERVICE"):    return self._cmd_isolate_service(cmd)
        if u.startswith("ROLLBACK_DEPLOYMENT"): return self._cmd_rollback(cmd)
        if u.startswith("RUN_MITIGATION"):     return self._cmd_run_mitigation(cmd)
        # Security
        if u.startswith("QUARANTINE_SERVICE"): return self._cmd_quarantine(cmd)
        if u.startswith("BLOCK_ROLLOUT"):      return self._cmd_block_rollout(cmd)
        if u.startswith("APPROVE_EXCEPTION"):  return self._cmd_approve_exception(cmd)
        if u.startswith("SCAN_CVE"):           return self._cmd_scan_cve(cmd)
        # Support
        if u.startswith("TRIAGE_TICKET"):      return self._cmd_triage(cmd)
        if u.startswith("MERGE_TICKETS"):      return self._cmd_merge_tickets(cmd)
        if u.startswith("DRAFT_COMMS"):        return self._cmd_draft_comms(cmd)
        if u.startswith("PRIORITIZE_VIP"):     return self._cmd_prioritize_vip(cmd)
        # Release
        if u.startswith("RERUN_PIPELINE"):     return self._cmd_rerun_pipeline(cmd)
        if u.startswith("CANCEL_PIPELINE"):    return self._cmd_cancel_pipeline(cmd)
        if u.startswith("FLIP_FLAG"):          return self._cmd_flip_flag(cmd)
        if u.startswith("PAUSE_ROLLOUT"):      return self._cmd_pause_rollout(cmd)
        # Inspection
        if u.startswith("INSPECT_RUNBOOK"):    return self._cmd_inspect_runbook(cmd)
        if u.startswith("CHECK_APPROVAL"):     return self._cmd_check_approval(cmd)
        if u.startswith("VERIFY_FLAG"):        return self._cmd_verify_flag(cmd)
        if u.startswith("ASSESS_BLAST_RADIUS"): return self._cmd_assess_blast_radius()
        if u.startswith("REQUEST_FORECAST"):   return self._cmd_request_forecast()
        # Info and control
        if u.startswith("REQUEST_INFO"):       return self._cmd_info(cmd)
        if u.startswith("ESCALATE_TO_IC"):     return self._cmd_escalate()
        if u.startswith("DONE"):               return (0.0, "Finishing incident response.")
        return (-0.02, "Unknown command. Use REQUEST_INFO for help.")

    # ==========================================================
    # Desk navigation
    # ==========================================================
    def _cmd_switch_desk(self, cmd: str) -> Tuple[float, str]:
        p = cmd.split()
        if len(p) < 2:
            return (-0.02,
                    "Usage: SWITCH_DESK <INCIDENT_COMMAND|SRE|SECURITY|SUPPORT|RELEASE>")
        return self._desks.switch_desk(p[1])

    def _cmd_send_message(self, cmd: str) -> Tuple[float, str]:
        p = cmd.split(maxsplit=2)
        if len(p) < 3:
            return (-0.02, "Usage: SEND_MESSAGE <desk> <message>")
        if not self._desks.is_active:
            return (-0.02, "Desk system not active. Use SWITCH_DESK first.")
        return self._desks.send_message(p[1], p[2])

    def _cmd_read_messages(self) -> Tuple[float, str]:
        return self._desks.read_messages()

    # ==========================================================
    # Helpers: issue resolution
    # ==========================================================
    def _try_resolve_service_outage(self, service_id: str, action: str) -> float:
        """If `action` is a valid resolution for an outage, credit its points."""
        for iss in self._issues["service_outages"]:
            if iss["service_id"] != service_id:
                continue
            if service_id in self._resolved["service_outages"]:
                return 0.0
            valid = [v.upper() for v in iss.get("valid_actions", [])]
            if action.upper() in valid:
                self._resolved["service_outages"].add(service_id)
                # Flip status to HEALTHY so graph alerts stop firing
                if service_id in self._services:
                    self._services[service_id]["status"] = "HEALTHY"
                return iss["points"]
        return 0.0

    def _try_resolve_ticket(self, ticket_id: str, action: str) -> float:
        for iss in self._issues["ticket_escalations"]:
            if iss["ticket_id"] != ticket_id:
                continue
            if ticket_id in self._resolved["ticket_escalations"]:
                return 0.0
            valid = [v.upper() for v in iss.get("valid_resolutions", [])]
            if action.upper() in valid:
                self._resolved["ticket_escalations"].add(ticket_id)
                if ticket_id in self._tickets:
                    self._tickets[ticket_id]["status"] = "resolved"
                return iss["points"]
        return 0.0

    def _try_resolve_approval(self, change_id: str, action: str) -> float:
        for iss in self._issues["approval_blocks"]:
            if iss["change_id"] != change_id:
                continue
            if change_id in self._resolved["approval_blocks"]:
                return 0.0
            if action.upper().startswith(iss.get("blocking_action", "").upper()):
                self._resolved["approval_blocks"].add(change_id)
                return iss["points"]
        return 0.0

    def _sweep_clear_alerts(self) -> float:
        """
        Re-check every alerts_to_clear issue against the current world state.
        Called once per step after the action handler runs. This lets:
          - Alerts clear when their service is fixed (via any action).
          - Alerts clear when their required inspections are complete
            and the service is in a non-failing state (for false-positive-
            style scenarios where investigation IS the resolution).

        Returns the total points credited this step.
        """
        total = 0.0
        for iss in self._issues["alerts_to_clear"]:
            aid = iss["alert_id"]
            if aid in self._resolved["alerts_to_clear"]:
                continue
            if aid not in self._alerts:
                continue
            requires = iss.get("requires_inspection", [])
            if requires and not all(r in self._hidden._revealed for r in requires):
                continue
            svc_id = self._alerts[aid].get("service_id")
            svc_status = self._services.get(svc_id, {}).get("status", "HEALTHY")
            # If inspections are required and satisfied: clear regardless of
            # service status. That's the false-positive case.
            # Otherwise, require the service to be healthy/quarantined.
            if requires or svc_status in ("HEALTHY", "QUARANTINED"):
                self._resolved["alerts_to_clear"].add(aid)
                total += iss["points"]
        return total

    def _try_clear_alerts_for_service(self, service_id: str) -> float:
        """Backward-compat wrapper: clears any alerts whose service is now healthy/quarantined.
        Kept for call sites that pass a specific service. Equivalent coverage is now
        provided by the per-step _sweep_clear_alerts pass, so this is a no-op for
        scenarios with requires_inspection preconditions."""
        total = 0.0
        for iss in self._issues["alerts_to_clear"]:
            aid = iss["alert_id"]
            if aid in self._resolved["alerts_to_clear"]:
                continue
            if aid not in self._alerts:
                continue
            if self._alerts[aid].get("service_id") != service_id:
                continue
            requires = iss.get("requires_inspection", [])
            if requires and not all(r in self._hidden._revealed for r in requires):
                continue
            svc_status = self._services.get(service_id, {}).get("status", "HEALTHY")
            if svc_status in ("HEALTHY", "QUARANTINED") or requires:
                self._resolved["alerts_to_clear"].add(aid)
                total += iss["points"]
        return total

    # ==========================================================
    # SRE commands
    # ==========================================================
    def _cmd_restart_service(self, cmd: str) -> Tuple[float, str]:
        p = cmd.split()
        if len(p) < 2: return (-0.02, "Usage: RESTART_SERVICE <service_id>")
        sid = p[1]
        if sid not in self._services:
            return (-0.02, f"Service {sid} not found.")
        svc = self._services[sid]
        old = svc.get("status", "HEALTHY")
        if old == "HEALTHY":
            return (-0.05, f"{sid} already HEALTHY. Unnecessary restart.")
        # Restart is a weak action: mitigates symptoms briefly, doesn't fix root cause
        svc["status"] = "DEGRADED" if old == "FAILED" else old
        return (0.0, f"{sid} restarted. Symptoms may return if root cause not addressed.")

    def _cmd_isolate_service(self, cmd: str) -> Tuple[float, str]:
        p = cmd.split()
        if len(p) < 2: return (-0.02, "Usage: ISOLATE_SERVICE <service_id>")
        sid = p[1]
        if sid not in self._services: return (-0.02, f"Service {sid} not found.")
        self._services[sid]["status"] = "ISOLATED"
        return (0.0, f"{sid} isolated. Traffic drained. Customer impact likely.")

    def _cmd_rollback(self, cmd: str) -> Tuple[float, str]:
        p = cmd.split()
        if len(p) < 2: return (-0.02, "Usage: ROLLBACK_DEPLOYMENT <deploy_id>")
        dep_id = p[1]
        if dep_id not in self._pipelines:
            return (-0.02, f"Deployment {dep_id} not found.")
        self._executed_rollbacks.add(dep_id)
        self._pipelines[dep_id]["status"] = "ROLLED_BACK"
        reward = 0.0

        # Credit if this matches a mandatory_rollback
        for iss in self._issues["mandatory_rollbacks"]:
            if iss.get("pipeline_id") == dep_id and dep_id not in self._resolved["mandatory_rollbacks"]:
                self._resolved["mandatory_rollbacks"].add(dep_id)
                reward += iss["points"]

        # Policy check: action may require blast_radius_assessed
        policy = f"ROLLBACK_DEPLOYMENT {dep_id}"
        required = self._hidden.get_policy_flags().get(policy, [])
        if "blast_radius_assessed" in required and self._hidden.blast_radius_assessed:
            self._policy_satisfied.add("blast_radius_assessed")
        elif "blast_radius_assessed" in required:
            reward -= 0.05
            return (reward,
                    f"{dep_id} rolled back WITHOUT blast-radius assessment. "
                    f"Policy violation.")
        return (reward if reward != 0 else 0.0,
                f"{dep_id} rolled back to previous version.")

    def _cmd_run_mitigation(self, cmd: str) -> Tuple[float, str]:
        p = cmd.split()
        if len(p) < 2: return (-0.02, "Usage: RUN_MITIGATION <mitigation_id>")
        mid = p[1]
        # Mitigations resolve service outages whose valid_action matches
        reward = 0.0
        full_action = f"RUN_MITIGATION {mid}"
        for iss in self._issues["service_outages"]:
            sid = iss["service_id"]
            r = self._try_resolve_service_outage(sid, full_action)
            if r > 0:
                reward += r
                reward += self._try_clear_alerts_for_service(sid)
                return (reward,
                        f"Mitigation {mid} applied to {sid}. Service restored. (+{reward:.2f})")
        return (-0.02, f"Mitigation {mid} did not apply to any active outage.")

    # ==========================================================
    # Security commands
    # ==========================================================
    def _cmd_quarantine(self, cmd: str) -> Tuple[float, str]:
        p = cmd.split()
        if len(p) < 2: return (-0.02, "Usage: QUARANTINE_SERVICE <service_id>")
        sid = p[1]
        if sid not in self._services: return (-0.02, f"Service {sid} not found.")

        # Policy check: quarantining may require legal_notify and break SLAs
        policy_key = f"QUARANTINE_SERVICE {sid}"
        requirements = self._hidden.get_policy_flags().get(policy_key, [])
        breaks_sla = "breaks_vip_sla" in requirements

        self._services[sid]["status"] = "QUARANTINED"
        reward = 0.0

        # If quarantine matches a service outage resolution, credit it
        reward += self._try_resolve_service_outage(sid, f"QUARANTINE_SERVICE {sid}")
        reward += self._try_clear_alerts_for_service(sid)

        if "requires_legal_notify" in requirements and \
                "requires_legal_notify" not in self._policy_satisfied:
            reward -= 0.10  # policy violation (scored via security_compliance too)

        msg = f"{sid} quarantined. Containment complete."
        if breaks_sla:
            msg += " [!] Warning: action likely breaks VIP SLA commitments."
        return (reward, msg)

    def _cmd_block_rollout(self, cmd: str) -> Tuple[float, str]:
        p = cmd.split()
        if len(p) < 2: return (-0.02, "Usage: BLOCK_ROLLOUT <pipeline_id>")
        pid = p[1]
        if pid not in self._pipelines: return (-0.02, f"Pipeline {pid} not found.")
        self._pipelines[pid]["status"] = "BLOCKED"
        return (0.0, f"Rollout {pid} blocked.")

    def _cmd_approve_exception(self, cmd: str) -> Tuple[float, str]:
        p = cmd.split()
        if len(p) < 2: return (-0.02, "Usage: APPROVE_EXCEPTION <change_id>")
        cid = p[1]
        prev = self._hidden.get_approval(cid)
        self._hidden.set_approval(cid, "approved")
        self._policy_satisfied.add("requires_legal_notify")  # exception covers this
        reward = self._try_resolve_approval(cid, f"APPROVE_EXCEPTION {cid}")
        msg = f"Exception approved for {cid} (was: {prev}, now: approved)."
        if reward > 0:
            msg += f" [OK] Approval block cleared. (+{reward:.2f})"
        return (reward, msg)

    def _cmd_scan_cve(self, cmd: str) -> Tuple[float, str]:
        p = cmd.split()
        if len(p) < 2: return (-0.02, "Usage: SCAN_CVE <dependency_id>")
        dep = p[1]
        # Scan confirms exploitation when it matches root cause
        rc = self._hidden.get_root_cause()
        matches = "cve" in rc.lower() and (dep.lower() in rc.lower() or rc.lower() in dep.lower())
        if matches:
            return (0.02, f"CVE scan on {dep}: ACTIVE EXPLOITATION CONFIRMED. "
                          f"Patch application is the right fix.")
        return (0.0, f"CVE scan on {dep}: flagged but low exploit signal.")

    # ==========================================================
    # Support commands
    # ==========================================================
    def _cmd_triage(self, cmd: str) -> Tuple[float, str]:
        p = cmd.split()
        if len(p) < 3: return (-0.02, "Usage: TRIAGE_TICKET <ticket_id> <P1|P2|P3>")
        tid = p[1]
        priority = p[2].upper()
        if priority not in ("P1", "P2", "P3"):
            return (-0.02, f"Invalid priority {priority}. Use P1|P2|P3.")
        if tid not in self._tickets: return (-0.02, f"Ticket {tid} not found.")
        self._tickets[tid]["priority"] = priority
        reward = self._try_resolve_ticket(tid, f"TRIAGE_TICKET {tid} {priority}")
        msg = f"Triaged {tid} -> {priority}."
        if reward > 0:
            msg += f" [OK] Resolved. (+{reward:.2f})"
        return (reward, msg)

    def _cmd_merge_tickets(self, cmd: str) -> Tuple[float, str]:
        p = cmd.split()
        if len(p) < 3: return (-0.02, "Usage: MERGE_TICKETS <tid1> <tid2>")
        t1, t2 = p[1], p[2]
        if t1 not in self._tickets or t2 not in self._tickets:
            return (-0.02, "One or both tickets not found.")
        self._tickets[t1]["status"] = "merged"
        return (0.0, f"Merged {t1} into {t2}.")

    def _cmd_draft_comms(self, cmd: str) -> Tuple[float, str]:
        p = cmd.split(maxsplit=2)
        if len(p) < 3: return (-0.02, "Usage: DRAFT_COMMS <internal|external> <message>")
        audience = p[1].lower()
        message = p[2]
        if audience not in ("internal", "external"):
            return (-0.02, "Audience must be 'internal' or 'external'.")
        self._draft_comms_log.append({
            "audience": audience,
            "message": message,
            "step": self._state.step_count,
        })
        # Credit pending_comms resolution
        reward = 0.0
        for iss in self._issues["pending_comms"]:
            aud = iss.get("audience")
            key = aud  # we key resolved_comms by audience string
            if aud == audience and aud not in self._resolved["pending_comms"]:
                self._resolved["pending_comms"].add(aud)
                reward += iss["points"]
                break
        msg = f"[COMMS/{audience}] {message[:80]}..."
        if reward > 0:
            msg += f" [OK] Required comms closed. (+{reward:.2f})"
        return (reward, msg)

    def _cmd_prioritize_vip(self, cmd: str) -> Tuple[float, str]:
        p = cmd.split()
        if len(p) < 2: return (-0.02, "Usage: PRIORITIZE_VIP <ticket_id>")
        tid = p[1]
        if tid not in self._tickets: return (-0.02, f"Ticket {tid} not found.")
        t = self._tickets[tid]
        if not t.get("is_vip"):
            return (-0.02, f"{tid} is not flagged VIP.")
        t["status"] = "vip_fasttracked"
        reward = self._try_resolve_ticket(tid, f"PRIORITIZE_VIP {tid}")
        msg = f"{tid} fast-tracked as VIP."
        if reward > 0:
            msg += f" [OK] Resolved. (+{reward:.2f})"
        return (reward, msg)

    # ==========================================================
    # Release commands
    # ==========================================================
    def _cmd_rerun_pipeline(self, cmd: str) -> Tuple[float, str]:
        p = cmd.split()
        if len(p) < 2: return (-0.02, "Usage: RERUN_PIPELINE <pipeline_id>")
        pid = p[1]
        if pid not in self._pipelines: return (-0.02, f"Pipeline {pid} not found.")

        if pid in self._pipeline_cooldowns:
            cd = self._pipeline_cooldowns[pid]
            return (-0.02, f"{pid} in cooldown ({cd} steps).")

        # Approval gate: if a stale approval applies to this pipeline, reject
        approval = self._hidden.get_approval(pid)
        if approval in ("stale", "pending"):
            return (-0.05, f"{pid} cannot run: approval is {approval}. "
                           f"Use APPROVE_EXCEPTION {pid} first.")

        prev_status = self._pipelines[pid]["status"]
        self._pipelines[pid]["status"] = "ACTIVE"
        self._pipeline_cooldowns[pid] = self.PIPELINE_COOLDOWN_STEPS

        # Resolve any service_outages that required this rerun
        reward = 0.0
        full_action = f"RERUN_PIPELINE {pid}"
        for iss in self._issues["service_outages"]:
            sid = iss["service_id"]
            r = self._try_resolve_service_outage(sid, full_action)
            if r > 0:
                reward += r
                reward += self._try_clear_alerts_for_service(sid)

        if reward > 0:
            return (reward, f"Pipeline {pid} rerun complete. Service restored. (+{reward:.2f})")

        # Rerun on healthy pipeline is wasteful
        if prev_status == "ACTIVE":
            self._pipeline_reruns_on_healthy += 1
            return (-0.05, f"Pipeline {pid} was already ACTIVE. Unnecessary rerun.")
        return (0.0, f"Pipeline {pid} rerun complete.")

    def _cmd_cancel_pipeline(self, cmd: str) -> Tuple[float, str]:
        p = cmd.split()
        if len(p) < 2: return (-0.02, "Usage: CANCEL_PIPELINE <pipeline_id>")
        pid = p[1]
        if pid not in self._pipelines: return (-0.02, f"Pipeline {pid} not found.")
        self._pipelines[pid]["status"] = "CANCELLED"
        return (0.0, f"Pipeline {pid} cancelled.")

    def _cmd_flip_flag(self, cmd: str) -> Tuple[float, str]:
        p = cmd.split()
        if len(p) < 3: return (-0.02, "Usage: FLIP_FLAG <flag_id> <on|off>")
        fid = p[1]
        val = p[2].lower()
        if val not in ("on", "off"): return (-0.02, "Value must be on|off.")
        new_val = val == "on"
        old_val = self._hidden.get_flag(fid)
        if old_val is None:
            return (-0.02, f"Flag {fid} not found.")
        self._hidden.set_flag(fid, new_val)

        # Any service_outage whose valid_action is exactly this flip?
        reward = 0.0
        full_action = f"FLIP_FLAG {fid} {val}"
        for iss in self._issues["service_outages"]:
            sid = iss["service_id"]
            r = self._try_resolve_service_outage(sid, full_action)
            if r > 0:
                reward += r
                reward += self._try_clear_alerts_for_service(sid)
        msg = f"Flag {fid} -> {val}."
        if reward > 0:
            msg += f" [OK] Service restored. (+{reward:.2f})"
        return (reward, msg)

    def _cmd_pause_rollout(self, cmd: str) -> Tuple[float, str]:
        p = cmd.split()
        if len(p) < 2: return (-0.02, "Usage: PAUSE_ROLLOUT <deploy_id>")
        dep_id = p[1]
        if dep_id not in self._pipelines: return (-0.02, f"Deploy {dep_id} not found.")
        self._pipelines[dep_id]["status"] = "PAUSED"
        return (0.0, f"Rollout {dep_id} paused.")

    # ==========================================================
    # Inspection (reveals hidden state)
    # ==========================================================
    def _cmd_inspect_runbook(self, cmd: str) -> Tuple[float, str]:
        p = cmd.split()
        if len(p) < 2: return (-0.02, "Usage: INSPECT_RUNBOOK <service_id>")
        sid = p[1]
        reward, msg, _ = self._hidden.inspect_runbook(sid)
        return (reward, msg)

    def _cmd_check_approval(self, cmd: str) -> Tuple[float, str]:
        p = cmd.split()
        if len(p) < 2: return (-0.02, "Usage: CHECK_APPROVAL <change_id>")
        cid = p[1]
        reward, msg, _ = self._hidden.check_approval(cid)
        return (reward, msg)

    def _cmd_verify_flag(self, cmd: str) -> Tuple[float, str]:
        p = cmd.split()
        if len(p) < 2: return (-0.02, "Usage: VERIFY_FLAG <flag_id>")
        fid = p[1]
        reward, msg, _ = self._hidden.verify_flag(fid)
        return (reward, msg)

    def _cmd_assess_blast_radius(self) -> Tuple[float, str]:
        reward, msg, _ = self._hidden.assess_blast_radius()
        # Register revealed edges into the visible graph
        for a, b in self._hidden.revealed_edges:
            if b not in self._graph.edges.get(a, set()):
                self._graph.add_hidden_edge(a, b)
        return (reward, msg)

    def _cmd_request_forecast(self) -> Tuple[float, str]:
        reward, msg, _ = self._hidden.request_forecast()
        return (reward, msg)

    # ==========================================================
    # ESCALATE_TO_IC
    # ==========================================================
    def _cmd_escalate(self) -> Tuple[float, str]:
        if self._escalation_used:
            return (0.0, "Incident Commander already consulted this episode.")
        self._escalation_used = True
        self._hint = self._generate_hint()
        return (0.0, f"[IC] {self._hint}")

    def _generate_hint(self) -> str:
        # 1) outstanding service outages
        for iss in self._issues["service_outages"]:
            if iss["service_id"] not in self._resolved["service_outages"]:
                sid = iss["service_id"]
                act = iss.get("valid_actions", ["?"])[0]
                return (f"URGENT: {sid} still degraded. Root cause investigation "
                        f"suggests: {act}. Confirm via INSPECT_RUNBOOK {sid} first.")
        # 2) approval blocks
        for iss in self._issues["approval_blocks"]:
            if iss["change_id"] not in self._resolved["approval_blocks"]:
                return (f"Approval block on {iss['change_id']}. "
                        f"{iss.get('blocking_action', 'APPROVE_EXCEPTION')} clears it.")
        # 3) VIP tickets
        for tid, t in self._tickets.items():
            if t.get("is_vip") and tid not in self._resolved["ticket_escalations"]:
                return f"VIP ticket {tid} still open. PRIORITIZE_VIP {tid}."
        # 4) pending comms
        unsent = [c for c in self._issues["pending_comms"]
                  if c.get("audience") not in self._resolved["pending_comms"]]
        if unsent:
            aud = unsent[0]["audience"]
            return f"Required {aud} comms not sent. DRAFT_COMMS {aud} <message>."
        # 5) else
        return f"All key issues look resolved. Issue DONE when confident."

    # ==========================================================
    # REQUEST_INFO
    # ==========================================================
    def _cmd_info(self, cmd: str) -> Tuple[float, str]:
        parts = cmd.split(maxsplit=1)
        if len(parts) < 2:
            return (0.0,
                    "Subjects: services, tickets, pipelines, alerts, summary, "
                    "scoring, audit, graph, uncertainty, policies, triage, service <ID>")
        subject = parts[1].strip().lower()

        if self._desks.is_active and not self._desks.is_info_subject_allowed(subject):
            return (-0.02, f"'{subject}' not viewable from {self._desks.active_desk}.")

        if subject == "services":
            return (0.0, self._render_services())
        if subject == "tickets":
            return (0.0, self._render_tickets())
        if subject == "pipelines":
            return (0.0, self._render_pipelines())
        if subject == "alerts":
            return (0.0, self._render_alerts())
        if subject == "summary":
            return (0.0, self._render_summary())
        if subject == "scoring":
            return (0.0, self._render_scoring())
        if subject == "audit":
            return (0.0, self._render_audit())
        if subject == "graph":
            return (0.0, self._graph.render_topology(self._services))
        if subject == "uncertainty":
            return (0.0, self._render_uncertainty())
        if subject == "policies":
            return (0.0, POLICY_REFERENCE)
        if subject == "triage":
            return (0.0, self._render_triage())
        if subject.startswith("service "):
            sid = subject.split(maxsplit=1)[1]
            return (0.0, self._render_service_detail(sid))

        return (0.0, f"Unknown subject '{subject}'.")

    def _render_services(self) -> str:
        lines = [f"Services at {self._fmt()}:"]
        for sid, s in sorted(self._services.items()):
            slo = s.get("current_slo")
            slo_s = f" slo={slo:.4f}" if isinstance(slo, float) else ""
            lines.append(f"  {sid}: [{s['status']}]{slo_s}")
        return "\n".join(lines)

    def _render_tickets(self) -> str:
        lines = [f"Tickets at {self._fmt()}:"]
        for tid, t in sorted(self._tickets.items()):
            vip = "[VIP]" if t.get("is_vip") else ""
            sla = (f" SLA:{t['sla_minutes_remaining']}m"
                   if t.get("sla_minutes_remaining") is not None else "")
            lines.append(f"  {tid} [{t['priority']}]{vip}{sla} "
                         f"status={t.get('status', 'open')}: {t['description']}")
        return "\n".join(lines)

    def _render_pipelines(self) -> str:
        lines = [f"Pipelines at {self._fmt()}:"]
        for pid, p in sorted(self._pipelines.items()):
            cd = (f" cooldown={self._pipeline_cooldowns[pid]}"
                  if pid in self._pipeline_cooldowns else "")
            lines.append(f"  {pid}: status={p['status']}{cd} ({p.get('name', pid)})")
        return "\n".join(lines)

    def _render_alerts(self) -> str:
        lines = [f"Alerts at {self._fmt()}:"]
        for aid, a in sorted(self._alerts.items()):
            cleared = "[CLEARED]" if aid in self._resolved["alerts_to_clear"] else ""
            lines.append(f"  {aid}[{a['severity']}]{cleared}: {a['description']} "
                         f"(svc: {a['service_id']})")
        return "\n".join(lines)

    def _render_summary(self) -> str:
        nr = self._nresolved()
        fs = self._final_scalar()
        lines = [
            f"=== Incident Summary at {self._fmt()} ===",
            f"  Progress: {nr}/{self._state.total_issues} "
            f"| Score so far: {fs:.2%} "
            f"| Steps: {self._state.step_count}/{self._max_steps}",
            f"  Severity: {self._incident_severity}/5",
            f"  Active desk: {self._desks.active_desk or 'NONE'}",
            f"  Service outages: {len(self._resolved['service_outages'])}"
            f"/{len(self._issues['service_outages'])}",
            f"  Ticket escalations: {len(self._resolved['ticket_escalations'])}"
            f"/{len(self._issues['ticket_escalations'])}",
            f"  Approval blocks: {len(self._resolved['approval_blocks'])}"
            f"/{len(self._issues['approval_blocks'])}",
            f"  Pending comms: {len(self._resolved['pending_comms'])}"
            f"/{len(self._issues['pending_comms'])}",
            f"  Alerts to clear: {len(self._resolved['alerts_to_clear'])}"
            f"/{len(self._issues['alerts_to_clear'])}",
            f"  Blast radius assessed: {'Yes' if self._hidden.blast_radius_assessed else 'No'}",
            f"  IC hint used: {'Yes' if self._escalation_used else 'No'}",
        ]
        return "\n".join(lines)

    def _render_scoring(self) -> str:
        mo = self._compute_mo_scores()
        lines = ["=== Multi-Objective Scores ==="]
        for k in ("service_recovery", "customer_outcome", "security_compliance",
                  "change_hygiene", "communication_quality", "operational_efficiency",
                  "weighted_final"):
            lines.append(f"  {k.replace('_', ' ').title():24s}: {mo.get(k, 0):.2%}")
        return "\n".join(lines)

    def _render_audit(self) -> str:
        if not self._audit_trail:
            return "No actions taken yet."
        lines = ["=== Audit Trail (last 10) ==="]
        for entry in self._audit_trail[-10:]:
            lines.append(f"  S{entry['step']}: {entry['action']} "
                         f"-> r={entry['reward']:+.2f} resolved={entry['resolved']}")
        return "\n".join(lines)

    def _render_uncertainty(self) -> str:
        summary = self._hidden.reveal_summary()
        alerts = self._hidden.get_uncertainty_alerts()
        lines = ["=== Hidden State ==="]
        for k, v in summary.items():
            lines.append(f"  {k}: {v}")
        if alerts:
            lines.append("  Alerts:")
            for a in alerts:
                lines.append(f"    [{a['severity']}] {a['message']}")
        return "\n".join(lines)

    def _render_triage(self) -> str:
        open_t = [(tid, t) for tid, t in self._tickets.items()
                  if t.get("status", "open") == "open"]
        # Sort by priority and VIP
        order = {"P1": 0, "P2": 1, "P3": 2}
        open_t.sort(key=lambda x: (0 if x[1].get("is_vip") else 1,
                                    order.get(x[1].get("priority", "P3"), 3)))
        lines = [f"Triage queue ({len(open_t)} open):"]
        for tid, t in open_t:
            vip = "[VIP]" if t.get("is_vip") else ""
            lines.append(f"  {tid} [{t['priority']}]{vip}: {t['description']}")
        return "\n".join(lines)

    def _render_service_detail(self, sid: str) -> str:
        if sid not in self._services:
            return f"Service {sid} not found."
        s = self._services[sid]
        dep = ", ".join(s.get("dependencies", [])) or "(none)"
        lines = [
            f"Service {sid}:",
            f"  Status: {s['status']}",
            f"  SLO: target={s.get('slo_target', 1.0):.4f} "
            f"current={s.get('current_slo', 1.0):.4f}",
            f"  Dependencies: {dep}",
        ]
        return "\n".join(lines)

    # ==========================================================
    # Observation construction
    # ==========================================================
    def _nresolved(self) -> int:
        return sum(len(v) for v in self._resolved.values())

    def _final_scalar(self) -> float:
        mo = self._compute_mo_scores()
        return mo.get("weighted_final", 0.01)

    def _compute_mo_scores(self) -> Dict[str, float]:
        necessary_rollbacks = {iss["pipeline_id"]
                               for iss in self._issues.get("mandatory_rollbacks", [])}
        unresolved_cves = sum(1 for a in self._alerts.values()
                              if "cve" in a.get("description", "").lower()
                              and a["alert_id"] not in self._resolved["alerts_to_clear"])
        pending_tickets = {tid: t for tid, t in self._tickets.items()}
        return compute_multi_objective_scores(
            resolved_outages=len(self._resolved["service_outages"]),
            total_outages=len(self._issues["service_outages"]),
            blast_radius_assessed=self._hidden.blast_radius_assessed,
            blast_radius_contained=self._hidden.blast_radius_fully_contained,
            tickets=pending_tickets,
            resolved_ticket_ids=self._resolved["ticket_escalations"],
            audit_trail=self._audit_trail,
            policy_flags=self._hidden.get_policy_flags(),
            policy_satisfied=self._policy_satisfied,
            unresolved_cves=unresolved_cves,
            unsafe_disclosures=self._unsafe_disclosures,
            necessary_rollbacks=necessary_rollbacks,
            executed_rollbacks=self._executed_rollbacks,
            pipeline_reruns_on_healthy=self._pipeline_reruns_on_healthy,
            pending_comms_required=self._issues.get("pending_comms", []),
            resolved_comms_keys=self._resolved["pending_comms"],
            draft_comms_log=self._draft_comms_log,
            steps_used=self._state.step_count,
            max_steps=self._max_steps,
            total_issues=self._state.total_issues,
            resolved_issues=self._nresolved(),
            unnecessary_actions=sum(1 for e in self._audit_trail
                                     if e["reward"] <= -0.02),
        )

    def _obs(self, msg: str = "", reward: Optional[float] = None,
             force_done: bool = False) -> OpsObservation:
        done = self._done or force_done
        fs = self._final_scalar()
        triage = [{"ticket_id": tid, **t}
                  for tid, t in self._tickets.items()
                  if t.get("status", "open") == "open"]
        uncertainty_alerts = self._hidden.get_uncertainty_alerts()
        graph_alerts = self._graph.compute_cascade_alerts(self._services)
        active_desk = self._desks.active_desk or ""
        unread_msgs = [m for m in self._desks.messages if not m.get("read")]

        obs_payload = dict(
            done=done, reward=reward, current_time=self._fmt(),
            incident_severity=self._incident_severity,
            incident_description=self._description,
            ops_status=(f"Task:{self._state.task_name} | {self._fmt()} | "
                        f"Sev:{self._incident_severity}/5 | "
                        f"Issues:{self._nresolved()}/{self._state.total_issues} | "
                        f"Step:{self._state.step_count}/{self._max_steps}"),
            services=[{"id": sid, "status": s["status"],
                       "slo_target": s.get("slo_target"),
                       "current_slo": s.get("current_slo"),
                       "dependencies": s.get("dependencies", [])}
                      for sid, s in sorted(self._services.items())],
            tickets=[{"id": tid, **t} for tid, t in sorted(self._tickets.items())],
            pipelines=[{"id": pid, **p} for pid, p in sorted(self._pipelines.items())],
            alerts=[{"id": aid, **a,
                     "cleared": aid in self._resolved["alerts_to_clear"]}
                    for aid, a in sorted(self._alerts.items())],
            triage_queue=triage,
            graph_alerts=graph_alerts,
            uncertainty_alerts=uncertainty_alerts,
            active_desk=active_desk,
            desk_messages=unread_msgs,
            pending_issues_count=self._state.total_issues - self._nresolved(),
            resolved_issues_count=self._nresolved(),
            total_issues_count=self._state.total_issues,
            message=msg,
            available_commands=self.AVAILABLE_COMMANDS,
            multi_objective_scores=self._compute_mo_scores(),
            score=fs,
            hint=self._hint,
            memory_hints=self._memory_hints,
        )

        # Apply desk-level filtering to the broadcastable lists
        filtered = self._desks.filter_observation(obs_payload)
        return OpsObservation(**filtered)

    @property
    def state(self) -> OpsState:
        return self._state
