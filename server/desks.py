"""
Desk-Based Multi-Agent Coordination
=====================================
Five operational desks with partial visibility into the incident. The agent
switches desks to unlock different command sets and observation views.

Desks:
  INCIDENT_COMMAND - strategic oversight, blast radius, DONE
  SRE              - infrastructure recovery (restart, rollback, mitigate)
  SECURITY         - containment, CVE, approvals
  SUPPORT          - customer-facing (triage, comms, VIP)
  RELEASE          - change management (pipelines, flags, approvals)

Desk system is OPT-IN: becomes active on the first SWITCH_DESK call. Before
that, all commands are available.
"""
from typing import Dict, List, Optional, Set, Tuple


VALID_DESKS = {"INCIDENT_COMMAND", "SRE", "SECURITY", "SUPPORT", "RELEASE"}

# Commands available per desk (in addition to UNIVERSAL_COMMANDS).
DESK_COMMANDS: Dict[str, Set[str]] = {
    "INCIDENT_COMMAND": {
        "REQUEST_INFO", "ESCALATE_TO_IC", "DONE",
        "ASSESS_BLAST_RADIUS",
    },
    "SRE": {
        "RESTART_SERVICE", "ISOLATE_SERVICE", "ROLLBACK_DEPLOYMENT",
        "RUN_MITIGATION", "REQUEST_INFO", "INSPECT_RUNBOOK",
        "REQUEST_FORECAST",
    },
    "SECURITY": {
        "QUARANTINE_SERVICE", "BLOCK_ROLLOUT", "APPROVE_EXCEPTION",
        "SCAN_CVE", "REQUEST_INFO", "INSPECT_RUNBOOK",
    },
    "SUPPORT": {
        "TRIAGE_TICKET", "MERGE_TICKETS", "DRAFT_COMMS", "PRIORITIZE_VIP",
        "REQUEST_INFO",
    },
    "RELEASE": {
        "RERUN_PIPELINE", "CANCEL_PIPELINE", "FLIP_FLAG", "PAUSE_ROLLOUT",
        "REQUEST_INFO", "VERIFY_FLAG", "CHECK_APPROVAL",
    },
}

# Commands available from any desk.
UNIVERSAL_COMMANDS = {
    "SWITCH_DESK", "SEND_MESSAGE", "READ_MESSAGES", "INSPECT_RUNBOOK",
    "DONE",
}

# REQUEST_INFO subjects allowed per desk
DESK_INFO_SUBJECTS: Dict[str, Set[str]] = {
    "INCIDENT_COMMAND": {"services", "tickets", "pipelines", "alerts",
                         "summary", "scoring", "audit", "graph", "uncertainty",
                         "policies"},
    "SRE":              {"services", "pipelines", "alerts", "summary",
                         "graph", "uncertainty"},
    "SECURITY":         {"services", "alerts", "summary", "policies",
                         "uncertainty"},
    "SUPPORT":          {"tickets", "alerts", "summary", "triage"},
    "RELEASE":          {"pipelines", "services", "summary", "uncertainty"},
}


class DeskCoordinator:
    """Desk switching, message passing, command gating."""

    def __init__(self):
        self._active_desk: Optional[str] = None
        self._desk_active = False  # True after first SWITCH_DESK
        self._messages: List[Dict] = []
        self._desks_used: Set[str] = set()

    def reset(self):
        self._active_desk = None
        self._desk_active = False
        self._messages = []
        self._desks_used = set()

    @property
    def active_desk(self) -> Optional[str]:
        return self._active_desk

    @property
    def is_active(self) -> bool:
        return self._desk_active

    @property
    def desks_used(self) -> Set[str]:
        return self._desks_used

    @property
    def messages(self) -> List[Dict]:
        return self._messages

    def switch_desk(self, desk: str) -> Tuple[float, str]:
        """Switch to a new desk. Reward +0.01 for each newly visited desk."""
        desk = desk.upper().strip()
        if desk not in VALID_DESKS:
            return (-0.02,
                    f"Unknown desk '{desk}'. Valid: {', '.join(sorted(VALID_DESKS))}")
        first_time = desk not in self._desks_used
        old_desk = self._active_desk
        self._active_desk = desk
        self._desk_active = True
        self._desks_used.add(desk)

        if old_desk == desk:
            return (0.0, f"Already at {desk} desk.")

        unread = sum(1 for m in self._messages
                     if m.get("to") == desk and not m.get("read"))
        msg = f"Switched to {desk} desk."
        if unread:
            msg += f" [{unread} unread message(s)]"
        return (0.01 if first_time else 0.0, msg)

    def send_message(self, to_desk: str, message: str) -> Tuple[float, str]:
        to_desk = to_desk.upper().strip()
        if to_desk not in VALID_DESKS:
            return (-0.02, f"Unknown desk '{to_desk}'.")
        if to_desk == self._active_desk:
            return (-0.01, "Cannot send message to yourself.")
        self._messages.append({
            "from": self._active_desk or "UNSET",
            "to": to_desk,
            "message": message,
            "read": False,
        })
        return (0.01, f"Message sent to {to_desk}.")

    def read_messages(self) -> Tuple[float, str]:
        if not self._active_desk:
            return (0.0, "No active desk. Use SWITCH_DESK first.")
        unread = [m for m in self._messages
                  if m["to"] == self._active_desk and not m.get("read")]
        if not unread:
            return (0.0, "No unread messages.")
        lines = [f"=== Messages for {self._active_desk} ==="]
        for m in unread:
            lines.append(f"  From {m['from']}: {m['message']}")
            m["read"] = True
        return (0.0, "\n".join(lines))

    def is_command_allowed(self, cmd_upper: str) -> Tuple[bool, str]:
        """Check whether `cmd_upper` is executable from the current desk."""
        if not self._desk_active:
            return (True, "")
        cmd_name = cmd_upper.split()[0] if cmd_upper.strip() else ""
        if cmd_name in UNIVERSAL_COMMANDS:
            return (True, "")
        if self._active_desk and cmd_name in DESK_COMMANDS.get(self._active_desk, set()):
            return (True, "")
        allowed_desks = [d for d, cmds in DESK_COMMANDS.items() if cmd_name in cmds]
        suggestion = f" Try: SWITCH_DESK {allowed_desks[0]}" if allowed_desks else ""
        return (False,
                f"{cmd_name} not available at {self._active_desk} desk.{suggestion}")

    def is_info_subject_allowed(self, subject: str) -> bool:
        if not self._desk_active:
            return True
        allowed = DESK_INFO_SUBJECTS.get(self._active_desk or "", set())
        if subject.startswith("service "):
            return True
        return subject in allowed

    def filter_observation(self, obs_data: dict) -> dict:
        """Return a filtered view of the observation for the current desk."""
        if not self._desk_active or not self._active_desk:
            return obs_data
        filtered = dict(obs_data)
        desk = self._active_desk

        if desk == "SRE":
            # SRE does not see customer-facing tickets
            filtered["tickets"] = []
            filtered["triage_queue"] = []
        elif desk == "SECURITY":
            # Security focuses on alerts + services, not pipelines
            filtered["tickets"] = []
            filtered["triage_queue"] = []
        elif desk == "SUPPORT":
            # Support sees tickets but not internal infra
            filtered["services"] = []
            filtered["pipelines"] = []
            filtered["graph_alerts"] = []
        elif desk == "RELEASE":
            # Release sees pipelines + services but not customer tickets
            filtered["tickets"] = []
            filtered["triage_queue"] = []
        # INCIDENT_COMMAND sees everything
        return filtered

    def coordination_bonus(self, total_issues: int) -> float:
        """Bonus for multi-desk usage on harder scenarios."""
        if not self._desk_active:
            return 0.0
        if len(self._desks_used) >= 3 and total_issues >= 8:
            return 0.03
        if len(self._desks_used) >= 2:
            return 0.01
        return 0.0
