"""
Hidden State Layer
==================
Replaces the airport UncertaintyLayer. In OpsTwin the hidden things are
categorical and discrete (a root cause IS something, an approval IS stale or
fresh) rather than noisy numeric estimates. So we store one dict of truths and
a set of revealed keys.

Hidden variables tracked per episode (all set in environment._load()):

  root_cause          : str
      The actual trigger. e.g. "flag_checkout_v2" when the symptom is a
      checkout latency spike. Revealed by INSPECT_RUNBOOK on the affected
      service.

  blast_radius_edges  : list[tuple[str, str]]
      Service-to-service edges that exist in reality but are NOT in the
      visible dependency graph. Revealed one at a time by
      ASSESS_BLAST_RADIUS. These are what make aggressive actions backfire:
      rolling back a deploy fixes the visible problem but cascades through
      a hidden edge.

  approval_states     : dict[str, str]
      Per change_id: "approved" | "pending" | "stale". Agent sees "unknown"
      until CHECK_APPROVAL. A stale approval means the patch was signed off
      48h ago and needs refreshing before it can be deployed.

  flag_states         : dict[str, bool]
      Per flag_id: the actual value. Visible initial value may be wrong
      (cached telemetry). VERIFY_FLAG reveals the real value.

  policy_flags        : dict[str, list[str]]
      Per action pattern: list of required preconditions before that action
      is safe. e.g. "QUARANTINE_SERVICE auth-svc" may require
      "requires_legal_notify" first. Revealed by INSPECT_RUNBOOK on that
      service, or by scoring if the agent violates.

  stale_telemetry     : set[str]
      Set of metric keys whose displayed value lags reality. Revealed by
      REQUEST_FORECAST.
"""
from typing import Any, Dict, List, Optional, Set, Tuple


class HiddenStateLayer:
    """Tracks hidden-vs-revealed categorical truth per episode."""

    def __init__(self):
        self._truth: Dict[str, Any] = {}
        self._revealed: Set[str] = set()
        # Track which blast-radius edges have been revealed (incremental).
        self._hidden_edges_remaining: List[Tuple[str, str]] = []
        self._hidden_edges_revealed: List[Tuple[str, str]] = []

    def reset(self, scenario: dict):
        """Initialize from a scenario's `hidden_state` block."""
        self._truth = {}
        self._revealed = set()
        self._hidden_edges_remaining = []
        self._hidden_edges_revealed = []

        hs = scenario.get("hidden_state", {}) or {}

        # Scalar categorical truths
        if "root_cause" in hs:
            self._truth["root_cause"] = hs["root_cause"]
        self._truth["approval_states"] = dict(hs.get("approval_states", {}))
        self._truth["flag_states"] = dict(hs.get("flag_states", {}))
        self._truth["policy_flags"] = {
            k: list(v) for k, v in (hs.get("policy_flags", {}) or {}).items()
        }
        self._truth["stale_telemetry"] = set(hs.get("stale_telemetry", []) or [])

        # Blast-radius edges are a queue — revealed one per ASSESS call.
        raw_edges = hs.get("blast_radius_edges", []) or []
        self._hidden_edges_remaining = [
            (e[0], e[1]) if isinstance(e, (list, tuple)) and len(e) >= 2 else (str(e), "?")
            for e in raw_edges
        ]

    # --- Inspection primitives -----------------------------------

    def inspect_runbook(self, service_id: str) -> Tuple[float, str, bool]:
        """
        INSPECT_RUNBOOK <service_id> reveals the root cause hint (if the
        service is the one the root cause touches) and any policy_flags
        that apply to actions on this service.

        Returns (reward, message, was_new_reveal).
        """
        key = f"runbook:{service_id}"
        was_new = key not in self._revealed
        self._revealed.add(key)

        root_cause = self._truth.get("root_cause", "")
        # Simple heuristic: if service_id appears as a substring in the root
        # cause identifier, the runbook "confirms" it. Otherwise generic.
        lines = [f"[RUNBOOK {service_id}]"]
        if root_cause and service_id.replace("-svc", "").replace("-pipeline", "") in root_cause:
            lines.append(f"  Root cause identified: {root_cause}")
            lines.append(f"  Revealed. Apply the corresponding remediation.")
        elif root_cause:
            lines.append(f"  No direct root cause match for {service_id}.")
            lines.append(f"  (Root cause is: {root_cause}. Try inspecting the affected service.)")
        else:
            lines.append("  No known issues documented for this service.")

        # Also leak any policy flags that target this service
        pol = self._truth.get("policy_flags", {})
        relevant_pols = [p for p in pol.keys() if service_id in p]
        if relevant_pols:
            lines.append("  Policy constraints on this service:")
            for p in relevant_pols:
                lines.append(f"    {p} requires: {', '.join(pol[p])}")

        reward = 0.02 if was_new and root_cause else 0.0
        return (reward, "\n".join(lines), was_new)

    def check_approval(self, change_id: str) -> Tuple[float, str, bool]:
        """
        CHECK_APPROVAL <change_id> reveals the actual approval state.

        Returns (reward, message, was_new_reveal).
        """
        key = f"approval:{change_id}"
        was_new = key not in self._revealed
        self._revealed.add(key)

        approvals = self._truth.get("approval_states", {})
        state = approvals.get(change_id, "unknown")
        if state == "unknown":
            return (0.0, f"[APPROVAL {change_id}] No record found. No approval required.", was_new)

        msg = f"[APPROVAL {change_id}] Current state: {state.upper()}."
        if state == "stale":
            msg += "\n  [!] Approval expired. Needs APPROVE_EXCEPTION to refresh before deploy."
        elif state == "pending":
            msg += "\n  [!] Awaiting approver. Cannot deploy until approved."
        elif state == "approved":
            msg += "\n  Cleared for deploy."

        # Reward only when we reveal non-trivial state for the first time
        reward = 0.02 if was_new and state in ("stale", "pending") else 0.0
        return (reward, msg, was_new)

    def verify_flag(self, flag_id: str) -> Tuple[float, str, bool]:
        """
        VERIFY_FLAG <flag_id> reveals the actual flag value.

        Returns (reward, message, was_new_reveal).
        """
        key = f"flag:{flag_id}"
        was_new = key not in self._revealed
        self._revealed.add(key)

        flags = self._truth.get("flag_states", {})
        if flag_id not in flags:
            return (0.0, f"[FLAG {flag_id}] No flag record found.", was_new)

        val = flags[flag_id]
        msg = f"[FLAG {flag_id}] Actual state: {'ON' if val else 'OFF'}."
        # If this flag is the root cause, mention it
        if self._truth.get("root_cause", "").endswith(flag_id) or flag_id in self._truth.get("root_cause", ""):
            msg += "\n  [!] This flag is the likely root cause. Flipping it should resolve symptoms."
        reward = 0.02 if was_new else 0.0
        return (reward, msg, was_new)

    def request_forecast(self) -> Tuple[float, str, bool]:
        """
        REQUEST_FORECAST reveals which telemetry metrics are stale.

        Returns (reward, message, was_new_reveal).
        """
        key = "forecast:global"
        was_new = key not in self._revealed
        self._revealed.add(key)

        stale = self._truth.get("stale_telemetry", set())
        if not stale:
            return (0.0, "[FORECAST] All telemetry fresh.", was_new)
        lines = ["[FORECAST] Stale telemetry detected:"]
        for key in sorted(stale):
            lines.append(f"  {key}: displayed value lags reality")
        reward = 0.02 if was_new else 0.0
        return (reward, "\n".join(lines), was_new)

    def assess_blast_radius(self) -> Tuple[float, str, bool]:
        """
        ASSESS_BLAST_RADIUS reveals ONE hidden dependency edge per call.

        Returns (reward, message, was_new_reveal).
        """
        key = "blast_radius:called"
        was_new_call = key not in self._revealed
        self._revealed.add(key)

        if not self._hidden_edges_remaining:
            msg = "[BLAST RADIUS] All hidden dependencies surfaced. No further edges."
            if self._hidden_edges_revealed:
                msg += "\n  Revealed so far:"
                for a, b in self._hidden_edges_revealed:
                    msg += f"\n    {a} -> {b}"
            # Small credit for still calling it to close the loop
            reward = 0.01 if was_new_call else 0.0
            return (reward, msg, was_new_call)

        edge = self._hidden_edges_remaining.pop(0)
        self._hidden_edges_revealed.append(edge)
        msg = (f"[BLAST RADIUS] Hidden dependency uncovered: "
               f"{edge[0]} -> {edge[1]}.\n"
               f"  Actions on {edge[0]} may now cascade to {edge[1]}.")
        return (0.02, msg, True)

    # --- Read-only introspection -------------------------------

    @property
    def revealed_edges(self) -> List[Tuple[str, str]]:
        return list(self._hidden_edges_revealed)

    @property
    def hidden_edges_remaining(self) -> List[Tuple[str, str]]:
        return list(self._hidden_edges_remaining)

    @property
    def blast_radius_assessed(self) -> bool:
        """True if ASSESS_BLAST_RADIUS was called at least once."""
        return "blast_radius:called" in self._revealed

    @property
    def blast_radius_fully_contained(self) -> bool:
        """
        True if the agent has surfaced every hidden edge. Scoring uses this
        as the blast-radius modifier for service_recovery.
        """
        return self.blast_radius_assessed and not self._hidden_edges_remaining

    def get_root_cause(self) -> str:
        return self._truth.get("root_cause", "")

    def get_approval(self, change_id: str) -> str:
        return self._truth.get("approval_states", {}).get(change_id, "unknown")

    def get_flag(self, flag_id: str) -> Optional[bool]:
        return self._truth.get("flag_states", {}).get(flag_id, None)

    def set_flag(self, flag_id: str, value: bool):
        """Called when FLIP_FLAG is executed to update the hidden truth."""
        if "flag_states" not in self._truth:
            self._truth["flag_states"] = {}
        self._truth["flag_states"][flag_id] = value

    def set_approval(self, change_id: str, state: str):
        """Called when APPROVE_EXCEPTION refreshes a stale approval."""
        if "approval_states" not in self._truth:
            self._truth["approval_states"] = {}
        self._truth["approval_states"][change_id] = state

    def get_policy_flags(self) -> Dict[str, List[str]]:
        return dict(self._truth.get("policy_flags", {}))

    def get_uncertainty_alerts(self) -> List[Dict]:
        """Return alerts for hidden state the agent should investigate."""
        alerts = []
        if self._hidden_edges_remaining and not self.blast_radius_assessed:
            alerts.append({
                "kind": "unknown_blast_radius",
                "severity": "high",
                "message": "Hidden dependency edges exist. Run ASSESS_BLAST_RADIUS.",
            })
        # Alert if any approvals are stale and unchecked
        approvals = self._truth.get("approval_states", {})
        for cid, state in approvals.items():
            key = f"approval:{cid}"
            if state in ("stale", "pending") and key not in self._revealed:
                alerts.append({
                    "kind": "unchecked_approval",
                    "severity": "medium",
                    "change_id": cid,
                    "message": f"Approval {cid} not yet checked.",
                })
        return alerts

    def reveal_summary(self) -> Dict:
        """Summary shown by REQUEST_INFO uncertainty."""
        return {
            "root_cause_known": "root_cause" in self._truth and any(
                r.startswith("runbook:") for r in self._revealed
            ),
            "blast_radius_assessed": self.blast_radius_assessed,
            "hidden_edges_remaining": len(self._hidden_edges_remaining),
            "hidden_edges_revealed": len(self._hidden_edges_revealed),
            "approvals_checked": sum(
                1 for r in self._revealed if r.startswith("approval:")
            ),
            "flags_verified": sum(
                1 for r in self._revealed if r.startswith("flag:")
            ),
        }
