"""
Postmortem Memory
==================
Lightweight episode memory. At the end of each episode the environment
records a structured postmortem (failure category, first bad action,
missed signal, preferred order, final score). At the start of the next
episode, the top-k most similar past postmortems are retrieved and
injected into the initial observation as text hints.

"Similar" = same scenario family first, then lowest score (so the model
sees the WORST prior outcome on this family as a warning).

Storage is a simple append-only JSONL at
baselines/trajectories/postmortems.jsonl. No embeddings, no vector DB --
the retrieval is keyword match on scenario family. This is deliberate:
hackathon judges want to see the loop works, not the infrastructure
underneath it.
"""
from __future__ import annotations

import datetime
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional


# --- Classification heuristics -------------------------------------

FAILURE_CATEGORIES = {
    "missed_hidden_state": "Agent did not ASSESS_BLAST_RADIUS or inspect root cause before acting.",
    "policy_violation":    "Agent triggered a policy-flagged action without satisfying preconditions.",
    "cascade_ignored":     "Agent resolved a service but ignored downstream cascade risk.",
    "comms_forgotten":     "Agent finished without sending required external/internal communication.",
    "wrong_fix":           "Agent restarted or rolled back when the real fix was a flag/mitigation.",
    "thrashing":           "Agent issued duplicate or contradictory commands.",
    "none":                "Clean run (>=0.8 score).",
}


def classify_failure(summary: Dict) -> str:
    """Pick one failure category from the episode's dimension breakdown."""
    dims = summary.get("multi_objective", {}) or {}
    score = summary.get("final_score", 0.0)
    if score >= 0.8:
        return "none"
    # Lowest-scoring dimension drives classification
    priorities = [
        ("security_compliance", 0.7, "policy_violation"),
        ("service_recovery",    0.5, "missed_hidden_state"),
        ("communication_quality", 0.5, "comms_forgotten"),
        ("change_hygiene",      0.7, "thrashing"),
        ("customer_outcome",    0.5, "cascade_ignored"),
    ]
    for dim_name, threshold, category in priorities:
        if dims.get(dim_name, 1.0) < threshold:
            return category
    return "wrong_fix"


def first_bad_action(trace: List[Dict]) -> str:
    """First action with a negative per-step reward."""
    for entry in trace:
        if entry.get("reward", 0.0) < 0:
            return entry.get("action", "")
    return ""


def preferred_intervention_order(trace: List[Dict]) -> List[str]:
    """Abstract the actions that DID earn positive reward into a concise list."""
    good = []
    for entry in trace:
        if entry.get("reward", 0.0) > 0.01:
            action = entry.get("action", "")
            # Keep the command head (first token) to abstract over IDs
            head = action.split()[0] if action else ""
            if head and head not in good:
                good.append(head)
    return good


# --- Main store class ----------------------------------------------

class PostmortemMemory:
    """Append-only JSONL store + top-k retrieval by scenario family."""

    def __init__(self, store_path: Optional[Path] = None):
        self.store_path = Path(store_path) if store_path else Path(
            __file__).parent.parent / "baselines" / "trajectories" / "postmortems.jsonl"
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, episode_summary: Dict) -> Dict:
        """Append one postmortem. `episode_summary` shape:
            {
              "task": str,                 # scenario family
              "final_score": float,
              "multi_objective": dict,
              "trace": list[dict],         # per-step entries
              "resolved": str,             # "N/M"
            }
        """
        trace = episode_summary.get("trace", [])
        entry = {
            "episode_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "scenario_family": episode_summary.get("task", "unknown"),
            "failure_category": classify_failure(episode_summary),
            "first_bad_action": first_bad_action(trace),
            "missed_signal": self._missed_signal(episode_summary),
            "violated_policy": self._violated_policy(trace),
            "preferred_intervention_order": preferred_intervention_order(trace),
            "final_score": float(episode_summary.get("final_score", 0.0)),
            "steps_used": len(trace),
            "resolved": episode_summary.get("resolved", "?"),
        }
        with self.store_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def retrieve(self, scenario_family: str, k: int = 2) -> List[Dict]:
        """Return the k lowest-scoring past postmortems for this family.

        Worst-case retrieval is intentional: we want the model to see
        what went wrong, not what went right.
        """
        if not self.store_path.exists():
            return []
        entries = []
        with self.store_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        same_family = [e for e in entries
                       if e.get("scenario_family") == scenario_family
                       and e.get("failure_category") != "none"]
        same_family.sort(key=lambda e: e.get("final_score", 1.0))
        return same_family[:k]

    def build_hints(self, scenario_family: str, k: int = 2) -> List[str]:
        """Retrieve past postmortems and render them as one-line hints
        suitable for injection into the initial observation."""
        hints = []
        for pm in self.retrieve(scenario_family, k):
            cat = pm.get("failure_category", "?")
            first_bad = pm.get("first_bad_action", "")
            missed = pm.get("missed_signal", "")
            order = pm.get("preferred_intervention_order", [])
            score = pm.get("final_score", 0.0)
            line = (f"Prior {scenario_family} episode scored {score:.2f}. "
                    f"Failure: {cat}.")
            if first_bad:
                line += f" First bad action: {first_bad}."
            if missed:
                line += f" Missed: {missed}."
            if order:
                line += f" Preferred order: {' -> '.join(order[:5])}."
            hints.append(line)
        return hints

    # --- Helpers ---------------------------------------------------

    def _missed_signal(self, summary: Dict) -> str:
        """Heuristic: if blast radius never assessed, that's the missed signal."""
        trace = summary.get("trace", [])
        assessed = any("ASSESS_BLAST_RADIUS" in (e.get("action") or "").upper()
                       for e in trace)
        inspected = any("INSPECT_RUNBOOK" in (e.get("action") or "").upper()
                        for e in trace)
        comms_drafted = any("DRAFT_COMMS" in (e.get("action") or "").upper()
                            for e in trace)
        missed = []
        if not assessed:
            missed.append("blast radius never assessed")
        if not inspected:
            missed.append("runbook never read")
        if not comms_drafted and summary.get("final_score", 0) < 0.8:
            missed.append("required comms never sent")
        return "; ".join(missed)

    def _violated_policy(self, trace: List[Dict]) -> Optional[str]:
        """Heuristic: detect a negative-reward action that looks policy-like."""
        for entry in trace:
            action = (entry.get("action") or "").upper()
            if entry.get("reward", 0) < -0.04:
                if action.startswith("ROLLBACK_DEPLOYMENT") or \
                        action.startswith("QUARANTINE_SERVICE"):
                    return action
        return None


# --- Convenience: end-of-episode hook from environment -------------

def record_from_env(env, memory: Optional[PostmortemMemory] = None) -> Dict:
    """Call at episode end to build + record a postmortem from env state."""
    if memory is None:
        memory = PostmortemMemory()
    summary = {
        "task": env.state.task_name,
        "final_score": env._final_scalar(),
        "multi_objective": env._compute_mo_scores(),
        "trace": [dict(e) for e in env._audit_trail],
        "resolved": f"{env._nresolved()}/{env.state.total_issues}",
    }
    return memory.record(summary)
