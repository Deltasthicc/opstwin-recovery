"""Quick smoke test to verify the environment loads and scores optimally.

Cross-platform: no hard-coded paths, uses pathlib."""
import sys
from pathlib import Path

# Ensure repo root is on sys.path whether or not `pip install -e .` ran.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import OpsAction
from server.environment import OpsTwinEnvironment

env = OpsTwinEnvironment()
obs = env.reset(task="bad_release")
print(f"RESET OK  task={env.state.task_name} "
      f"issues={env.state.total_issues} max_steps={env.state.max_steps}")
print(f"  desk={obs.active_desk!r}  score={obs.score:.3f}")

# Optimal trajectory. ASSESS_BLAST_RADIUS MUST come before any fix that
# resolves the last issue -- otherwise the episode ends and the blast
# radius never gets assessed, tanking service_recovery to the 0.85 modifier.
sequence = [
    "SWITCH_DESK INCIDENT_COMMAND",
    "ASSESS_BLAST_RADIUS",
    "SWITCH_DESK RELEASE",
    "VERIFY_FLAG checkout_v2_ui",
    "INSPECT_RUNBOOK checkout-svc",
    "FLIP_FLAG checkout_v2_ui off",
    "SWITCH_DESK SUPPORT",
    "PRIORITIZE_VIP T-003",
    "TRIAGE_TICKET T-001 P2",
    "TRIAGE_TICKET T-002 P2",
    "DRAFT_COMMS external Checkout performance restored. Investigation complete.",
    "DONE",
]

total = 0.0
for i, cmd in enumerate(sequence, 1):
    o = env.step(OpsAction(command=cmd))
    r = o.reward or 0.0
    total += r
    print(f"  S{i:>2d} {cmd[:60]:60s} -> r={r:+.3f}  "
          f"resolved={o.resolved_issues_count}/{o.total_issues_count}  done={o.done}")
    if o.done:
        break

print(f"\nSUM step rewards = {total:.3f}")
print(f"FINAL multi-objective score: {o.score:.4f}")
print(f"Dimensions: {o.multi_objective_scores}")
