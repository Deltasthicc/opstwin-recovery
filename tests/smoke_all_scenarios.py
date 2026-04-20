"""Run optimal trajectories for all three core scenarios."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import OpsAction
from server.environment import OpsTwinEnvironment


TRAJECTORIES = {
    "bad_release": [
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
    ],
    "security_cve": [
        "SWITCH_DESK INCIDENT_COMMAND",
        "ASSESS_BLAST_RADIUS",
        "SWITCH_DESK SECURITY",
        "SCAN_CVE libcurl-8.4.0",
        "CHECK_APPROVAL deploy-auth-patch-1.8.2",
        "APPROVE_EXCEPTION deploy-auth-patch-1.8.2",
        "SWITCH_DESK RELEASE",
        "RERUN_PIPELINE deploy-auth-patch-1.8.2",
        "SWITCH_DESK SUPPORT",
        "PRIORITIZE_VIP T-002",
        "TRIAGE_TICKET T-001 P1",
        "TRIAGE_TICKET T-003 P2",
        "DRAFT_COMMS internal CVE patched. No data exfil confirmed. Auth restored.",
        "DONE",
    ],
    "data_pipeline_regression": [
        "SWITCH_DESK INCIDENT_COMMAND",
        "ASSESS_BLAST_RADIUS",
        "SWITCH_DESK SRE",
        "INSPECT_RUNBOOK analytics-pipeline",
        "RUN_MITIGATION fix-schema-transform",
        "SWITCH_DESK RELEASE",
        "RERUN_PIPELINE etl-pipeline-nightly",
        "SWITCH_DESK SUPPORT",
        "PRIORITIZE_VIP T-004",
        "TRIAGE_TICKET T-001 P1",
        "TRIAGE_TICKET T-003 P1",
        "DRAFT_COMMS external Billing data restored. Invoices reprocessing. ETA 30 min.",
        "DONE",
    ],
}


def run(task):
    env = OpsTwinEnvironment()
    env.reset(task=task)
    total = 0.0
    for i, cmd in enumerate(TRAJECTORIES[task], 1):
        o = env.step(OpsAction(command=cmd))
        total += (o.reward or 0.0)
        if o.done:
            break
    mo = o.multi_objective_scores
    print(f"{task:30s}  sum_rewards={total:+.3f}  final={mo.get('weighted_final', 0):.4f}  "
          f"resolved={o.resolved_issues_count}/{o.total_issues_count}")
    print(f"  dims: recov={mo.get('service_recovery', 0):.2f} "
          f"cust={mo.get('customer_outcome', 0):.2f} "
          f"sec={mo.get('security_compliance', 0):.2f} "
          f"hyg={mo.get('change_hygiene', 0):.2f} "
          f"comm={mo.get('communication_quality', 0):.2f} "
          f"eff={mo.get('operational_efficiency', 0):.2f}")

for t in TRAJECTORIES:
    run(t)


# Also verify a FAILED baseline scores low
print("\n--- FAILED baseline: random noisy actions ---")
env = OpsTwinEnvironment()
env.reset(task="bad_release")
bad_seq = ["RESTART_SERVICE checkout-svc", "RESTART_SERVICE checkout-svc",
           "ROLLBACK_DEPLOYMENT deploy-checkout-v2.3.1",
           "RESTART_SERVICE checkout-svc", "DONE"]
for cmd in bad_seq:
    o = env.step(OpsAction(command=cmd))
    if o.done: break
print(f"bad_release baseline: final={o.multi_objective_scores.get('weighted_final', 0):.4f} "
      f"resolved={o.resolved_issues_count}/{o.total_issues_count}")
