"""
Expert Solver
==============
Rule-based solver that produces optimal (or near-optimal) action sequences
for the hand-authored scenarios. Used to:

1. Generate gold trajectories stored as .jsonl for supervised warm-start
   or for the RL trainer's reference rollouts.
2. Sanity-check that each scenario is solvable end-to-end.

The solver is hand-authored per scenario because the scenarios themselves
encode a specific "right way" to solve them. It does NOT read the agent's
observation; it simply emits the pre-computed optimal sequence.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Make the repo importable when run as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import OpsAction
from server.environment import OpsTwinEnvironment


OPTIMAL_TRAJECTORIES = {
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
    "false_positive": [
        "SWITCH_DESK SRE",
        "INSPECT_RUNBOOK auth-svc",
        "SCAN_CVE libfoo-0.3",
        "SWITCH_DESK SUPPORT",
        "DRAFT_COMMS internal Both alerts investigated and closed as false positives.",
        "DONE",
    ],
}


def run_trajectory(task: str, out_path: Path | None = None) -> dict:
    """Run the optimal trajectory for one scenario and return a summary."""
    env = OpsTwinEnvironment()
    env.reset(task=task)

    trace = []
    total_reward = 0.0
    for step_idx, cmd in enumerate(OPTIMAL_TRAJECTORIES[task], start=1):
        obs = env.step(OpsAction(command=cmd))
        r = obs.reward or 0.0
        total_reward += r
        trace.append({
            "step": step_idx,
            "action": cmd,
            "reward": round(r, 4),
            "cumulative_reward": round(total_reward, 4),
            "resolved": obs.resolved_issues_count,
            "total_issues": obs.total_issues_count,
            "done": obs.done,
            "score": round(obs.score, 4),
            "message": obs.message[:200],
        })
        if obs.done:
            break

    summary = {
        "task": task,
        "steps_used": len(trace),
        "total_reward": round(total_reward, 4),
        "final_score": round(obs.score, 4),
        "multi_objective": obs.multi_objective_scores,
        "resolved": f"{obs.resolved_issues_count}/{obs.total_issues_count}",
        "trace": trace,
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for entry in trace:
                f.write(json.dumps(entry) + "\n")
        print(f"  wrote {out_path} ({len(trace)} steps)", flush=True)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="One task name, or omit for all.")
    parser.add_argument("--out-dir",
                        default=str(ROOT / "baselines" / "trajectories"),
                        help="Where to write .jsonl trace files.")
    args = parser.parse_args()

    tasks = [args.task] if args.task else list(OPTIMAL_TRAJECTORIES.keys())
    out_dir = Path(args.out_dir)

    print(f"Generating {len(tasks)} expert trajectories...")
    summaries = []
    for task in tasks:
        print(f"[{task}]", flush=True)
        summary = run_trajectory(task, out_path=out_dir / f"{task}.jsonl")
        summaries.append(summary)
        print(f"  steps={summary['steps_used']} "
              f"cum_reward={summary['total_reward']:+.4f} "
              f"final={summary['final_score']:.4f} "
              f"resolved={summary['resolved']}")

    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nSummary written to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
