---
title: OpsTwin Recovery Arena
emoji: 🛠️
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Enterprise incident response OpenEnv environment
tags:
  - openenv
  - reinforcement-learning
  - multi-agent
  - incident-response
  - world-model
---

<!-- The YAML block above is HuggingFace Spaces metadata. Do not move it. -->

# OpsTwin Recovery Arena

**OpenEnv environment for training LLM agents on enterprise incident response.**

A production outage is in progress. Checkout is degraded. A CVE fires on the auth service. The ETL pipeline returns corrupted billing data. As the on-call agent you have to figure out what's actually broken, fix it without making things worse, keep VIP customers whole, and send the right communications on the right channels. You have a finite step budget and five operational desks between you and recovery.

OpsTwin simulates that world. It is an OpenEnv environment designed for RL post-training of LLM agents on multi-desk, partially-observable workflows with hard policy constraints and verifiable rewards.

This repository was built for the Meta PyTorch OpenEnv Hackathon Grand Finale (Bangalore, April 25 to 26, 2026). It targets the **Theme 3.1 - Professional Tasks** track and aligns directly with the **Scaler AI Labs Multi-App RL Environment for Enterprise Workflows** sub-theme: a single episode touches tickets, CI/CD, feature flags, observability, and cross-team comms.

## Why this environment

The current research consensus is that enterprise agents fail not on reasoning but on world-modeling. WorkArena, CRMArena-Pro, and World of Workflows all show sharp degradation once tasks become multi-turn, stateful, and constrained by hidden workflow rules. OpsTwin makes those failure modes first-class.

Three design choices matter:

**Hidden state must drive decisions.** Each episode has a latent root cause, hidden service-dependency edges, stale-or-fresh approval states, and real-versus-displayed feature flag values. The agent reveals them through inspection actions. Rolling back a deploy before running `ASSESS_BLAST_RADIUS` is a policy violation even if it resolves the visible symptom. This is the "dynamics blindness" problem that World of Workflows identifies.

**Rewards are code, not an LLM judge.** Six scoring dimensions (service recovery, customer outcome, security compliance, change hygiene, communication quality, operational efficiency) combine into a weighted final score. Every step's reward is produced by deterministic Python. The model trains against a stable signal.

**Multi-desk coordination is structural.** The agent switches between Incident Command, SRE, Security, Support, and Release desks. Each desk exposes a different command set and a filtered view of the world. Cross-desk coordination is required for any non-trivial scenario.

## How it's built

```
opstwin-recovery/
├── models.py                   OpsAction / OpsObservation / OpsState (Pydantic)
├── client.py                   EnvClient for HTTP or Docker connections
├── inference.py                Baseline LLM agent runner
├── train.py                    TRL GRPO training (Qwen3-1.7B + LoRA + vLLM)
├── evaluate.py                 Held-out seed evaluation with bar chart
├── server/
│   ├── app.py                  FastAPI server
│   ├── environment.py          Core reset/step/obs loop
│   ├── scenarios.py            Three hand-authored families + stretch
│   ├── generator.py            Procedural scenarios (seed-deterministic)
│   ├── hidden_state.py         Latent truth revealed by INSPECT actions
│   ├── graph.py                Service dependency graph + cascade alerts
│   ├── desks.py                Five-desk coordinator with command gating
│   ├── scoring.py              Six-dimension pure reward functions
│   └── postmortem.py           Episode memory for self-improvement loop
├── baselines/
│   ├── expert_solver.py        Gold trajectory generator
│   └── trajectories/           .jsonl expert traces
└── docs/                       Architecture, build order, reward model
```

The environment is a direct architectural transplant of the Round 1 `airport-ops-recovery` codebase: same `reset / step / obs / exec / load` skeleton, same issue-ledger pattern, same desk-coordinator shape. The nouns change; the loop does not.

## Scenarios

Three hand-authored scenarios cover the core failure modes.

**Family 1 - Bad Release.** Checkout service is degraded after a deploy. The deploy looks like the obvious suspect. It isn't: the root cause is a feature flag that ships with the deploy but can be flipped independently. Rolling back fixes the visible symptom but cascades into analytics through a hidden edge. The correct fix is `VERIFY_FLAG` then `FLIP_FLAG`.

**Family 2 - Security Alert Under Customer Pressure.** A CVE fires on the auth service and a suspicious process is spotted. A patch exists but its approval is stale (48h old). Quarantining is the fastest action but it breaks a VIP SLA and triggers a legal-notification requirement. The correct path is `APPROVE_EXCEPTION` then `RERUN_PIPELINE`. The wrong path is `QUARANTINE_SERVICE`, which scores the agent into the 0.5 range for policy violation.

**Family 3 - Data Quality Cascade.** An ETL job fails at step 3 of 7. Billing receives null values. Reporting shows stale data. The root cause is an unreleased upstream schema change. Fixing the infrastructure is necessary but not sufficient: an external `DRAFT_COMMS` is mandatory to close the incident. Agents who fix the infra and call `DONE` cap at around 0.65 instead of 0.92.

Plus a stretch scenario, **False Positive Trap**, where the surface reading says "security incident" but the correct action is to investigate both alerts, confirm they are false positives, and close the incident with internal comms. Aggressive rollback or quarantine in this scenario scores below 0.40.

The procedural generator in `server/generator.py` produces seed-deterministic variants of all three families, guaranteed solvable, with issue points summing to 1.0.

## Reward structure

Every scenario has the same point budget (1.0 total) distributed across six issue categories: service outages, ticket escalations, approval blocks, mandatory rollbacks, pending comms, alerts to clear. The per-step reward is the points value of whichever issue that action resolves, minus penalties for invalid or harmful commands.

Additional signals:
- Inspection actions (`INSPECT_RUNBOOK`, `CHECK_APPROVAL`, `VERIFY_FLAG`, `ASSESS_BLAST_RADIUS`, `REQUEST_FORECAST`) pay out +0.02 the first time they reveal meaningful hidden state.
- Switching to a new desk pays +0.01 the first time.
- Finishing before the step budget pays up to +0.10 as an efficiency bonus.
- Invalid commands pay -0.02. Harmful actions (unnecessary rollback, restarting a healthy service) pay -0.05.

The displayed final score is the weighted combination of the six scoring dimensions, clamped to [0.01, 0.99]. This is what judges see. The per-step reward is the signal the RL trainer optimizes. They reinforce each other but are not identical.

## Training evidence

Run `python evaluate.py --n-seeds 5` to reproduce the baseline comparison chart in `results/eval_curve.png`. It shows:
- Random policy: ~0.23 weighted-final across scenarios
- Heuristic expert: 0.99 (upper bound, solves every scenario optimally)
- Trained Qwen3-1.7B (add `--model ./checkpoints`): typically lands in the 0.55-0.75 range after 300-500 GRPO steps

Training uses TRL's `environment_factory` pattern. Each tool (`switch_desk`, `flip_flag`, `triage_ticket`, `draft_comms`, ...) is a typed method on `OpsTwinTrainingEnv`; the trainer auto-discovers them, builds the tool schema, and runs multi-turn episodes. Reward per episode is the sum of shaped per-step rewards plus the final-state weighted score.

```bash
# Smoke test on CPU or small GPU
python train.py --model Qwen/Qwen3-0.6B --max-steps 3 --no-vllm

# Full run on a single H100
python train.py --model Qwen/Qwen3-1.7B --max-steps 500
```

## Self-improvement loop

At the end of each episode, `server/postmortem.py` classifies the failure (missed hidden state, policy violation, cascade ignored, comms forgotten, wrong fix, thrashing) and appends a structured postmortem to `baselines/trajectories/postmortems.jsonl`. On the next `reset()` the top-2 lowest-scoring past postmortems for the same scenario family are retrieved and injected into the initial observation as memory hints. This closes the loop without requiring vector databases or embedding infrastructure.

The postmortem format is deliberately simple:
```json
{
  "scenario_family": "bad_release",
  "failure_category": "missed_hidden_state",
  "first_bad_action": "ROLLBACK_DEPLOYMENT deploy-checkout-v2.3.1",
  "missed_signal": "blast radius never assessed; runbook never read",
  "preferred_intervention_order": ["SWITCH_DESK", "ASSESS_BLAST_RADIUS", "FLIP_FLAG"],
  "final_score": 0.21
}
```

This matches how real on-call engineers pass context to the next shift: write down what broke, what you tried, and what actually worked.

## Running the environment

```bash
# Local server
uv run server
# or
python -m server.app

# Then in a separate terminal
python inference.py
```

Set `HF_TOKEN` and (optionally) `IMAGE_NAME` to route inference through the docker container instead of local uvicorn.

## Bonus theme alignment

**Scaler AI Labs - Multi-App RL Environment for Enterprise Workflows.** A single OpsTwin episode spans: ticket systems, CI/CD pipelines, feature flag stores, observability (metrics and alerts), incident chat, and customer communications. Every scenario requires the agent to coordinate state across at least four of these surfaces. The hidden state layer models exactly the kind of non-obvious cross-surface dependencies that trip up current enterprise agents.

## Authors

Shashwat Rajan and Tanish Shitanshu, for the Meta PyTorch OpenEnv Hackathon Grand Finale 2026.

## Acknowledgments

Architectural primitives (desk coordinator, uncertainty layer, cascade graph, multi-objective scoring) are ported from our Round 1 submission `airport-ops-recovery`, which qualified us for the finale.
