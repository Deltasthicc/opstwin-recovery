# OpsTwin Recovery Arena

<!-- Maintainer note: keep this file under 150 lines. Detailed specs live in docs/. -->

## What This Is

Enterprise incident-response OpenEnv environment for the Meta PyTorch OpenEnv Hackathon Grand
Finale (April 25–26, 2026, Bangalore). Domain transplant of `airport-ops-recovery` — keep the
architecture, change the nouns. Reference the airport codebase when porting logic.

**Team:** Shashwat Rajan + Tanish Shitanshu

## The One Thing That Matters Most

**There is zero training evidence right now. A GRPO reward curve must exist before demo time.**
Do not add features before Phase 3 (training) is complete. See @docs/build-order.md.

## Judging Weights

| Criterion | Weight |
|-----------|--------|
| Environment Innovation | 40% |
| Storytelling | 30% |
| Reward Improvement Evidence | 20% |
| Training Pipeline | 10% |

## Repository Layout

```
opstwin-recovery/
├── CLAUDE.md                  ← this file (always loaded)
├── models.py                  ← OpsAction, OpsObservation, OpsState (Pydantic)
├── client.py                  ← OpenEnv client
├── inference.py               ← baseline inference runner
├── train.py                   ← TRL/GRPO script — CRITICAL DELIVERABLE
├── evaluate.py                ← held-out seed evaluator
├── server/
│   ├── app.py                 ← FastAPI (copy from airport)
│   ├── environment.py         ← reset/step/_obs loop — MAIN FILE
│   ├── scenarios.py           ← hand-authored scenario dicts
│   ├── generator.py           ← procedural generator (seed-deterministic)
│   ├── desks.py               ← DeskCoordinator (replaces roles.py)
│   ├── hidden_state.py        ← HiddenStateLayer (replaces visibility.py)
│   ├── graph.py               ← ServiceDependencyGraph (replaces network.py)
│   ├── scoring.py             ← 6-dim scorer
│   └── postmortem.py          ← PostmortemMemory
├── baselines/
│   ├── expert_solver.py       ← rule-based gold trajectory generator
│   └── trajectories/          ← .jsonl expert traces
├── docs/
│   ├── build-order.md         ← PHASE-BY-PHASE BUILD PLAN
│   ├── architecture.md        ← component specs (desks, hidden state, graph)
│   ├── scenarios.md           ← scenario family specs
│   └── reward-model.md        ← 6-dim scoring spec
└── notebooks/
    └── training_analysis.ipynb
```

## Code Conventions

- Python 3.12, `uv` for deps
- Pydantic for all action/observation/state types — no raw dicts at API boundary
- `copy.deepcopy(sc)` on every scenario load — never mutate templates
- Reward values: positive for correct resolutions, `-0.02` for invalid commands,
  `-0.05` for actively harmful actions
- `_audit_trail`: append `{step, action, reward, resolved_count}` every step
- All scoring functions must be **pure** — no side effects, no LLM calls
- `SUPPORTS_CONCURRENT_SESSIONS = True` on environment class
- Keep `MINUTES_PER_STEP = 5` and clock simulation

## What NOT to Build

- No browser UI / Playwright / Selenium
- No live SaaS API calls (no real PagerDuty, no real GitHub)
- No LLM-judged rewards — everything deterministic
- No RAG over real runbooks — synthetic is fine
- Don't expand scenarios before training works

## Bonus Prize Alignment

**Scaler AI Labs** (hosts): "Multi-App RL Environment for Enterprise Workflows" → direct hit.
Mention multi-app explicitly: tickets + CI/CD + feature flags + observability + comms.

## When You Need Details

- Build phases and task order → @docs/build-order.md
- Desks, hidden state, graph, actions → @docs/architecture.md
- Scenario family specs → @docs/scenarios.md
- Scoring dimensions and weights → @docs/reward-model.md
