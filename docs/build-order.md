# Build Order

Follow this phase sequence strictly. Do not advance to the next phase if the current one is
incomplete. The training evidence in Phase 3 is non-negotiable — it is 20% of the judging score.

---

## Phase 1 — Environment Skeleton

Goal: `reset()/step()` loop works end-to-end. No real scenarios yet.

1. Copy `airport-ops-recovery/server/app.py` → `server/app.py` (minimal changes)
2. Rewrite `models.py`: define `OpsAction`, `OpsObservation`, `OpsState` using airport models as
   template. Replace flight/gate/crew/pax fields with service/ticket/pipeline/alert fields.
3. Port `server/environment.py`:
   - Rename class to `OpsTwinEnvironment`
   - Replace `_flights/_gates/_crew/_passengers` → `_services/_tickets/_pipelines/_alerts`
   - Keep `reset()/step()/_obs()/_exec()/_load()` signatures identical
   - Wire all new action handlers as stubs returning `(0, "stub")`
   - Keep `_audit_trail`, `_clock_minutes`, `MINUTES_PER_STEP = 5`, `_done` pattern
4. Port `server/graph.py` from `network.py` (see architecture.md for spec)
5. Port `server/hidden_state.py` from `visibility.py` (see architecture.md for spec)
6. Port `server/desks.py` from `roles.py` (see architecture.md for spec)
7. Implement `server/scoring.py` — 6 pure functions + `compute_multi_objective_scores()`
   (see reward-model.md for exact weights and logic)
8. Smoke test: `reset()` returns an `OpsObservation`, `step(OpsAction(command="REQUEST_INFO summary"))`
   returns without error, `DONE` terminates the episode.

**Exit criteria:** Server starts, one episode completes, score is non-zero.

---

## Phase 2 — Scenarios and Reward Signal

Goal: 3 hand-authored scenarios that produce meaningful reward variation.

9. Write 3 scenario dicts in `server/scenarios.py`, one per family (see scenarios.md).
   Each must have: `task_name`, `description`, `disruption_type`, `max_steps`, `max_score`,
   `services`, `tickets`, `pipelines`, `alerts`, `issues`, `dynamic_events`.
10. For each scenario, manually trace an optimal action sequence and verify it scores > 0.80.
11. Trace a suboptimal sequence (wrong actions) and verify it scores < 0.40.
12. Write `baselines/expert_solver.py`: rule-based solver that reads the scenario and emits the
    optimal action sequence. Run it against all 3 scenarios. Store traces as `.jsonl` in
    `baselines/trajectories/`.

**Exit criteria:** 3 scenarios, expert solver works, reward signal is clearly discriminative.

---

## Phase 3 — Training (HIGHEST PRIORITY)

Goal: at least one GRPO reward curve showing improvement. This is what judges will ask about.

13. Write `train.py`:
    ```python
    from trl import GRPOTrainer, GRPOConfig
    from openenv import OpenEnvClient

    config = GRPOConfig(
        model_name="Qwen/Qwen3-4B-Instruct",   # or Gemma-3-4B-IT
        reward_funcs=["environment"],
        max_steps=500,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_steps=50,
        logging_steps=10,
    )
    ```
    Use `AsyncGRPO` if separate rollout/training GPUs are available.
    Use GRPO with replay buffer if early batches show low reward variance.

14. Run training job on available hardware. Target: 100+ steps minimum, 500 preferred.
    Save reward curve to `notebooks/training_analysis.ipynb`.

15. Write `evaluate.py`: evaluate trained checkpoint vs. untrained baseline on 5+ held-out seeds.
    Output: mean score ± std for each. Save as `results/eval_summary.json`.

16. Produce the evidence package (minimum for demo):
    - Plot: episode reward vs. training step (baseline flat line + trained policy rising)
    - Table: baseline vs. trained mean score across held-out seeds
    - One qualitative example: show a trajectory where trained policy avoids a mistake
      that baseline makes

**Exit criteria:** One reward curve exists. Trained policy measurably outperforms baseline.

---

## Phase 4 — Self-Improvement Story

Goal: postmortem memory loop that produces a visible improvement delta.

17. Implement `server/postmortem.py`:
    ```python
    # Schema for each postmortem entry
    {
        "scenario_family": str,
        "failure_category": str,   # "missed_hidden_state" | "policy_violation" | "cascade_ignored"
        "first_bad_action": str,
        "missed_signal": str,
        "violated_policy": str | None,
        "preferred_intervention_order": list[str],
        "final_score": float,
        "episode_id": str,
        "timestamp": str,
    }
    ```
    Store as append-only `.jsonl` in `baselines/trajectories/postmortems.jsonl`.

18. Hook postmortem generation into `environment.py`: when `_done = True`, call
    `PostmortemMemory.record(episode_summary)`.

19. In `reset()`, retrieve top-2 most similar past postmortems (by scenario family match, then
    lowest score). Inject them into the initial observation message as:
    ```
    [MEMORY] Similar past incident: {failure_category}. First bad action: {first_bad_action}.
    Missed signal: {missed_signal}. Preferred order: {preferred_intervention_order}.
    ```

20. Run eval: postmortem-augmented policy vs. base policy on 10 held-out seeds.
    Show improvement delta. This is the ablation judges will ask for.

**Exit criteria:** Memory retrieval works. Augmented policy shows measurable improvement.

---

## Phase 5 — Polish (only after Phase 3 done)

21. Procedural generator in `server/generator.py` (3 families, seeded, difficulty param)
22. Stretch scenario: false positive trap (see scenarios.md)
23. `docs/architecture.md` as judge-facing design doc and HuggingFace blog source
24. README: setup, training, eval instructions
25. `openenv.yaml`: update registration from airport version

**Exit criteria:** Demo-ready. Blog post drafted. README complete.

---

## Pre-Onsite Checklist (must complete before April 25)

- [ ] Phases 1 and 2 complete
- [ ] `train.py` written and tested locally (even if short run)
- [ ] Expert trajectories generated and stored
- [ ] At least 1 reward curve exists (can be short run on laptop)
- [ ] `evaluate.py` works
- [ ] Phase 3 is ready to run with compute credits on Day 1

Phase 3 full training run happens on-site with HuggingFace compute credits.
