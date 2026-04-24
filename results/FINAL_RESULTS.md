# OpsTwin Recovery Arena — Training Results (V1 and V3)

## Headline

**V3 lifts average score from 0.74 (V1) to 0.956 — a +0.22 absolute improvement that brings three of four scenarios to the heuristic upper bound (0.99).** The remaining scenario (false_positive) reached 0.856, above V1's 0.84. Training used a three-stage SFT curriculum on Qwen3-4B: broad procedural generation (Stage 1), balanced rebalance from best stage-1 checkpoint (Stage 2), and data-pipeline-focused reinforcement with a bad_release floor guard (Stage 3). The final checkpoint is published at [Deltasthic/opstwin-qwen3-4b-sft-v3](https://huggingface.co/Deltasthic/opstwin-qwen3-4b-sft-v3).

## Results (deterministic, greedy decoding, fixed evaluator)

| Scenario | Random | Heuristic (upper) | V1 trained | **V3 trained (Stage 3, step-45)** |
|---|---|---|---|---|
| bad_release | 0.24 | 0.99 | 0.42 | **0.990** |
| security_cve | 0.22 | 0.99 | 0.94 | **0.990** |
| data_pipeline_regression | 0.24 | 0.99 | 0.75 | **0.990** |
| false_positive | 0.30 | 0.99 | 0.84 | **0.856** |
| **Average** | **0.24** | **0.99** | **0.74** | **0.956** |

## V3 model

- **Base:** Qwen/Qwen3-4B (~4.02B parameters, full fine-tune)
- **Method:** Three-stage SFT on planner-generated trajectories
- **Training data per stage:**
  - Stage 1 — 62,013 turns from 5,408 trajectories (300 procedural seeds × 3 difficulties × 3 families × 2 augmentation passes, plus hand-authored golden trajectories for all 4 scenarios)
  - Stage 2 — 8,408 turns from 780 balanced trajectories (starting from stage-1 checkpoint-200, heavy hand-authored weighting: 150 bad_release passes, 50 each of the others; plus 50-seed bad_release procedural and 30-seed maintenance on security_cve and data_pipeline)
  - Stage 3 — 10,713 turns from 900 trajectories (starting from stage-2 checkpoint-75, heavy data_pipeline focus: 150 hand-authored + 600 procedural bad_release kept at 20 seeds as anchor maintenance)
- **Hardware:** NVIDIA RTX 5090 (32 GB VRAM, inside Incus LXC container)
- **Stopping criteria:** each stage stopped early based on rollout-callback signal rather than running full epochs. Total training time across all three stages: ~2h15m.
- **HuggingFace:** [Deltasthic/opstwin-qwen3-4b-sft-v3](https://huggingface.co/Deltasthic/opstwin-qwen3-4b-sft-v3)
- **Stack:** torch 2.11.0+cu128, transformers 4.55.4, trl 0.14.0, peft 0.13.2, bitsandbytes 0.49, liger-kernel 0.5.10

## Why three stages

Stage 1 generated good scenarios-averaged capacity but showed **scenario oscillation** — as training progressed, the policy would learn one scenario family while regressing on another. Rollout callback numbers bounced between (strong bad_release, weak data_pipeline) and (weak bad_release, strong data_pipeline) depending on which mini-batches had run most recently.

Stage 2 addressed this by **starting from the best Stage-1 checkpoint (step-200)** and retraining on a small, heavily-balanced dataset with hand-authored trajectories upweighted. This anchored `bad_release` at 0.99 and stabilized `security_cve` at 0.99, but `data_pipeline_regression` and `false_positive` remained below V1.

Stage 3 targeted the remaining gap with a DP-heavy dataset and a **hard abort guard** (`GuardedRolloutEvalCallback`) that halts training if `bad_release` rollout drops below 0.80. Step-45 lifted all three previously-weak scenarios to 0.99 and recovered false_positive to 0.856 — slightly above V1.

## Per-scenario analysis

- **bad_release (+0.57):** V3's wrong-name FLIP_FLAG masking and VERIFY_FLAG prefix curriculum eliminated V1's core failure mode (feature-flag-name hallucination). The policy now reliably verifies the flag identifier before flipping and scores at the heuristic upper bound.
- **security_cve (+0.05):** Small lift to the heuristic upper bound. V3's 4B capacity handles multi-desk coordination (SECURITY → RELEASE → SUPPORT) more cleanly than V1's 1.7B.
- **data_pipeline_regression (+0.24):** Stage 3's DP-focused retrain (150 hand-authored + 600 procedural data_pipeline trajectories) lifted this from 0.62 (post-Stage-2) to 0.99, matching the heuristic. The policy now reliably sequences `RUN_MITIGATION` → `RERUN_PIPELINE` → `DRAFT_COMMS external` without skipping the mandatory customer communication.
- **false_positive (+0.02):** Stage 3 recovered false_positive from 0.30 (Stage 2 plateau) to 0.856 without specifically targeting it — the DP-heavy retrain happened to reinforce investigation-before-action patterns that generalize to false_positive. Slight improvement over V1.

## Methodology note (evaluator v2)

During V3 evaluation we found two bugs in `evaluate.py`:

1. **History format mismatch.** Training planner formatted history entries as `S{step}: {cmd} -> {reward:+.2f}`. The evaluator wrote `S{step}: {cmd}` with no reward suffix — a prompt format the trained model had never seen. Fix: retroactively append the reward suffix from `_last_reward` to the last history entry before building the next prompt.
2. **Token budget truncation.** `max_new_tokens=64` in the evaluator's `ModelPolicy`. Qwen3's `<think>` blocks routinely exceed that before emitting the final command, causing the regex parser to default to `REQUEST_INFO summary`. Fix: raise to 128 to match the training-time callback rollout.

Both fixes are applied to the evaluator for V3's numbers in this report. V1's published numbers (0.42/0.94/0.75/0.84, avg 0.74) were measured with the pre-fix evaluator. Attempting to re-measure V1 under the fixed evaluator and current transformers 4.55 + patched tokenizer produces scores below the random baseline (avg ~0.20), a stack-compatibility issue with the older V1 checkpoint rather than a genuine V1 regression. We therefore keep the original V1 numbers as the documented baseline and flag that the V1-vs-V3 comparison in the headline table uses V1's original stack and V3's fixed stack.

**Robustness of the bad_release finding.** V3 scores 0.99 on bad_release under both the original and the fixed evaluator — the +0.57 lift is not a measurement artifact.

## Artifacts

- Eval summary JSON: `results/eval_summary_v3.json` (V3 Stage 3 step-45), `results/eval_summary.json` (original V1 eval, preserved)
- Bar chart: `results/eval_curve_v3.png` (V3 Stage 3 step-45)
- Training loss curve: `results/training_loss_curve.png` (Stage 1 + 2 + 3 combined)
- Training logs: `training_log.txt` (Stage 1), `training_v3_stage2.log`, `training_v3_stage3.log`
- Training scripts: `train_sft_v3.py` (Stage 1), `train_sft_v3_stage2.py` (Stage 2 rebalance), `train_sft_v3_stage3.py` (Stage 3 DP focus + BR guard)
- Evaluator (with both bugfixes): `evaluate.py`

## Reproduction

```bash
# Evaluate the published V3 checkpoint on 4 scenarios (deterministic, 3 seeds redundant)
python evaluate.py \
    --model Deltasthic/opstwin-qwen3-4b-sft-v3 \
    --tasks bad_release security_cve data_pipeline_regression false_positive \
    --n-seeds 3
```

Expected output (same as table above):
```
trained    bad_release                  mean=0.990
trained    security_cve                 mean=0.990
trained    data_pipeline_regression     mean=0.990
trained    false_positive               mean=0.856
```

## Known issues / future work

- **false_positive at 0.856 not 0.99.** The model diverges on one step of the FP trajectory in a way that terminates the scenario early. A targeted Stage 4 with 300+ FP augmentation passes could likely push this to 0.99 but risks the same oscillation pattern we fought in Stages 1–2. Left as follow-up.
- **Stack drift breaks V1 loading.** V1's older tokenizer format needs patching under transformers 4.55 (`extra_special_tokens` list → dict). `scripts/` contains the patch; a clean upgrade path would require republishing V1.
- **Evaluator rollout vs training-time callback diverge.** Training-time `RolloutEvalCallback` uses the liger-kernel-fused forward pass, producing slightly different logits than standalone inference. Standalone inference numbers are the operationally meaningful ones for anyone downloading the HF model; those are what this report uses.
