# Training & Evaluation — OpsTwin V3

End-to-end reproduction of the published V3 checkpoint
[`Deltasthic/opstwin-qwen3-4b-sft-v3`](https://huggingface.co/Deltasthic/opstwin-qwen3-4b-sft-v3)
and its evaluation numbers. See [`results/FINAL_RESULTS.md`](results/FINAL_RESULTS.md)
for the full V1 vs V3 comparison table and methodology notes.

## Compute

- 1× NVIDIA GPU with **≥ 32 GB VRAM** (we used RTX 5090, full fine-tune at batch=1 + grad_accum=32)
- ~**3 h** wall-clock for the full 3-stage curriculum (all early-stopped, nowhere near the config ceilings)
- ~**100 GB** disk for intermediate checkpoints across the three stages. Can be pruned after each stage.

## Environment

```bash
# Python 3.12 + uv for dep management
python3 -m venv .venv && source .venv/bin/activate

# Torch with CUDA 12.8 wheels (required for Blackwell sm_120; earlier CUDA wheels work on older GPUs)
uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch

# Training stack (pins matter — see results/FINAL_RESULTS.md methodology note)
uv pip install \
  'trl==0.14.0' 'transformers==4.55.4' 'peft==0.13.2' \
  'datasets==3.2.0' 'accelerate>=1.3' \
  'liger-kernel<0.6' 'bitsandbytes>=0.49' \
  matplotlib protobuf python-dotenv

# OpsTwin environment deps
uv pip install -e .

# HuggingFace auth (for push + private models). Token needs write access.
echo "HF_TOKEN=hf_xxxxx" > .env
```

Key stack gotchas:
- **transformers 4.47.0** (the V1 Colab pin) does **not** recognize Qwen3 architecture. Use **4.51 or newer**.
- **accelerate 1.2.1** (the V1 Colab pin) is **incompatible with transformers 4.55** (missing `keep_torch_compile` kwarg in `Accelerator.unwrap_model`). Use **1.3 or newer**.
- **liger-kernel ≥ 0.6** requires transformers ≥ 4.52. Pin to `<0.6` if you stay on 4.47–4.51.

## Three-stage training

### Stage 1 — broad procedural coverage

```bash
source .venv/bin/activate && set -a && source .env && set +a
python train_sft_v3.py \
  --model Qwen/Qwen3-4B \
  --hf-model-id Deltasthic/opstwin-qwen3-4b-sft-v3 \
  --epochs 5 --batch-size 1 --lr 1e-5 --seeds 300 \
  --output-dir ./sft_checkpoints_v3
```

Dataset: 300 procedural seeds × 3 difficulties × 3 families × 2 augmentation passes + all 4 hand-authored trajectories = **5,408 trajectories / 62,013 training turns**. Augmentation: wrong-name `FLIP_FLAG` masking (trained to ignore them), `VERIFY_FLAG` / `CHECK_APPROVAL` / `INSPECT_RUNBOOK` prefix curriculum (trained normally).

Stop when rollout callback avg plateaus. The best stage-1 checkpoint by rollout avg was `checkpoint-200` (avg 0.64). Pass to stage 2.

### Stage 2 — balanced rebalance from stage-1 best checkpoint

```bash
python train_sft_v3_stage2.py --base ./sft_checkpoints_v3/checkpoint-200
```

Dataset: **780 balanced trajectories / 8,408 turns**. Heavy hand-authored weighting (150 bad_release passes, 50 each of security_cve / data_pipeline_regression / false_positive), plus 50-seed bad_release procedural and 30-seed maintenance on security_cve + data_pipeline. Goal: anchor bad_release at 0.99 and hold security_cve.

Best rollout at step 75 (avg 0.864 callback, avg 0.725 evaluator). Pass to stage 3.

### Stage 3 — data_pipeline focus with bad_release floor guard

```bash
python train_sft_v3_stage3.py --base ./sft_checkpoints_v3_stage2/checkpoint-75
```

Dataset: **900 trajectories / 10,713 turns**. Heavy data_pipeline weighting (150 hand-authored + 600 procedural), maintenance on bad_release (20 seeds procedural + 30 hand-authored), 30-pass hand-authored on the other two. `GuardedRolloutEvalCallback` auto-aborts training if the bad_release rollout drops below **0.80** — protects against the oscillation seen in stage 1.

Best: `checkpoint-45`. Evaluator numbers:
- bad_release: **0.99**
- security_cve: **0.99**
- data_pipeline_regression: **0.99**
- false_positive: **0.856**
- **Average: 0.956**

## Publishing

```bash
# Edit scripts/push_v3_step75.py -> change CKPT to sft_checkpoints_v3_stage3/checkpoint-45
python scripts/push_v3_step75.py
```

## Evaluation (against published HF model)

```bash
python evaluate.py \
  --model Deltasthic/opstwin-qwen3-4b-sft-v3 \
  --tasks bad_release security_cve data_pipeline_regression false_positive \
  --n-seeds 3
# Outputs:
#   results/eval_summary.json
#   results/eval_curve.png
```

## Plotting the training curve

```bash
python scripts/plot_training_curve.py
# Output: results/training_loss_curve.png
# Requires sft_checkpoints_v3*/ to be on disk.
```

## Notes on the evaluator

`evaluate.py` was fixed during the V3 investigation — see [`results/FINAL_RESULTS.md`](results/FINAL_RESULTS.md) methodology section. The two bugs it had were (1) a history-format mismatch with training, and (2) `max_new_tokens=64` truncating Qwen3's `<think>` blocks. Both are now fixed. V1's originally-published numbers were measured with the pre-fix evaluator; re-measuring V1 under current transformers / tokenizer produces below-random scores (a stack-drift issue, not a genuine V1 regression), so V1's baseline numbers are kept as reported.

## Artifacts

- `results/FINAL_RESULTS.md` — full V1 vs V3 comparison + methodology
- `results/eval_summary_v3.json` — V3 Stage-3 step-45 numbers
- `results/eval_curve_v3.png` — bar chart
- `results/training_loss_curve.png` — 3-stage loss curve
- `results_v3_stage3_step45/` — raw eval artifacts from the winning checkpoint
- `training_log.txt`, `training_v3_stage2.log`, `training_v3_stage3.log` — full training logs

## Files not in git (regeneratable)

- `sft_dataset_v3*.jsonl` — produced by the training scripts on each run
- `sft_checkpoints_v3*/` — ~174 GB of checkpoint data
- `v1_local/` — local snapshot of V1 model with patched tokenizer config
