# GPU Handoff: Tasks for a GPU-Enabled Claude Session

This file lets a fresh Claude Code session on a GPU-enabled server pick up
OpsTwin Recovery Arena training / evaluation work that the WSL session
(no GPU) could not run. Read this end-to-end before starting.

## Project context (one-minute read)

OpsTwin Recovery Arena is an OpenEnv environment for training LLM agents on
enterprise incident response. Submission: Meta PyTorch OpenEnv Hackathon Grand
Finale, Bangalore, April 25 to 26, 2026. Theme 3.1 Professional Tasks. Authors:
Shashwat Rajan, Tanish Shitanshu. Repo: https://github.com/Deltasthicc/opstwin-recovery.

Rubric weights: Environment Innovation 40, Storytelling 30, Reward Improvement 20,
Training Pipeline 10. Minimums: OpenEnv latest, Colab-runnable training script,
HF Space hosting, blog OR video under 2 min.

Current V1 evaluation numbers (published model `Deltasthic/opstwin-qwen3-1.7b-sft`):

| scenario | random | untrained | trained | heuristic |
|---|---|---|---|---|
| bad_release | 0.24 | 0.18 | **0.42** | 0.99 |
| security_cve | 0.22 | 0.20 | **0.94** | 0.99 |
| data_pipeline_regression | 0.24 | 0.20 | **0.75** | 0.99 |
| false_positive | 0.24 | 0.85 | **0.84** | 0.99 |
| **average** | 0.24 | 0.36 | **0.74** | 0.99 |

The V1 weakness is `bad_release` at 0.42. Root cause is flag-name hallucination.
`train_sft_v3.py` on main branch adds wrong-name augmentation and a dedicated
VERIFY_FLAG curriculum to fix this.

## Environment setup on the GPU server

```bash
# Clone (or pull if already present)
git clone git@github.com:Deltasthicc/opstwin-recovery.git
cd opstwin-recovery

# Python 3.10+ (3.12 preferred)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_v1_frozen.txt
# Training extras:
pip install -e ".[train,inference,eval]"

# Hugging Face auth (needed to push the trained model)
huggingface-cli login
# OR: export HF_TOKEN=hf_xxxx

# Sanity-check GPU is visible
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## What the WSL session left in place

- `server/app.py` uses a shared env singleton so HTTP multi-step demos work.
  Verified: `tests/smoke_http.py` passes all 5 checks.
- `notebooks/opstwin_training_colab.ipynb` has pinned versions
  (trl 0.14.0, transformers 4.47.0, peft 0.13.2, datasets 3.2.0, accelerate 1.2.1).
  Structurally valid, non-GPU cells verified. GPU cells never executed.
- `DEPLOY_SPACE.md` documents HF Space deployment. Artifacts ready
  (`Dockerfile`, `.dockerignore`, `openenv.yaml`, `README.md` frontmatter).
  Space has NOT been deployed yet.
- `demo-assets/blog-post.md` and `demo-assets/video-script.md` drafted.
  Neither is published.
- Last commit on local main: `5611130` "fix: persist env state across HTTP calls
  + pin Colab notebook deps". Not pushed. Author: Shashwat Rajan.

## Current state of every checkpoint

### Checkpoint 1: minimum requirements (gating)

- [x] OpenEnv latest (0.2.3)
- [x] TRL training script exists (`train.py`, `train_sft_5090.py`, `train_sft_v3.py`)
- [x] Colab notebook exists (`notebooks/opstwin_training_colab.ipynb`)
- [ ] **Colab notebook actually executed end-to-end on Colab Free T4.** OPEN.
- [ ] **HF Space deployed.** OPEN. See `DEPLOY_SPACE.md`.
- [ ] **Blog post published on HF.** OPEN. Draft at `demo-assets/blog-post.md`.
- [ ] **YouTube video recorded and uploaded.** OPEN. Script at `demo-assets/video-script.md`.

### Checkpoint 2: lift performance

- [ ] Run `train_sft_v3.py` on the GPU. Target: `bad_release` > 0.60, average > 0.80.
- [ ] Re-run `evaluate.py` against the new checkpoint with at least 5 held-out seeds.
- [ ] Regenerate `results/eval_curve.png` with the new comparison bar chart.
- [ ] Produce a separate training-loss curve PNG (step vs. loss, from the trainer's log_history) and save to `results/training_loss_curve.png`.
- [ ] Update `results/FINAL_RESULTS.md` with new numbers.
- [ ] Push the new checkpoint to Hugging Face (same repo `Deltasthic/opstwin-qwen3-1.7b-sft` or a new `-v3` variant).

### Checkpoint 3: stretch (only after Checkpoint 2 lands)

- [ ] Add a fifth scenario family. Options: multi-region outage, SOC-2 compliance audit, runbook-drift cascade. Touches the Environment Innovation score.
- [ ] Implement or verify Phase 4 postmortem memory loop. Ablation: base policy vs. postmortem-augmented policy on 10 held-out seeds, target a visible improvement delta. File: `server/postmortem.py`.
- [ ] Record a live terminal demo (not just slides) for the YouTube video. Voiceover script already in `demo-assets/video-script.md`.
- [ ] Ensure `v3-better-training` branch is fully merged to main (already fast-forwarded locally as of commit `ce1cbe0`).

---

# GPU-Required Task List (execute in this order)

## Task 1: validate the Colab notebook on the actual Colab runtime

Not GPU-server work per se, but it's GPU-required and a Checkpoint 1 minimum.

**What to do:**
1. Open `notebooks/opstwin_training_colab.ipynb` in Colab.
2. Runtime, Change runtime type, T4 GPU.
3. Run all cells top to bottom.
4. If any cell fails, patch the notebook and try again. Likely failure modes:
   - Cell 5 (model load): if `dtype=` rejected, change to `torch_dtype=`. If Qwen3-0.6B is gated, swap to `Qwen/Qwen2.5-0.5B-Instruct` as a fallback.
   - Cell 6 (training): if `processing_class=` rejected, change to `tokenizer=`.
   - Cell 8 (rollout): if OOM, reduce `max_new_tokens` to 16.
5. Save the notebook with outputs checked in: `File, Download .ipynb`, replace the file in the repo.

**Success:** notebook runs end-to-end, training loss drops, rollout prints a non-empty trace.

**Commit message:** `notebook: verified on Colab Free T4, <summary of any patches>`

## Task 2: run the V3 training

**File:** `train_sft_v3.py` (481 lines, lives on main since `5b3bd42`).

Key differences from `train_sft_5090.py`:
- Base model: `Qwen/Qwen3-4B` (upgraded from 1.7B)
- 200 seeds per family (up from 50)
- Wrong-name augmentation (targeted at `bad_release` weakness)
- `VERIFY_FLAG` coverage in the expert solver
- Sanity check + rollout callback + seed holdout

**What to do:**
```bash
# From repo root, with venv active
python train_sft_v3.py \
    --model Qwen/Qwen3-4B \
    --epochs 2 \
    --lora-rank 0 \
    --batch-size 4 \
    --lr 2e-4 \
    --hf-model-id Deltasthic/opstwin-qwen3-4b-sft-v3 \
    --output-dir ./sft_checkpoints_v3
```

Before running, READ `train_sft_v3.py` for any bugs. The script was added in 3 commits (`5b3bd42`, `4f062ae`, `ce1cbe0`) and has not been executed yet.

**Success criteria:**
- Training completes without OOM.
- Final eval loss comparable to or better than V1 (V1 was 0.02).
- Checkpoint pushed to `Deltasthic/opstwin-qwen3-4b-sft-v3` on HuggingFace.
- Training-loss curve saved to `results/training_loss_curve.png` (see Task 3).

**Expected runtime:** ~30 to 90 minutes on an RTX 5090 (32 GB) or A100 (40 GB).
On smaller GPUs, keep the `--lora-rank 32` default (add it explicitly) instead of full fine-tune.

## Task 3: generate the training-loss curve PNG

The rubric's 20 percent "Reward Improvement" criterion is clearest when judges
see BOTH a training curve (loss going down) AND an eval bar chart (score going up).

**What to do.** After Task 2 finishes, the trainer's `state.log_history` contains all the loss data. Add this to `train_sft_v3.py` or write a small helper `scripts/plot_training_curve.py`:

```python
import json
import matplotlib.pyplot as plt
from transformers.trainer_callback import TrainerState

state = TrainerState.load_from_json("./sft_checkpoints_v3/checkpoint-<last>/trainer_state.json")
history = state.log_history
steps = [h["step"] for h in history if "loss" in h]
loss  = [h["loss"]  for h in history if "loss" in h]

plt.figure(figsize=(10, 5))
plt.plot(steps, loss, linewidth=2)
plt.xlabel("training step")
plt.ylabel("loss")
plt.title("OpsTwin V3 SFT (Qwen3-4B, full fine-tune, ~3,600 trajectories)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/training_loss_curve.png", dpi=150)
```

**Success:** `results/training_loss_curve.png` exists, shows a downward trend.

## Task 4: re-run evaluate.py against V3 checkpoint

**File:** `evaluate.py` already exists.

**What to do:**
```bash
python evaluate.py \
    --model Deltasthic/opstwin-qwen3-4b-sft-v3 \
    --n-seeds 5 \
    --output-dir results/ \
    --scenarios bad_release,security_cve,data_pipeline_regression,false_positive
```

Read `evaluate.py` first and match the actual CLI arg names (the above is a best guess).

**Success criteria:**
- `results/eval_summary.json` updated with V3 numbers.
- `results/eval_curve.png` regenerated.
- `bad_release` score > 0.60 (target). If still below 0.60, diagnose (maybe add more augmentation and re-train).
- Average score across 4 scenarios > 0.80 (target).

## Task 5: update FINAL_RESULTS.md

Rewrite `results/FINAL_RESULTS.md` with:
- New baseline vs. V1 vs. V3 comparison table.
- A narrative paragraph on what V3 fixed (wrong-name aug, VERIFY_FLAG).
- Links to the new HF model and the training-loss curve PNG.

Keep the style similar to the existing V1 FINAL_RESULTS.md. No em dashes (user preference, global CLAUDE.md rule).

## Task 6 (optional, Checkpoint 3): postmortem ablation

Only if Checkpoints 1 and 2 are fully landed.

**Goal:** prove the postmortem memory loop does something.

**What to do:**
1. Confirm `server/postmortem.py` exists and writes to `baselines/trajectories/postmortems.jsonl`.
2. Confirm `reset()` injects memory hints when `memory_hints` kwarg is passed.
3. Run 10 held-out seeds with `memory_hints=[]` (base) and 10 with top-2 postmortems from the `bad_release` family. Average the final scores.
4. If the augmented version scores higher, write a new section in `results/FINAL_RESULTS.md` titled "Self-Improvement Ablation" with the before / after table.

**Success:** visible score delta (even 0.05 is enough to tell a story).

## Task 7 (optional, Checkpoint 3): fifth scenario family

Only if Checkpoints 1 and 2 are landed AND you have bandwidth.

Authoring a new family takes about an hour. Use `server/scenarios.py` and
`docs/scenarios.md` as templates. Suggested families:
- **Multi-region outage.** One region fully down, another degrading. Agent must
  fail over without violating data-residency policy.
- **SOC-2 compliance audit.** A compliance alert fires; agent must pull audit
  logs, identify the missing control, and draft an external disclosure.
- **Runbook drift.** The runbook tells the agent to do X; the actual correct
  action is Y because infra changed. Tests whether the agent trusts its
  environment observations over stale docs.

---

# Non-GPU follow-ups for the WSL session

The user runs these on the WSL machine after the GPU server finishes. Do not
do these from the GPU server.

- `git pull --ff-only origin main` once GPU commits land.
- Deploy to HF Space (`DEPLOY_SPACE.md`).
- Record the YouTube video using `demo-assets/video-script.md`.
- Publish the blog at huggingface.co/blog, save URL to `demo-assets/hf-blog-link.txt`.
- Revoke the old PAT at github.com/settings/tokens.

# Coordination notes between WSL and GPU Claude sessions

- **Git branching:** all work goes on `main`. There is no feature-branch workflow for this project. Push frequently.
- **Force push is allowed on `main`** (we already did one to fix authorship) but try not to. Force push breaks the other session's pulls.
- **Merge conflicts:** if both sessions edit the same file, the later pusher rebases onto main before pushing: `git fetch origin && git rebase origin/main`.
- **Commit author:** set `git config user.email "shashwat@deltasthic.dev"` and `git config user.name "Shashwat Rajan"` in the repo on the GPU server so commits are attributed consistently.
- **Large files:** do NOT commit training checkpoints or generated datasets. They are in `.gitignore`. The only artifacts that go in git are `results/*.json`, `results/*.png`, `results/FINAL_RESULTS.md`, `sft_dataset.jsonl` (already committed, regenerate only if schema changes).

# Quick reference

| What | Where |
|---|---|
| Repo | https://github.com/Deltasthicc/opstwin-recovery |
| Current published model | https://huggingface.co/Deltasthic/opstwin-qwen3-1.7b-sft |
| V1 training script | `train_sft_5090.py` |
| V3 training script | `train_sft_v3.py` |
| Eval script | `evaluate.py` |
| Colab notebook | `notebooks/opstwin_training_colab.ipynb` |
| Blog draft | `demo-assets/blog-post.md` |
| Video script | `demo-assets/video-script.md` |
| Space deploy steps | `DEPLOY_SPACE.md` |
| Full TODO | `TODO.md` |
| Architecture docs | `docs/architecture.md`, `docs/scenarios.md`, `docs/reward-model.md`, `docs/build-order.md` |
