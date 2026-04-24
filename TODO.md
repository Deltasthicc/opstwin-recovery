# OpsTwin Recovery, Checkpoint Tracker

Submission: Meta PyTorch OpenEnv Hackathon Grand Finale, Bangalore, April 25 to 26, 2026.

## Rubric: Minimum Requirements (gating)

- [x] **W1** Uses latest OpenEnv (`openenv-core==0.2.3`).
- [x] **W1** Minimal TRL training script runnable in Colab. See `notebooks/opstwin_training_colab.ipynb`.
- [ ] **W2** Environment hosted on HuggingFace Spaces. Artifacts ready; deploy commands in `DEPLOY_SPACE.md`. User must run.
- [~] **W3** Mini-blog + YouTube video. Blog drafted at `demo-assets/blog-post.md` (635 words). Video script at `demo-assets/video-script.md` (~1:37 runtime). Neither is published yet.
- [~] **W4** `demo-assets/` has blog draft and script. Still missing: hf-blog-link.txt, youtube-link.txt, hf-space-link.txt (all populated after publishing).

## Current Scoring Estimate

V1 pipeline only (no W2/W3/W4): ~65 to 80 out of 100.
After Fix 3 deploy and Fix 4 blog or video lands: ~75 to 85.

## Deploy Queue (commands the user still has to run)

### Git remote hygiene (security)
Current `origin` URL has a GitHub PAT embedded in `.git/config`.
1. Revoke at https://github.com/settings/tokens.
2. `git remote set-url origin https://github.com/Deltasthicc/opstwin-recovery.git`.
3. Use `gh auth login` or a credential helper.

### Push main to GitHub
BLOCKED until GitHub auth is fixed (2026-04-23). The embedded PAT in `.git/config` is invalid.
Fix with ONE of:
- `gh auth login` then `git remote set-url origin https://github.com/Deltasthicc/opstwin-recovery.git` then `git push origin main`
- Or new fine-grained PAT with credential.helper store
- Or SSH key: `git remote set-url origin git@github.com:Deltasthicc/opstwin-recovery.git`

After push works, revoke the old token at https://github.com/settings/tokens.

### Deploy to HuggingFace Space (see `DEPLOY_SPACE.md`)
1. `huggingface-cli login`
2. `huggingface-cli repo create opstwin-recovery --type space --space_sdk docker --organization Deltasthic`
3. `git remote add hfspace https://huggingface.co/spaces/Deltasthic/opstwin-recovery`
4. `git push hfspace main:main`
5. `python tests/smoke_http.py --base-url https://Deltasthic-opstwin-recovery.hf.space`
6. Save the URL to `demo-assets/hf-space-link.txt`.

### Colab notebook smoke
Open `notebooks/opstwin_training_colab.ipynb` in Colab (Free tier T4), run all cells, confirm the loss curve + rollout print appear. Should take ~4 min.

## Known Issues

- ~~Default `env.reset()` without a task arg returns `done=True` after one REQUEST_INFO step.~~ FIXED: `server/app.py` now uses a single shared env instance so HTTP state persists across `/reset` and `/step`. Smoke test asserts `total_issues > 0` on reset and `done is False` after one REQUEST_INFO.
- `bad_release` scenario trained score still at 0.42 (V1). `train_sft_v3.py` on main attempts to fix this through wrong-name augmentation and VERIFY_FLAG coverage; evaluation of v3 checkpoint not yet run.
- Postmortem memory loop (Phase 4) not confirmed in main. Ablation table for judges would be nice to have.

## Checkpoint 2 Targets (post-minimums) — DONE 2026-04-24

- [x] Run v3 training on the 5090, push new checkpoint. Published as
      `Deltasthic/opstwin-qwen3-4b-sft-v3` (Qwen3-4B full fine-tune, not 1.7B
      as originally planned — handoff pre-dated the decision to upgrade).
- [x] Re-run `evaluate.py` with **5 held-out seeds** against the published HF
      checkpoint. `bad_release=0.99` (target was >0.60), average=0.956
      (target was >0.80). Numbers in `results/eval_summary.json` (canonical).
      See `results/FINAL_RESULTS.md` for full table.
- [x] Regenerate `results/eval_curve.png` with the V3 bar chart (canonical name).
      V1 baseline preserved at `results/eval_curve_v1.png` for traceability.
- [x] Training-loss curve at `results/training_loss_curve.png`, covering all
      3 stages with transition markers.

Deviations from the handoff's plan:
- Used a 3-stage curriculum (stage 1 broad, stage 2 balanced-rebalance, stage 3
  DP focus + BR floor guard) instead of a single training run, to fix scenario
  oscillation observed during stage 1.
- Two `evaluate.py` bugs were fixed during the V3 investigation
  (history-format mismatch, `max_new_tokens` too small). See methodology note
  in `results/FINAL_RESULTS.md`.
- Published checkpoint went to a new `...-v3` HF repo rather than overwriting
  V1's `...-1.7b-sft`, keeping V1 available for historical comparison.

## Checkpoint 3 Targets (stretch)

- [ ] Add a 5th scenario family (multi-region outage OR compliance audit). Touches innovation score.
- [ ] Postmortem ablation table: base policy vs. postmortem-augmented policy on 10 held-out seeds. Implements and proves Phase 4.
- [ ] Record a live demo-terminal video (not just slides). More impactful than a plain voiceover.
- [ ] Merge v3-better-training back to main after evaluation (already done locally, still needs `git push origin main`).

## Done

- [x] Local HTTP smoke test (`tests/smoke_http.py`).
- [x] HF Space metadata in `README.md` frontmatter.
- [x] `.dockerignore` so the Space image stays lean.
- [x] `DEPLOY_SPACE.md` with step-by-step commands.
- [x] `demo-assets/` folder with tracking README.
- [x] Colab training notebook.
- [x] `v3-better-training` fast-forwarded into local main (`train_sft_v3.py` now present).
- [x] `.gitignore` cleaned up (`[External]*.md`, `.venv-*/`, notebook allowlist exception).
