# OpsTwin Recovery, Checkpoint Tracker

Submission: Meta PyTorch OpenEnv Hackathon Grand Finale, Bangalore, April 25 to 26, 2026.

## Rubric: Minimum Requirements (gating)

- [x] **W1** Uses latest OpenEnv (`openenv-core==0.2.3`).
- [x] **W1** Minimal TRL training script runnable in Colab. See `notebooks/opstwin_training_colab.ipynb`.
- [ ] **W2** Environment hosted on HuggingFace Spaces. Artifacts ready; deploy commands in `DEPLOY_SPACE.md`. User must run.
- [ ] **W3** Mini-blog on HF OR YouTube video under 2 min. Not started.
- [ ] **W4** Fill `demo-assets/` with actual content (blog draft, video script, published URLs).

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
`git push origin main`  (fast-forward only, includes the v3-better-training commits).

### Deploy to HuggingFace Space (see `DEPLOY_SPACE.md`)
1. `huggingface-cli login`
2. `huggingface-cli repo create opstwin-recovery --type space --space_sdk docker --organization Deltasthic`
3. `git remote add hfspace https://huggingface.co/spaces/Deltasthic/opstwin-recovery`
4. `git push hfspace main:main`
5. `python tests/smoke_http.py --base-url https://Deltasthic-opstwin-recovery.hf.space`
6. Save the URL to `demo-assets/hf-space-link.txt`.

### Colab notebook smoke
Open `notebooks/opstwin_training_colab.ipynb` in Colab (Free tier T4), run all cells, confirm the loss curve + rollout print appear. Should take ~4 min.

## Known Issues (not rubric blockers)

- Default `env.reset()` without a task arg returns `done=True` after one REQUEST_INFO step. Probably a small fallback scenario. Investigate during Colab run; not urgent for minimums.
- `bad_release` scenario trained score still at 0.42 (V1). v3-better-training branch attempts to fix this through wrong-name augmentation and VERIFY_FLAG coverage; evaluation of v3 checkpoint not yet run.
- Postmortem memory loop (Phase 4) not confirmed in main. Ablation table for judges would be nice to have.

## Checkpoint 2 Targets (post-minimums)

- [ ] Run v3 training on the 5090, push new checkpoint to `Deltasthic/opstwin-qwen3-1.7b-sft` (or a v3 variant).
- [ ] Re-run `evaluate.py`, expect `bad_release` above 0.60 and average above 0.80.
- [ ] Regenerate `results/eval_curve.png` with the new bar chart.
- [ ] Add a proper training-loss curve PNG (separate from the eval bar chart) so the rubric's "reward improvement evidence" line has an unambiguous single image.

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
