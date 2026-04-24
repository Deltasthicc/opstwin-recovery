"""Plot V3 training loss curve from checkpoint trainer_states (stage 1 + stage 2)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_loss_history(ckpt_dir: Path):
    """Return list of (step, train_loss, eval_loss) from trainer_state.json."""
    state_path = ckpt_dir / "trainer_state.json"
    if not state_path.exists():
        return []
    with open(state_path) as f:
        state = json.load(f)
    history = state.get("log_history", [])
    out = []
    for h in history:
        step = h.get("step")
        if step is None:
            continue
        out.append({
            "step": step,
            "loss": h.get("loss"),
            "eval_loss": h.get("eval_loss"),
        })
    return out


def find_latest(root: Path):
    """Highest checkpoint-N folder has the fullest trainer_state.json."""
    ckpts = sorted(root.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[1]))
    return ckpts[-1] if ckpts else None


stage1 = find_latest(Path("sft_checkpoints_v3"))
stage2 = find_latest(Path("sft_checkpoints_v3_stage2"))
stage3 = find_latest(Path("sft_checkpoints_v3_stage3"))
print(f"Stage 1 latest: {stage1}")
print(f"Stage 2 latest: {stage2}")
print(f"Stage 3 latest: {stage3}")

h1 = load_loss_history(stage1) if stage1 else []
h2 = load_loss_history(stage2) if stage2 else []
h3 = load_loss_history(stage3) if stage3 else []

# Each stage's trainer_state numbers steps from 0. Offset stages 2/3 for a continuous x-axis.
offset_s2 = max((h["step"] for h in h1), default=0)
s1_steps = [h["step"] for h in h1 if h.get("loss") is not None]
s1_loss = [h["loss"] for h in h1 if h.get("loss") is not None]
s1_eval_steps = [h["step"] for h in h1 if h.get("eval_loss") is not None]
s1_eval = [h["eval_loss"] for h in h1 if h.get("eval_loss") is not None]

s2_steps = [h["step"] + offset_s2 for h in h2 if h.get("loss") is not None]
s2_loss = [h["loss"] for h in h2 if h.get("loss") is not None]
s2_eval_steps = [h["step"] + offset_s2 for h in h2 if h.get("eval_loss") is not None]
s2_eval = [h["eval_loss"] for h in h2 if h.get("eval_loss") is not None]

offset_s3 = offset_s2 + max((h["step"] for h in h2), default=0)
s3_steps = [h["step"] + offset_s3 for h in h3 if h.get("loss") is not None]
s3_loss = [h["loss"] for h in h3 if h.get("loss") is not None]
s3_eval_steps = [h["step"] + offset_s3 for h in h3 if h.get("eval_loss") is not None]
s3_eval = [h["eval_loss"] for h in h3 if h.get("eval_loss") is not None]

fig, ax = plt.subplots(figsize=(12, 5.5))

ax.plot(s1_steps, s1_loss, color="#2b6cb0", linewidth=1.5, alpha=0.85,
        label="Stage 1 train loss (Qwen3-4B full FT, 62k turns)")
ax.plot(s1_eval_steps, s1_eval, color="#2b6cb0", linewidth=2.0,
        linestyle="--", marker="o", markersize=5, label="Stage 1 eval loss")

if s2_steps:
    ax.plot(s2_steps, s2_loss, color="#c05621", linewidth=1.5, alpha=0.85,
            label="Stage 2 train loss (balanced retrain from stage-1 cp-200)")
    ax.plot(s2_eval_steps, s2_eval, color="#c05621", linewidth=2.0,
            linestyle="--", marker="s", markersize=5, label="Stage 2 eval loss")
    ax.axvline(offset_s2, color="gray", linestyle=":", linewidth=1,
               label=f"Stage 1 → 2 transition (step {offset_s2})")

if s3_steps:
    ax.plot(s3_steps, s3_loss, color="#276749", linewidth=1.5, alpha=0.85,
            label="Stage 3 train loss (DP focus from stage-2 cp-75 + BR guard)")
    ax.plot(s3_eval_steps, s3_eval, color="#276749", linewidth=2.0,
            linestyle="--", marker="^", markersize=6, label="Stage 3 eval loss")
    ax.axvline(offset_s3, color="gray", linestyle=":", linewidth=1,
               label=f"Stage 2 → 3 transition (step {offset_s3})")

ax.set_xlabel("Global training step")
ax.set_ylabel("Loss")
ax.set_title("OpsTwin V3 SFT — Qwen3-4B full fine-tune (RTX 5090, 32GB)")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=9)
ax.set_yscale("log")

fig.tight_layout()
out = Path("results/training_loss_curve.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(f"Saved {out}")
print(f"Stage 1: {len(s1_steps)} loss points, {len(s1_eval)} eval points")
print(f"Stage 2: {len(s2_steps)} loss points, {len(s2_eval)} eval points")
print(f"Stage 3: {len(s3_steps)} loss points, {len(s3_eval)} eval points")
