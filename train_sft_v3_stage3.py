"""
train_sft_v3_stage3.py — targeted data_pipeline recovery from stage-2 step-75.

Context at entry:
  - Stage-2 step-75 rollout avg=0.864 (bad_release=0.99, security_cve=0.99,
    false_positive=0.86, data_pipeline_regression=0.62).
  - Only remaining sub-V1 scenario is data_pipeline (V1=0.75).
  - Goal: lift DP to >=0.75 without breaking bad_release's 0.99 anchor.

Strategy: small LR, DP-heavy dataset, tiny eval budget, hard abort on BR regression.
If any rollout shows bad_release < 0.80, the callback raises and training stops.

Dataset (~1000 trajectories, ~10k turns):
  - 150 aug-passes of hand-authored data_pipeline_regression
  - 100 seeds × 3 diff × 2 aug of data_pipeline procedural (~600)
  - 30 aug-passes of hand-authored bad_release, security_cve, false_positive each
  - 20 seeds × 3 diff × 1 aug of bad_release procedural (maintain anchor)
"""
import argparse, json, sys, random, os
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from train_sft_v3 import trajectory_to_examples, RolloutEvalCallback
from models import OpsAction
from server.environment import OpsTwinEnvironment
from server.generator import GENERATED_FAMILIES
from baselines.expert_solver import OPTIMAL_TRAJECTORIES
from inference import SYSTEM_PROMPT, build_user_prompt


BAD_RELEASE_FLOOR = 0.80  # abort if bad_release rollout falls below this


class GuardedRolloutEvalCallback(RolloutEvalCallback):
    """Rollout callback that STOPS training if bad_release regresses below a floor."""
    def __init__(self, tokenizer, floor=BAD_RELEASE_FLOOR, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.floor = floor

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        super().on_evaluate(args, state, control, model=model, **kwargs)
        # Post-super: we have no direct handle on the scores the parent computed
        # because it prints them but doesn't store per-scenario values. Re-run
        # only the bad_release rollout to check the floor.
        from train_sft_v3 import _quick_rollout
        model.eval()
        try:
            with torch.no_grad():
                br = _quick_rollout(model, self.tokenizer, "bad_release")
            if br < self.floor:
                print(f"[GUARD] bad_release={br:.2f} < floor={self.floor}. STOPPING.",
                      flush=True)
                control.should_training_stop = True
        except Exception as e:
            print(f"[GUARD] failed to check bad_release: {e}", flush=True)
        finally:
            model.train()


def build_dataset():
    train_examples, eval_examples = [], []
    kept = dropped = 0

    # Hand-authored, heavy DP weighting
    hand_weights = {
        "data_pipeline_regression": 150,
        "bad_release": 30,
        "security_cve": 30,
        "false_positive": 30,
    }
    for task, passes in hand_weights.items():
        if task not in OPTIMAL_TRAJECTORIES:
            continue
        for i in range(passes):
            env = OpsTwinEnvironment()
            obs = env.reset(task=task)
            cmds = OPTIMAL_TRAJECTORIES[task]
            history, prev_reward = [], 0.0
            for step, cmd in enumerate(cmds, 1):
                user_prompt = build_user_prompt(step, obs, prev_reward, history)
                train_examples.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": cmd},
                    ]
                })
                obs = env.step(OpsAction(command=cmd))
                history.append(f"S{step}: {cmd} -> {obs.reward:+.2f}")
                history = history[-10:]
                prev_reward = obs.reward
                if obs.done:
                    break
            kept += 1

    # Procedural: DP heavy, BR maintenance
    proc_budget = {
        "data_pipeline": (100, ("easy", "medium", "hard"), [False, True]),
        "bad_release":   (20,  ("easy", "medium", "hard"), [True]),
    }
    for family, (n_seeds, difficulties, augs) in proc_budget.items():
        if family not in GENERATED_FAMILIES:
            print(f"  WARN: {family} not in GENERATED_FAMILIES, skipping")
            continue
        eval_cutoff = max(1, int(n_seeds * 0.9))
        for seed in range(n_seeds):
            target = eval_examples if seed >= eval_cutoff else train_examples
            for difficulty in difficulties:
                for augment in augs:
                    try:
                        exs, score = trajectory_to_examples({
                            "task": "generated", "family": family,
                            "seed": seed, "difficulty": difficulty,
                        }, augment=augment)
                        if score > 0.65:
                            target.extend(exs)
                            kept += 1
                        else:
                            dropped += 1
                    except Exception:
                        dropped += 1

    print(f"  Kept {kept} trajectories, dropped {dropped}")
    print(f"  Train: {len(train_examples)} turns  Eval: {len(eval_examples)} turns")
    return train_examples, eval_examples


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="./sft_checkpoints_v3_stage2/checkpoint-75")
    parser.add_argument("--hf-model-id", default="Deltasthic/opstwin-qwen3-4b-sft-v3")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--output-dir", default="./sft_checkpoints_v3_stage3")
    parser.add_argument("--no-push", action="store_true", default=True,
                        help="Default: DO NOT auto-push (user pushes manually after review)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA.")
        sys.exit(1)

    random.seed(44)
    torch.manual_seed(44)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Resuming from: {args.base}")
    print(f"Bad-release floor: {BAD_RELEASE_FLOOR} (training aborts if rollout < this)")
    print()

    print("Building DP-focused dataset...")
    train_examples, eval_examples = build_dataset()
    print()

    ds_path = Path("sft_dataset_v3_stage3.jsonl")
    with open(ds_path, "w") as f:
        for ex in train_examples + eval_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Dataset saved to {ds_path}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback

    print(f"\nLoading base {args.base}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    from datasets import Dataset
    random.shuffle(train_examples)
    train_ds = Dataset.from_list(train_examples)
    eval_ds = Dataset.from_list(eval_examples)
    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}\n")

    from trl import SFTTrainer, SFTConfig

    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 32 // args.batch_size),
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        weight_decay=0.01,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=15,
        save_strategy="steps",
        save_steps=15,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        max_seq_length=1024,
        packing=False,
        use_liger_kernel=True,
        report_to="none",
        seed=44,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            GuardedRolloutEvalCallback(tokenizer, every_n_evals=1),
        ],
    )

    print("=" * 60)
    print("STAGE 3 TRAINING (DP focus, BR floor guard)")
    print("=" * 60)
    trainer.train()
    print("\nStage 3 training complete.")

    rollout_cb = next((cb for cb in trainer.callback_handler.callbacks
                       if isinstance(cb, GuardedRolloutEvalCallback)), None)
    if rollout_cb and rollout_cb.best_step >= 0:
        print(f"Best rollout avg: {rollout_cb.best_avg:.3f} at step {rollout_cb.best_step}")

    out_path = f"{args.output_dir}/final"
    trainer.model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    print(f"Final checkpoint saved to {out_path}")
    print("NOT auto-pushing. Review checkpoints, then push manually if stage 3 improved.")


if __name__ == "__main__":
    main()
