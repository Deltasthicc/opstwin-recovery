"""
train_sft_v3_stage2.py — targeted re-balance stage on top of V3 checkpoint-200.

Goal: fix the oscillation seen in stage 1:
  - V1 baseline avg=0.74 (bad_release=0.42)
  - V3 stage 1 step 100: avg=0.512 (bad_release=0.61 ✓, but FP=0.30, DP=0.23)
  - V3 stage 1 step 200: avg=0.644 (FP=0.85 ✓, DP=0.61 ✓, but bad_release=0.21 ✗)

Strategy: start from checkpoint-200 (preserves the FP/DP wins) and retrain on a
heavily-balanced mini-dataset so bad_release recovers without losing the others.
Smaller dataset + lower LR + tighter eval schedule = targeted shift, not another
long optimization that drifts.

Dataset design (~880 trajectories, ~10k turns):
  - 150 aug-passes of hand-authored bad_release
  - 50 aug-passes of hand-authored false_positive
  - 50 aug-passes each of hand-authored security_cve, data_pipeline_regression
  - 50 procedural seeds × 3 diff × 2 aug of bad_release  (300 traj)
  - 30 seeds × 3 diff × 1 aug of security_cve, data_pipeline  (90 each)

LR 5e-6 (half of stage 1), 2 epochs, eval every 25, rollout every 2 evals.
"""
import argparse, json, sys, random, os
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# Import stage-1 planner + helpers
from train_sft_v3 import (
    trajectory_to_examples, plan_for_scenario,
    RolloutEvalCallback, COMMAND_TO_DESK,
)
from models import OpsAction
from server.environment import OpsTwinEnvironment
from server.generator import GENERATED_FAMILIES
from baselines.expert_solver import OPTIMAL_TRAJECTORIES
from inference import SYSTEM_PROMPT, build_user_prompt


def build_balanced_dataset():
    """Returns (train_examples, eval_examples) lists, 4-way-balanced with
    heavy weighting on hand-authored (especially bad_release and false_positive)."""
    train_examples, eval_examples = [], []
    kept = dropped = 0
    eval_reserve = []  # last ~10% of generated seeds per family go to eval

    # ---- Hand-authored, heavy aug passes -------------------------------
    hand_weights = {
        "bad_release": 150,
        "false_positive": 50,
        "security_cve": 50,
        "data_pipeline_regression": 50,
    }
    for task, passes in hand_weights.items():
        if task not in OPTIMAL_TRAJECTORIES:
            continue
        for i in range(passes):
            env = OpsTwinEnvironment()
            obs = env.reset(task=task)
            # Reuse the expert-solver trajectory exactly for hand-authored.
            # Augmentation variance in hand-authored runs comes from the PLANNER
            # in procedural mode — for hand-authored we want the golden sequence.
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

    # ---- Procedural, targeted -----------------------------------------
    # bad_release: 50 seeds × 3 diff × 2 aug
    # security_cve, data_pipeline: 30 seeds × 3 diff × 1 aug (maintenance)
    proc_budget = {
        "bad_release": (50, ("easy", "medium", "hard"), [False, True]),
        "security_cve": (30, ("easy", "medium", "hard"), [True]),
        "data_pipeline": (30, ("easy", "medium", "hard"), [True]),
    }
    for family, (n_seeds, difficulties, augs) in proc_budget.items():
        if family not in GENERATED_FAMILIES:
            print(f"  WARN: {family} not in GENERATED_FAMILIES, skipping")
            continue
        eval_cutoff = max(1, int(n_seeds * 0.9))  # last 10% to eval
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
                    except Exception as e:
                        dropped += 1

    print(f"  Kept {kept} trajectories, dropped {dropped}")
    print(f"  Train: {len(train_examples)} turns  Eval: {len(eval_examples)} turns")
    return train_examples, eval_examples


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="./sft_checkpoints_v3/checkpoint-200",
                        help="Starting checkpoint (stage-1 best by rollout avg)")
    parser.add_argument("--hf-model-id", default="Deltasthic/opstwin-qwen3-4b-sft-v3")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--output-dir", default="./sft_checkpoints_v3_stage2")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA.")
        sys.exit(1)

    random.seed(43)
    torch.manual_seed(43)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Resuming from: {args.base}")
    print()

    print("Building balanced dataset...")
    train_examples, eval_examples = build_balanced_dataset()
    print()

    ds_path = Path("sft_dataset_v3_stage2.jsonl")
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

    total = sum(p.numel() for p in model.parameters())
    print(f"Full fine-tune. Trainable: {total/1e9:.2f}B params\n")

    from datasets import Dataset
    random.shuffle(train_examples)
    train_ds = Dataset.from_list(train_examples)
    eval_ds = Dataset.from_list(eval_examples)
    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}")

    from trl import SFTTrainer, SFTConfig

    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 32 // args.batch_size),
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.01,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=4,
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
        seed=43,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            RolloutEvalCallback(tokenizer, every_n_evals=1),  # rollout EVERY eval
        ],
    )

    print("\n" + "=" * 60)
    print("STAGE 2 TRAINING")
    print("=" * 60)
    trainer.train()
    print("\nStage 2 training complete.")

    rollout_cb = next((cb for cb in trainer.callback_handler.callbacks
                       if isinstance(cb, RolloutEvalCallback)), None)
    if rollout_cb and rollout_cb.best_step >= 0:
        print(f"Best rollout avg: {rollout_cb.best_avg:.3f} at step {rollout_cb.best_step}")

    out_path = f"{args.output_dir}/final"
    trainer.model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    print(f"Final checkpoint saved to {out_path}")

    if not args.no_push:
        print(f"Pushing to {args.hf_model_id}...")
        trainer.model.push_to_hub(args.hf_model_id, private=False)
        tokenizer.push_to_hub(args.hf_model_id, private=False)
        print(f"Done: https://huggingface.co/{args.hf_model_id}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
