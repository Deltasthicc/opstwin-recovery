"""
train_sft_v3.py — Enhanced SFT training over v1:
  - 300 seeds x 3 difficulties (easy/medium/hard) x 3 families, seeds >=270 held out as eval
  - Augmentation: wrong-name FLIP_FLAG (masked), VERIFY_FLAG/CHECK_APPROVAL/INSPECT_RUNBOOK prefixes (trained)
  - Qwen3-4B full fine-tune default (fits on 5090 at batch=2)
  - Rollout-based eval callback + eval_loss-based early stopping (patience=4)
  - load_best_model_at_end=True selects best eval_loss checkpoint
"""
import argparse, json, sys, random, re
from pathlib import Path
from datetime import datetime
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from models import OpsAction
from server.environment import OpsTwinEnvironment
from server.generator import GENERATED_FAMILIES
from baselines.expert_solver import OPTIMAL_TRAJECTORIES
from inference import SYSTEM_PROMPT, build_user_prompt

COMMAND_TO_DESK = {
    "FLIP_FLAG": "RELEASE", "RERUN_PIPELINE": "RELEASE",
    "ROLLBACK_DEPLOYMENT": "RELEASE", "CANCEL_PIPELINE": "RELEASE",
    "VERIFY_FLAG": "RELEASE",
    "APPROVE_EXCEPTION": "SECURITY", "QUARANTINE_SERVICE": "SECURITY",
    "BLOCK_ROLLOUT": "SECURITY", "CHECK_APPROVAL": "SECURITY",
    "RESTART_SERVICE": "SRE", "RUN_MITIGATION": "SRE",
    "ISOLATE_SERVICE": "SRE", "INSPECT_RUNBOOK": "SRE",
    "PRIORITIZE_VIP": "SUPPORT", "TRIAGE_TICKET": "SUPPORT",
    "DRAFT_COMMS": "SUPPORT",
    "ASSESS_BLAST_RADIUS": "INCIDENT_COMMAND",
}


def plan_for_scenario(env, augment=False):
    """Returns list of (command, should_train) tuples. should_train=False means the
    step runs in env (for history continuity) but no training example is emitted."""
    issues = env._issues
    plan = [("SWITCH_DESK INCIDENT_COMMAND", True), ("ASSESS_BLAST_RADIUS", True)]
    current_desk = "INCIDENT_COMMAND"

    def switch_to(desk):
        nonlocal current_desk
        if current_desk != desk:
            plan.append((f"SWITCH_DESK {desk}", True))
            current_desk = desk

    def emit(cmd):
        nonlocal current_desk
        verb = cmd.split()[0]
        parts = cmd.split()

        # Aug 1: wrong FLIP_FLAG name first (30%), NOT trained on
        if augment and verb == "FLIP_FLAG" and len(parts) >= 3 and random.random() < 0.3:
            correct = parts[1]
            wrong = re.sub(r"_v2_ui$", "_beta",
                    re.sub(r"^flag_", "", correct))
            if wrong != correct:
                switch_to("RELEASE")
                plan.append((f"FLIP_FLAG {wrong} off", False))  # masked

        # Aug 2: VERIFY_FLAG before FLIP_FLAG (50%), trained on
        if augment and verb == "FLIP_FLAG" and len(parts) >= 2 and random.random() < 0.5:
            switch_to("RELEASE")
            plan.append((f"VERIFY_FLAG {parts[1]}", True))

        # Aug 3: CHECK_APPROVAL before APPROVE_EXCEPTION (50%), trained on
        if augment and verb == "APPROVE_EXCEPTION" and len(parts) >= 2 and random.random() < 0.5:
            switch_to("SECURITY")
            plan.append((f"CHECK_APPROVAL {parts[1]}", True))

        # Aug 4: INSPECT_RUNBOOK before RUN_MITIGATION (40%), trained on
        if augment and verb == "RUN_MITIGATION" and len(parts) >= 2 and random.random() < 0.4:
            switch_to("SRE")
            plan.append((f"INSPECT_RUNBOOK {parts[1]}", True))

        required = COMMAND_TO_DESK.get(verb)
        if required:
            switch_to(required)
        plan.append((cmd, True))

    for block in issues.get("approval_blocks", []):
        if "blocking_action" in block:
            emit(block["blocking_action"])
        elif "change_id" in block:
            emit(f"APPROVE_EXCEPTION {block['change_id']}")

    for rb in issues.get("mandatory_rollbacks", []):
        if "deployment_id" in rb:
            emit(f"ROLLBACK_DEPLOYMENT {rb['deployment_id']}")

    for outage in issues.get("service_outages", []):
        svc_id = outage.get("service_id", "")
        if svc_id:
            emit(f"INSPECT_RUNBOOK {svc_id}")
        action = outage.get("required_action") or \
                 (outage.get("valid_actions") or [None])[0]
        if action:
            emit(action)

    done_tids = set()
    for t in issues.get("ticket_escalations", []):
        for r in t.get("valid_resolutions", []):
            if "PRIORITIZE_VIP" in r:
                emit(r)
                done_tids.add(t["ticket_id"])
                break
    for t in issues.get("ticket_escalations", []):
        if t["ticket_id"] in done_tids:
            continue
        resolutions = t.get("valid_resolutions", [])
        if resolutions:
            emit(resolutions[0])

    for alert in issues.get("alerts_to_clear", []):
        for req in alert.get("requires_inspection", []):
            emit(f"INSPECT_RUNBOOK {req.replace('runbook:', '')}")

    for comm in issues.get("pending_comms", []):
        audience = comm.get("audience", "internal")
        emit(f"DRAFT_COMMS {audience} Incident resolved. Systems nominal.")

    plan.append(("DONE", True))
    return plan


def trajectory_to_examples(env_kwargs, augment=False):
    env = OpsTwinEnvironment()
    obs = env.reset(**env_kwargs)
    plan = plan_for_scenario(env, augment=augment)
    examples = []
    history = []
    prev_reward = 0.0
    for step, (cmd, should_train) in enumerate(plan, 1):
        if should_train:
            user_prompt = build_user_prompt(step, obs, prev_reward, history)
            examples.append({
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
    return examples, obs.score


def build_dataset(seeds_per_family=300, difficulties=("easy", "medium", "hard"),
                  eval_frac=0.10):
    """Seeds < cutoff go to train, seeds >= cutoff go to eval. No cross-split leakage.
    cutoff is derived as int(seeds_per_family * (1 - eval_frac)), with a minimum of
    seeds_per_family - 2 so even small dry runs get at least a couple eval seeds."""
    eval_seed_cutoff = max(1, min(seeds_per_family - 2,
                                  int(seeds_per_family * (1 - eval_frac))))
    print(f"  eval_seed_cutoff={eval_seed_cutoff} "
          f"(seeds {eval_seed_cutoff}..{seeds_per_family-1} held out for eval)")
    train_examples, eval_examples = [], []
    kept = dropped = 0

    # Hand-authored entirely to train
    for augment in [False, True]:
        for task, cmds in OPTIMAL_TRAJECTORIES.items():
            env = OpsTwinEnvironment()
            obs = env.reset(task=task)
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

    for family in GENERATED_FAMILIES:
        for seed in range(seeds_per_family):
            target = eval_examples if seed >= eval_seed_cutoff else train_examples
            for difficulty in difficulties:
                for augment in [False, True]:
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
    print(f"  Train: {len(train_examples)}  Eval: {len(eval_examples)}")
    return train_examples, eval_examples


# -----------------------------------------------------------------------------
# Rollout callback for real-score monitoring during training
# -----------------------------------------------------------------------------

from transformers import TrainerCallback


def _extract_cmd(raw):
    """Robust command extractor, mirrors rollout.py."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text).strip()
    text = text.strip('`"\'')
    text = re.sub(r"^S\d+:\s*", "", text)
    text = re.sub(r"\s*->\s*[+-]?\d*\.?\d+\s*$", "", text)
    text = text.split(" -> ")[0].split("\n")[0].strip()
    return text or "REQUEST_INFO summary"


def _quick_rollout(model, tokenizer, task, max_steps=20):
    """Minimal deterministic rollout for callback use."""
    env = OpsTwinEnvironment()
    obs = env.reset(task=task)
    history, prev_reward = [], 0.0
    for step in range(1, max_steps + 1):
        user_prompt = build_user_prompt(step, obs, prev_reward, history)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True,
            add_generation_prompt=True,
        ).to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=128, do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        cmd = _extract_cmd(raw)
        obs = env.step(OpsAction(command=cmd))
        history.append(f"S{step}: {cmd} -> {obs.reward:+.2f}")
        history = history[-10:]
        prev_reward = obs.reward
        if obs.done:
            break
    return obs.score


class RolloutEvalCallback(TrainerCallback):
    """Runs actual rollouts every N eval steps to get real scores, not loss proxy."""
    def __init__(self, tokenizer, scenarios=None, every_n_evals=2):
        self.tokenizer = tokenizer
        self.scenarios = scenarios or [
            "bad_release", "security_cve",
            "data_pipeline_regression", "false_positive",
        ]
        self.every_n_evals = every_n_evals
        self.eval_count = 0
        self.best_avg = -1.0
        self.best_step = -1

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        self.eval_count += 1
        if self.eval_count % self.every_n_evals != 0:
            return
        model.eval()
        scores = {}
        try:
            with torch.no_grad():
                for task in self.scenarios:
                    scores[task] = _quick_rollout(model, self.tokenizer, task)
            avg = sum(scores.values()) / len(scores)
            marker = ""
            if avg > self.best_avg:
                self.best_avg = avg
                self.best_step = state.global_step
                marker = "  *** NEW BEST ***"
            s_str = "  ".join(f"{k}={v:.2f}" for k, v in scores.items())
            print(f"[rollout step={state.global_step}] avg={avg:.3f}  {s_str}{marker}",
                  flush=True)
        except Exception as e:
            print(f"[rollout step={state.global_step}] FAILED: {e}", flush=True)
        finally:
            model.train()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--hf-model-id", default="Deltasthic/opstwin-qwen3-4b-sft-v3")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=0,
                        help="0 for full fine-tune (5090 has VRAM for 4B full FT at batch=2)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seeds", type=int, default=300,
                        help="seeds_per_family for the procedural generator")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--output-dir", default="./sft_checkpoints_v3")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA GPU visible.")
        sys.exit(1)

    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print()

    # Sanity: verify planner produces valid trajectories on one sample per family/difficulty
    print("Sanity check: clean trajectory per family x difficulty...")
    for family in GENERATED_FAMILIES:
        for difficulty in ("easy", "medium", "hard"):
            exs, score = trajectory_to_examples(
                {"task": "generated", "family": family, "seed": 999, "difficulty": difficulty},
                augment=False,
            )
            print(f"  {family}/{difficulty}: score={score:.2f} steps={len(exs)}")
            if score < 0.7:
                print(f"    PLANNER BUG on {family}/{difficulty}, aborting.")
                sys.exit(1)
    print("Sanity check passed.\n")

    print(f"Generating training data ({args.seeds} seeds per family, 3 difficulties)...")
    train_examples, eval_examples = build_dataset(seeds_per_family=args.seeds)
    print()

    if len(train_examples) < 3000:
        print(f"WARNING: only {len(train_examples)} train examples. Expected ~15000+.")

    ds_path = Path("sft_dataset_v3.jsonl")
    with open(ds_path, "w") as f:
        for ex in train_examples + eval_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Dataset saved to {ds_path}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
    from peft import LoraConfig, get_peft_model

    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if args.lora_rank > 0:
        lora = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"Full fine-tune mode (no LoRA). Trainable params: {total/1e9:.2f}B")

    from datasets import Dataset
    random.shuffle(train_examples)
    train_ds = Dataset.from_list(train_examples)
    eval_ds = Dataset.from_list(eval_examples)
    print(f"\nTrain: {len(train_ds)}  Eval: {len(eval_ds)}")

    from trl import SFTTrainer, SFTConfig

    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 32 // args.batch_size),
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=(args.lora_rank == 0),
        optim="adamw_8bit",
        max_seq_length=1024,
        packing=False,
        use_liger_kernel=True,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=4),
            RolloutEvalCallback(tokenizer, every_n_evals=2),
        ],
    )

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    trainer.train()
    print("\nTraining complete.")

    rollout_cb = next((cb for cb in trainer.callback_handler.callbacks
                       if isinstance(cb, RolloutEvalCallback)), None)
    if rollout_cb and rollout_cb.best_step >= 0:
        print(f"Best rollout avg: {rollout_cb.best_avg:.3f} at step {rollout_cb.best_step}")
        print(f"(load_best_model_at_end used eval_loss. If you want the rollout-best ckpt, "
              f"reload from {args.output_dir}/checkpoint-{rollout_cb.best_step})")

    adapter_path = f"{args.output_dir}/adapter"
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    if args.lora_rank > 0:
        merged = trainer.model.merge_and_unload()
        merged_path = f"{args.output_dir}/merged"
        merged.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
    else:
        merged = trainer.model
        merged_path = adapter_path

    if not args.no_push:
        print(f"Pushing to {args.hf_model_id}...")
        merged.push_to_hub(args.hf_model_id, private=False)
        tokenizer.push_to_hub(args.hf_model_id, private=False)
        print(f"Done: https://huggingface.co/{args.hf_model_id}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
