"""
train_sft_5090.py -- OpsTwin SFT training optimized for RTX 5090 (32GB VRAM).
"""
import argparse
import json
import os
import sys
import random
from pathlib import Path

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


def plan_for_scenario(env):
    issues = env._issues
    plan = ["SWITCH_DESK INCIDENT_COMMAND", "ASSESS_BLAST_RADIUS"]
    current_desk = "INCIDENT_COMMAND"

    def emit(cmd):
        nonlocal current_desk
        verb = cmd.split()[0]
        required = COMMAND_TO_DESK.get(verb)
        if required and required != current_desk:
            plan.append(f"SWITCH_DESK {required}")
            current_desk = required
        plan.append(cmd)

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

    done = set()
    for t in issues.get("ticket_escalations", []):
        for r in t.get("valid_resolutions", []):
            if "PRIORITIZE_VIP" in r:
                emit(r)
                done.add(t["ticket_id"])
                break
    for t in issues.get("ticket_escalations", []):
        if t["ticket_id"] in done:
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

    plan.append("DONE")
    return plan


def trajectory_to_examples(env_kwargs):
    env = OpsTwinEnvironment()
    obs = env.reset(**env_kwargs)
    plan = plan_for_scenario(env)
    examples = []
    history = []
    prev_reward = 0.0
    for step, cmd in enumerate(plan, 1):
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
        history = history[-5:]
        prev_reward = obs.reward
        if obs.done:
            break
    return examples, obs.score


def build_dataset():
    all_examples = []
    kept = 0
    dropped = 0

    for task, cmds in OPTIMAL_TRAJECTORIES.items():
        env = OpsTwinEnvironment()
        obs = env.reset(task=task)
        history, prev_reward = [], 0.0
        for step, cmd in enumerate(cmds, 1):
            user_prompt = build_user_prompt(step, obs, prev_reward, history)
            all_examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": cmd},
                ]
            })
            obs = env.step(OpsAction(command=cmd))
            history.append(f"S{step}: {cmd} -> {obs.reward:+.2f}")
            history = history[-5:]
            prev_reward = obs.reward
            if obs.done:
                break
        kept += 1

    for family in GENERATED_FAMILIES:
        for seed in range(50):
            for difficulty in ["easy", "medium"]:
                try:
                    exs, score = trajectory_to_examples({
                        "task": "generated",
                        "family": family,
                        "seed": seed,
                        "difficulty": difficulty,
                    })
                    if score > 0.7:
                        all_examples.extend(exs)
                        kept += 1
                    else:
                        dropped += 1
                except Exception:
                    dropped += 1

    print(f"  Kept {kept} trajectories, dropped {dropped}")
    print(f"  Total training examples: {len(all_examples)}")
    return all_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--hf-model-id", default="Deltasthic/opstwin-qwen3-1.7b-sft")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="0 for full fine-tune (5090 has the VRAM for it)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--output-dir", default="./sft_checkpoints")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA GPU visible.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print()

    print("Generating training data from expert trajectories...")
    all_examples = build_dataset()
    print()

    if len(all_examples) < 1000:
        print(f"WARNING: only {len(all_examples)} examples. Expected ~3000.")

    ds_path = Path("sft_dataset.jsonl")
    with open(ds_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Dataset saved to {ds_path}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
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
        print("Full fine-tune mode (no LoRA).")
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable params: {total/1e9:.2f}B")

    from datasets import Dataset
    random.seed(42)
    random.shuffle(all_examples)
    split = int(len(all_examples) * 0.95)
    train_ds = Dataset.from_list(all_examples[:split])
    eval_ds = Dataset.from_list(all_examples[split:])
    print(f"\nTrain: {len(train_ds)}  Eval: {len(eval_ds)}")

    from trl import SFTTrainer, SFTConfig

    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 16 // args.batch_size),
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=30,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=(args.lora_rank == 0),
        optim="adamw_torch",
        max_length=2048,
        packing=False,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    trainer.train()
    print("\nTraining complete.")

    adapter_path = f"{args.output_dir}/adapter"
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved to {adapter_path}")

    if args.lora_rank > 0:
        print("Merging LoRA into base weights...")
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
