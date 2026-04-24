"""Direct comparison: _quick_rollout vs ModelPolicy.act on false_positive.

Prints the command emitted at each step by each code path, on an IDENTICAL
scenario run, so we can find the divergence."""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models import OpsAction
from server.environment import OpsTwinEnvironment
from inference import SYSTEM_PROMPT, build_user_prompt

CKPT = "./sft_checkpoints_v3_stage2/checkpoint-75"
TASK = "false_positive"

print(f"Loading {CKPT}...")
tokenizer = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForCausalLM.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def _extract_cmd(raw):
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text).strip()
    text = text.strip('`"\'')
    text = re.sub(r"^S\d+:\s*", "", text)
    text = re.sub(r"\s*->\s*[+-]?\d*\.?\d+\s*$", "", text)
    text = text.split(" -> ")[0].split("\n")[0].strip()
    return text or "REQUEST_INFO summary"


def run_callback_style():
    """Exactly what _quick_rollout does in train_sft_v3.py"""
    print("\n=== CALLBACK STYLE (_quick_rollout) ===")
    env = OpsTwinEnvironment()
    obs = env.reset(task=TASK)
    history, prev_reward = [], 0.0
    for step in range(1, 21):
        user_prompt = build_user_prompt(step, obs, prev_reward, history)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True,
            add_generation_prompt=True,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=128, do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        cmd = _extract_cmd(raw)
        obs = env.step(OpsAction(command=cmd))
        print(f"  S{step}: {cmd!r:50s} -> r={obs.reward:+.3f}  score={obs.score:.3f}")
        history.append(f"S{step}: {cmd} -> {obs.reward:+.2f}")
        history = history[-10:]
        prev_reward = obs.reward
        if obs.done:
            break
    print(f"  FINAL SCORE: {obs.score:.3f}")
    return obs.score


def run_modelpolicy_style():
    """Exactly what evaluate.py's ModelPolicy + rollout does (with fixes applied)"""
    print("\n=== MODEL POLICY STYLE (evaluate.py) ===")
    env = OpsTwinEnvironment()
    obs = env.reset(task=TASK)
    history = []
    step = 0
    last_reward = 0.0
    for iteration in range(1, 41):  # rollout max_steps_cap=40
        # act():
        if history and " -> " not in history[-1]:
            history[-1] = f"{history[-1]} -> {last_reward:+.2f}"
        step += 1
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(step, obs, last_reward, history)},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
        cleaned = cleaned.strip('`"\'')
        cleaned = re.sub(r"^S\d+:\s*", "", cleaned)
        cleaned = re.sub(r"\s*->\s*[+-]?\d*\.?\d+\s*$", "", cleaned)
        cleaned = cleaned.split(" -> ")[0].split("\n")[0].strip()
        cmd = cleaned or "REQUEST_INFO summary"
        history.append(f"S{step}: {cmd}")
        # --- rollout loop ---
        obs = env.step(OpsAction(command=cmd))
        print(f"  S{step}: {cmd!r:50s} -> r={obs.reward:+.3f}  score={obs.score:.3f}")
        last_reward = obs.reward or 0.0
        if obs.done:
            break
    print(f"  FINAL SCORE: {obs.score:.3f}")
    return obs.score


cb_score = run_callback_style()
mp_score = run_modelpolicy_style()
print(f"\n=== SUMMARY ===")
print(f"Callback style:     {cb_score:.3f}")
print(f"ModelPolicy style:  {mp_score:.3f}")
print(f"Delta:              {abs(cb_score - mp_score):.3f}")
