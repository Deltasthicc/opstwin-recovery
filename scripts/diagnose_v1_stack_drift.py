"""V1 debug: test with both history formats to see which one V1 needs."""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models import OpsAction
from server.environment import OpsTwinEnvironment
from inference import SYSTEM_PROMPT, build_user_prompt

CKPT = "./v1_local"
print(f"Loading {CKPT}...")
tokenizer = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForCausalLM.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model.eval()


def run_scenario(task, history_format="with_reward", max_step=12):
    print(f"\n#### task={task}  history={history_format} ####")
    env = OpsTwinEnvironment()
    obs = env.reset(task=task)
    history, prev_reward = [], 0.0
    for step in range(1, max_step + 1):
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
        obs = env.step(OpsAction(command=cmd))
        print(f"  S{step}: {cmd!r:55s} r={obs.reward:+.3f} score={obs.score:.3f}")
        if history_format == "with_reward":
            history.append(f"S{step}: {cmd} -> {obs.reward:+.2f}")
        else:
            history.append(f"S{step}: {cmd}")
        prev_reward = obs.reward
        if obs.done:
            break
    print(f"  FINAL: {obs.score:.3f}")
    return obs.score


s1 = run_scenario("bad_release", "with_reward")
s2 = run_scenario("bad_release", "without_reward")
print(f"\nbad_release: with_reward={s1:.3f}  without_reward={s2:.3f}")
