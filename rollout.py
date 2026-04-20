"""Quick rollout of the trained model against all 4 scenarios."""
import sys, re, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from models import OpsAction
from server.environment import OpsTwinEnvironment
from inference import SYSTEM_PROMPT, build_user_prompt, _extract_command

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "./sft_checkpoints/adapter"

def robust_extract(text):
    text = _extract_command(text)
    text = re.sub(r"^S\d+:\s*", "", text)
    text = re.sub(r"\s*->\s*[+-]?\d*\.?\d+\s*$", "", text)
    text = text.split(" -> ")[0].strip().strip('`"\'')
    return text

print(f"Loading {MODEL_PATH}...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
mdl.eval()
print("Model loaded. Beginning rollouts.\n")

def act(step, obs, prev_r, hist, verbose=False):
    prompt = build_user_prompt(step, obs, prev_r, hist)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]

    # Get a proper tensor (new transformers returns BatchEncoding by default)
    encoded = tok.apply_chat_template(
        msgs, add_generation_prompt=True,
        return_tensors="pt", return_dict=True
    )
    input_ids = encoded["input_ids"].to(mdl.device)
    attn_mask = encoded["attention_mask"].to(mdl.device)

    with torch.no_grad():
        out = mdl.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=64,
            do_sample=False,   # deterministic to compare runs
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    raw = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    cleaned = robust_extract(raw) or "REQUEST_INFO summary"
    if verbose:
        print(f"    [RAW] {raw[:150]!r}")
        print(f"    [CLEAN] {cleaned!r}")
    return cleaned, raw

results = {}
VERBOSE_FIRST_STEPS = 3  # print raw output for first N steps of each scenario

for task in ["bad_release", "security_cve", "data_pipeline_regression", "false_positive"]:
    print(f"===== {task} =====")
    env = OpsTwinEnvironment()
    obs = env.reset(task=task)
    hist, prev_r = [], 0.0
    for step in range(1, env._max_steps + 1):
        verbose = step <= VERBOSE_FIRST_STEPS
        action, raw = act(step, obs, prev_r, hist, verbose=verbose)
        print(f"  step {step:2d}: {action[:80]}")
        obs = env.step(OpsAction(command=action))
        hist.append(f"S{step}: {action} -> {obs.reward:+.2f}")
        hist = hist[-5:]
        prev_r = obs.reward
        if obs.done: break
    results[task] = obs.score
    print(f"  FINAL: {obs.score:.3f} ({obs.resolved_issues_count}/{obs.total_issues_count} resolved)\n")

avg = sum(results.values()) / len(results)
print("=" * 50)
print(f"Average trained score: {avg:.3f}")
print(f"(random=0.24, untrained_llm=0.36, heuristic=0.99)")
for task, score in results.items():
    print(f"  {task:30s} {score:.3f}")
