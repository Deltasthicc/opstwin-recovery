"""
capture_everything.py — run the trained model against all scenarios with
COMPLETE logging of raw output, extracted command, env response, reward,
and observation deltas. Writes three artifact files per run.

Usage:
    python capture_everything.py ./sft_checkpoints/adapter
    python capture_everything.py Deltasthic/opstwin-qwen3-1.7b-sft

Outputs (all in ./captures/):
    rollout_<timestamp>.log         - full human-readable trace
    rollout_<timestamp>.jsonl       - one row per step, machine-readable
    rollout_<timestamp>_summary.json - final per-task scores
"""
import sys, re, json, torch, traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from models import OpsAction
from server.environment import OpsTwinEnvironment
from inference import SYSTEM_PROMPT, build_user_prompt, _extract_command

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "./sft_checkpoints/adapter"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

OUT_DIR = Path("captures")
OUT_DIR.mkdir(exist_ok=True)
LOG_PATH = OUT_DIR / f"rollout_{TIMESTAMP}.log"
JSONL_PATH = OUT_DIR / f"rollout_{TIMESTAMP}.jsonl"
SUMMARY_PATH = OUT_DIR / f"rollout_{TIMESTAMP}_summary.json"

log_file = open(LOG_PATH, "w")
jsonl_file = open(JSONL_PATH, "w")


def log(*args):
    """Print to stdout AND write to log file."""
    msg = " ".join(str(a) for a in args)
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()


def robust_extract(text):
    text = _extract_command(text)
    text = re.sub(r"^S\d+:\s*", "", text)
    text = re.sub(r"\s*->\s*[+-]?\d*\.?\d+\s*$", "", text)
    text = text.split(" -> ")[0].strip().strip('`"\'')
    return text


log(f"=" * 70)
log(f"OpsTwin Rollout Capture — {TIMESTAMP}")
log(f"Model: {MODEL_PATH}")
log(f"GPU:   {torch.cuda.get_device_name(0)}")
log(f"=" * 70)

log(f"\nLoading tokenizer + model...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
mdl.eval()
log(f"Model loaded.\n")


def act(step, obs, prev_r, hist):
    prompt = build_user_prompt(step, obs, prev_r, hist)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]
    encoded = tok.apply_chat_template(
        msgs, add_generation_prompt=True,
        return_tensors="pt", return_dict=True)
    input_ids = encoded["input_ids"].to(mdl.device)
    attn_mask = encoded["attention_mask"].to(mdl.device)

    with torch.no_grad():
        out = mdl.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    raw = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    cleaned = robust_extract(raw) or "REQUEST_INFO summary"
    return cleaned, raw, prompt


results = {}

for task in ["bad_release", "security_cve", "data_pipeline_regression", "false_positive"]:
    log(f"\n{'='*70}")
    log(f"TASK: {task}")
    log(f"{'='*70}")

    try:
        env = OpsTwinEnvironment()
        initial_obs = env.reset(task=task)

        # Log the full initial observation so we can see what the model sees
        log(f"\nINITIAL STATE:")
        log(f"  total_issues: {initial_obs.total_issues_count}")
        log(f"  incident:     {initial_obs.incident_description[:140]}")
        log(f"  services:     {[s['id'] for s in initial_obs.services]}")
        log(f"  pipelines:    {[p.get('id') for p in initial_obs.pipelines if p.get('id')]}")
        log(f"  tickets:      {[(t['id'], t.get('priority'), t.get('is_vip', False)) for t in initial_obs.tickets]}")

        # Log the hidden state to catch mismatches with what the model guesses
        hidden = env._hidden
        log(f"\nHIDDEN STATE (ground truth):")
        log(f"  root_cause:    {hidden.get_root_cause() if hasattr(hidden, 'get_root_cause') else '?'}")
        if hasattr(hidden, '_flag_states'):
            log(f"  flag_states:   {hidden._flag_states}")
        if hasattr(hidden, '_approval_states'):
            log(f"  approvals:     {hidden._approval_states}")

        log(f"\nROLLOUT:")
        obs = initial_obs
        hist, prev_r = [], 0.0
        cumulative_reward = 0.0

        for step in range(1, env._max_steps + 1):
            action, raw, prompt = act(step, obs, prev_r, hist)
            obs_new = env.step(OpsAction(command=action))

            # Log every detail
            log(f"\n  --- step {step} ---")
            log(f"  RAW MODEL: {raw[:200]!r}")
            log(f"  EXTRACTED: {action!r}")
            log(f"  REWARD:    {obs_new.reward:+.3f}")
            log(f"  FEEDBACK:  {obs_new.message[:200] if obs_new.message else '(none)'}")
            log(f"  RESOLVED:  {obs_new.resolved_issues_count}/{obs_new.total_issues_count}")
            if getattr(obs_new, "error", None):
                log(f"  ERROR:     {obs_new.error}")

            # JSONL row (machine-readable)
            jsonl_file.write(json.dumps({
                "task": task,
                "step": step,
                "raw_output": raw,
                "extracted_command": action,
                "reward": obs_new.reward,
                "cumulative_reward": cumulative_reward + obs_new.reward,
                "score": obs_new.score,
                "resolved": obs_new.resolved_issues_count,
                "total": obs_new.total_issues_count,
                "error": obs_new.error,
                "feedback": obs_new.message[:300] if obs_new.message else None,
                "done": obs_new.done,
            }) + "\n")
            jsonl_file.flush()

            cumulative_reward += obs_new.reward
            hist.append(f"S{step}: {action} -> {obs_new.reward:+.2f}")
            hist = hist[-5:]
            prev_r = obs_new.reward
            obs = obs_new
            if obs.done:
                break

        log(f"\n  FINAL SCORE: {obs.score:.3f}  "
            f"({obs.resolved_issues_count}/{obs.total_issues_count} resolved, "
            f"{step} steps)")
        results[task] = {
            "score": obs.score,
            "resolved": obs.resolved_issues_count,
            "total": obs.total_issues_count,
            "steps": step,
        }

    except Exception as e:
        log(f"\n  EXCEPTION: {e}")
        log(traceback.format_exc())
        results[task] = {"score": 0.0, "error": str(e)}


avg = sum(r.get("score", 0) for r in results.values()) / len(results)
log(f"\n{'='*70}")
log(f"SUMMARY")
log(f"{'='*70}")
log(f"Average trained score: {avg:.3f}")
log(f"Baseline (untrained):  0.360")
log(f"Random:                0.240")
log(f"Heuristic (ceiling):   0.990\n")
for task, r in results.items():
    log(f"  {task:30s} {r.get('score', 0):.3f}  "
        f"({r.get('resolved', 0)}/{r.get('total', 0)} resolved)")

with open(SUMMARY_PATH, "w") as f:
    json.dump({
        "timestamp": TIMESTAMP,
        "model": MODEL_PATH,
        "average_score": avg,
        "per_task": results,
    }, f, indent=2)

log(f"\nArtifacts written:")
log(f"  {LOG_PATH}")
log(f"  {JSONL_PATH}")
log(f"  {SUMMARY_PATH}")

log_file.close()
jsonl_file.close()
