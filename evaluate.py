"""
evaluate.py -- OpsTwin Recovery Arena
========================================
Evaluates a policy on held-out scenarios and produces the evidence artifacts
judges will ask about.

Two policies to compare:
  - "random"     : uniform over available commands (lower bound)
  - "heuristic"  : the expert solver's gold trajectory (upper bound)
  - "model"      : loads a trained HF checkpoint and rolls out

By default compares "random" vs "heuristic" so evaluation produces a chart
even when no trained checkpoint is available yet. Pass --model <path> to
swap in the trained policy.

Outputs:
  results/eval_summary.json     per-scenario mean/std scores
  results/eval_curve.png        bar chart: baseline vs trained by scenario
"""
import argparse
import json
import re
import random
import statistics
import sys
from pathlib import Path
from typing import List, Dict, Callable

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models import OpsAction
from server.environment import OpsTwinEnvironment
from baselines.expert_solver import OPTIMAL_TRAJECTORIES


# --- Policies -------------------------------------------------------

class RandomPolicy:
    """Uniform over a hand-picked action shortlist. Lower-bound baseline."""
    ACTIONS = [
        "REQUEST_INFO summary",
        "REQUEST_INFO services",
        "REQUEST_INFO tickets",
        "SWITCH_DESK SRE",
        "SWITCH_DESK SECURITY",
        "SWITCH_DESK SUPPORT",
        "SWITCH_DESK RELEASE",
        "SWITCH_DESK INCIDENT_COMMAND",
        "ASSESS_BLAST_RADIUS",
        "ESCALATE_TO_IC",
        "DONE",
    ]

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def reset(self, obs):
        pass

    def act(self, obs) -> str:
        return self.rng.choice(self.ACTIONS)


class HeuristicPolicy:
    """Replays the expert solver's optimal trajectory. Upper-bound baseline."""

    def __init__(self):
        self._sequence: List[str] = []
        self._idx = 0

    def reset(self, obs):
        task = obs.ops_status.split("|")[0].replace("Task:", "").strip()
        self._sequence = list(OPTIMAL_TRAJECTORIES.get(task, ["DONE"]))
        self._idx = 0

    def act(self, obs) -> str:
        if self._idx >= len(self._sequence):
            return "DONE"
        cmd = self._sequence[self._idx]
        self._idx += 1
        return cmd


class ModelPolicy:
    """Loads a Hugging Face checkpoint and rolls it out as the policy.

    Kept minimal: renders the same prompt as inference.py and samples a
    single command per step. Pass --model <dir-or-hf-id> to enable.
    """

    def __init__(self, model_path: str, max_new_tokens: int = 64):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
        )
        self.max_new_tokens = max_new_tokens
        self._history: List[str] = []
        # Reuse the exact system prompt from inference.py to keep the
        # baseline and trained runs apples-to-apples.
        from inference import SYSTEM_PROMPT, build_user_prompt
        self._system_prompt = SYSTEM_PROMPT
        self._build_user_prompt = build_user_prompt
        self._last_reward = 0.0
        self._step = 0

    def reset(self, obs):
        self._history = []
        self._last_reward = 0.0
        self._step = 0

    def act(self, obs) -> str:
        import torch
        self._step += 1
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": self._build_user_prompt(
                self._step, obs, self._last_reward, self._history)},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        # Strip <think>...</think> blocks (Qwen3), code fences, quotes,
        # training-history prefix leaks (S5: ...), trailing reward arrows.
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
        cleaned = cleaned.strip('`"\'')
        cleaned = re.sub(r"^S\d+:\s*", "", cleaned)
        cleaned = re.sub(r"\s*->\s*[+-]?\d*\.?\d+\s*$", "", cleaned)
        cleaned = cleaned.split(" -> ")[0]
        cleaned = cleaned.split("\n")[0].strip()
        cmd = cleaned or "REQUEST_INFO summary"
        self._history.append(f"S{self._step}: {cmd}")
        return cmd


# --- Rollout -------------------------------------------------------

def rollout(policy, task: str, max_steps_cap: int = 40) -> Dict:
    env = OpsTwinEnvironment()
    obs = env.reset(task=task)
    policy.reset(obs)
    total_reward = 0.0
    last_score = 0.0
    steps = 0
    for step in range(1, max_steps_cap + 1):
        cmd = policy.act(obs)
        # ModelPolicy receives reward from prev step via _last_reward
        obs = env.step(OpsAction(command=cmd))
        r = obs.reward or 0.0
        total_reward += r
        last_score = obs.score or last_score
        steps = step
        if hasattr(policy, "_last_reward"):
            policy._last_reward = r
        if obs.done:
            break
    return {
        "task": task,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "final_score": round(obs.score or last_score, 4),
        "resolved": obs.resolved_issues_count,
        "total_issues": obs.total_issues_count,
        "multi_objective": obs.multi_objective_scores,
    }


# --- Harness -------------------------------------------------------

def evaluate_policy(policy_factory: Callable, policy_name: str,
                    tasks: List[str], n_seeds: int) -> Dict:
    results = {}
    for task in tasks:
        scores = []
        rewards = []
        for seed in range(n_seeds):
            # Construct a fresh policy per seed so RNG state is clean
            policy = policy_factory(seed)
            r = rollout(policy, task)
            scores.append(r["final_score"])
            rewards.append(r["total_reward"])
        results[task] = {
            "policy": policy_name,
            "n_seeds": n_seeds,
            "mean_score": round(statistics.mean(scores), 4),
            "std_score": round(statistics.stdev(scores) if n_seeds > 1 else 0.0, 4),
            "min_score": round(min(scores), 4),
            "max_score": round(max(scores), 4),
            "mean_reward": round(statistics.mean(rewards), 4),
            "raw_scores": scores,
        }
    return results


def make_bar_chart(summary: Dict, out_path: Path):
    """Save a simple baseline-vs-trained bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping chart.")
        return

    policies = list(summary.keys())
    tasks = sorted(next(iter(summary.values())).keys())
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_width = 0.8 / len(policies)
    x_positions = range(len(tasks))
    for i, policy in enumerate(policies):
        means = [summary[policy][t]["mean_score"] for t in tasks]
        stds = [summary[policy][t]["std_score"] for t in tasks]
        offsets = [x + i * bar_width for x in x_positions]
        ax.bar(offsets, means, width=bar_width, yerr=stds,
               label=policy, capsize=3)
    ax.set_xticks([x + bar_width * (len(policies) - 1) / 2 for x in x_positions])
    ax.set_xticklabels(tasks, rotation=15, ha="right")
    ax.set_ylabel("Weighted-final score")
    ax.set_ylim(0, 1)
    ax.set_title("OpsTwin eval: policy comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Chart saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+",
                        default=["bad_release", "security_cve",
                                 "data_pipeline_regression"],
                        help="Scenarios to evaluate on.")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Seeds per (policy, scenario).")
    parser.add_argument("--model", default=None,
                        help="HF model id or local checkpoint to add as 'model' policy.")
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    # Always include random and heuristic
    print("Evaluating random policy...")
    summary["random"] = evaluate_policy(
        lambda seed: RandomPolicy(seed),
        "random", args.tasks, args.n_seeds)

    print("Evaluating heuristic policy...")
    summary["heuristic"] = evaluate_policy(
        lambda seed: HeuristicPolicy(),
        "heuristic", args.tasks, 1)  # deterministic, 1 seed suffices

    if args.model:
        print(f"Evaluating trained model at {args.model}...")
        cached_policy = ModelPolicy(args.model)
        # All seeds share the model, vary rollout RNG
        summary["trained"] = evaluate_policy(
            lambda seed: cached_policy,
            "trained", args.tasks, args.n_seeds)

    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResults written to {out_dir / 'eval_summary.json'}")

    for policy_name, per_task in summary.items():
        for task, stats in per_task.items():
            print(f"  {policy_name:10s} {task:28s} "
                  f"mean={stats['mean_score']:.3f} std={stats['std_score']:.3f}")

    make_bar_chart(summary, out_dir / "eval_curve.png")


if __name__ == "__main__":
    main()
