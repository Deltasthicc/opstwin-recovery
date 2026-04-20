"""
OpsTwin Recovery Arena -- TRL GRPO Training
=============================================
Trains a small instruction-tuned LLM (default Qwen3-1.7B) to coordinate
incident response on the OpsTwin environment via TRL's GRPOTrainer.

Design notes:

1. TRL's environment_factory expects a class whose PUBLIC METHODS are the
   tools the model can call. The trainer introspects each method's
   signature and docstring to build the tool schema. So we do NOT expose a
   generic `step(command_string)` -- we expose typed methods like
   switch_desk(desk), flip_flag(flag_id, state), draft_comms(audience,
   message). Each method translates its args to a command string and calls
   the underlying OpsTwin environment.

2. Reward accumulation: per-step rewards are summed into self.reward. On
   episode end we add a final-state bonus equal to the weighted_final
   multi-objective score so the model is shaped toward the display metric
   judges will see.

3. Default config targets a single H100 (or A100-40GB) with vLLM colocate.
   For laptop smoke testing, pass --model Qwen/Qwen3-0.6B and --no-vllm.

Usage:
    # Smoke test on CPU/small GPU:
    python train.py --max-steps 3 --no-vllm --model Qwen/Qwen3-0.6B

    # Full training run on one H100:
    python train.py --max-steps 500 --model Qwen/Qwen3-1.7B
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Make the repo importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Lightweight imports always available. Heavy training deps (trl, datasets,
# peft) are imported inside main()/build_dataset() so that `import train`
# works on machines without those installed (e.g. for unit-testing the
# training wrapper class, or running inference-only).
from models import OpsAction
from server.environment import OpsTwinEnvironment
from server.scenarios import ALL_TASK_NAMES


# --- Training environment wrapper ------------------------------------

# Module-level task roster used by the factory. We cycle through scenarios
# so each rollout trains on a mix of incident types.
_TRAIN_TASKS = [
    "bad_release",
    "security_cve",
    "data_pipeline_regression",
]
_TASK_CURSOR = {"i": 0}


def _next_task() -> str:
    """Round-robin through the training tasks."""
    t = _TRAIN_TASKS[_TASK_CURSOR["i"] % len(_TRAIN_TASKS)]
    _TASK_CURSOR["i"] += 1
    return t


SYSTEM_PROMPT = """You are an incident commander coordinating a production outage.

You have access to five operational desks (INCIDENT_COMMAND, SRE, SECURITY, SUPPORT, RELEASE). Each desk unlocks different actions. Call switch_desk first, then use the desk's tools.

Priority order:
1. Switch to INCIDENT_COMMAND and call assess_blast_radius to surface hidden dependencies.
2. Switch to the right operational desk and use inspect_runbook to understand root cause.
3. Apply the correct fix (flip_flag, run_mitigation, rerun_pipeline, etc.).
4. Switch to SUPPORT to triage customer tickets (prioritize_vip before others).
5. Call draft_comms to close required internal or external communications.
6. Call finish_incident when all issues are resolved.

Rules:
- One tool call per turn. Do not pre-plan multiple tool calls.
- VIP tickets must be handled before non-VIP P1 tickets.
- Do not rollback or quarantine without first running assess_blast_radius.
- Do not rerun a pipeline with a stale approval; refresh it with approve_exception first.
"""


class OpsTwinTrainingEnv:
    """
    Tool-exposing wrapper around OpsTwinEnvironment for TRL's GRPOTrainer.

    TRL constructs one instance of this class per generation, calls reset()
    at episode start, then invokes public methods as tool calls until the
    model stops calling tools. After the episode, the trainer reads
    self.reward to compute group advantages.
    """

    def __init__(self):
        self.env = OpsTwinEnvironment()
        self._last_obs = None
        self._done = False
        self.reward = 0.0
        self._step_count = 0
        self._max_steps = 16

    # --- Lifecycle ---------------------------------------------------

    def reset(self, **kwargs) -> str:
        """Called by TRL at the start of each episode. Returns the initial
        observation as a string for the model."""
        task = _next_task()
        obs = self.env.reset(task=task)
        self._last_obs = obs
        self._done = False
        self.reward = 0.0
        self._step_count = 0
        self._max_steps = obs.total_issues_count + 8  # generous budget
        return self._format_obs(obs, task)

    # --- Internal helpers -------------------------------------------

    def _format_obs(self, obs, task_name: Optional[str] = None) -> str:
        """Compact text representation of the current observation."""
        lines = []
        if task_name:
            lines.append(f"Incident: {task_name}")
            lines.append(f"Description: {obs.incident_description[:200]}")
            lines.append("")
        lines.append(f"Desk: {obs.active_desk or 'NONE'}  "
                     f"Resolved: {obs.resolved_issues_count}/{obs.total_issues_count}  "
                     f"Step: {obs.ops_status.split('|')[-1].strip() if obs.ops_status else '?'}")
        # Services
        if obs.services:
            lines.append("Services:")
            for s in obs.services[:6]:
                lines.append(f"  {s['id']}: [{s['status']}]")
        # Tickets (show only open)
        open_tix = [t for t in obs.tickets if t.get("status", "open") == "open"]
        if open_tix:
            lines.append("Open tickets:")
            for t in open_tix[:5]:
                vip = "[VIP]" if t.get("is_vip") else ""
                lines.append(f"  {t['id']} [{t['priority']}]{vip}: {t['description'][:60]}")
        # Pipelines
        if obs.pipelines:
            lines.append("Pipelines:")
            for p in obs.pipelines[:4]:
                lines.append(f"  {p['id']}: {p['status']}")
        # Alerts
        active_alerts = [a for a in obs.alerts if not a.get("cleared")]
        if active_alerts:
            lines.append("Active alerts:")
            for a in active_alerts[:3]:
                lines.append(f"  {a['id']}[{a['severity']}]: {a['description'][:60]}")
        # Hidden state hints
        if obs.uncertainty_alerts:
            lines.append("Hidden state alerts:")
            for ua in obs.uncertainty_alerts[:3]:
                lines.append(f"  {ua['message']}")
        if obs.message:
            lines.append(f"Last: {obs.message[:180]}")
        return "\n".join(lines)

    def _apply(self, command: str) -> str:
        """Dispatch a command string through the environment and update
        reward / done / last observation. Returns the formatted next obs."""
        if self._done:
            raise ValueError("Episode already over.")
        self._step_count += 1
        obs = self.env.step(OpsAction(command=command))
        self._last_obs = obs
        step_reward = obs.reward or 0.0
        self.reward += step_reward
        if obs.done:
            self._done = True
            # Add final-state bonus: the weighted_final score is what judges
            # actually display, so train against it.
            self.reward += obs.score
        if self._step_count >= self._max_steps and not obs.done:
            # Force-done on step budget to avoid runaway rollouts
            self._done = True
        return self._format_obs(obs)

    # --- TOOLS (each method is discovered by TRL) ------------------

    def switch_desk(self, desk: str) -> str:
        """Switch to an operational desk to unlock its command set.

        Args:
            desk: One of INCIDENT_COMMAND, SRE, SECURITY, SUPPORT, RELEASE.

        Returns:
            The updated observation.
        """
        return self._apply(f"SWITCH_DESK {desk}")

    def assess_blast_radius(self) -> str:
        """Reveal one hidden service dependency edge. Call this early.

        Returns:
            The updated observation.
        """
        return self._apply("ASSESS_BLAST_RADIUS")

    def inspect_runbook(self, service_id: str) -> str:
        """Read the runbook for a service to reveal its root cause.

        Args:
            service_id: e.g. checkout-svc, auth-svc, analytics-pipeline.

        Returns:
            The updated observation.
        """
        return self._apply(f"INSPECT_RUNBOOK {service_id}")

    def verify_flag(self, flag_id: str) -> str:
        """Reveal the actual value of a feature flag.

        Args:
            flag_id: e.g. checkout_v2_ui.

        Returns:
            The updated observation.
        """
        return self._apply(f"VERIFY_FLAG {flag_id}")

    def check_approval(self, change_id: str) -> str:
        """Reveal the current approval state of a change or deploy.

        Args:
            change_id: e.g. deploy-auth-patch-1.8.2.

        Returns:
            The updated observation.
        """
        return self._apply(f"CHECK_APPROVAL {change_id}")

    def flip_flag(self, flag_id: str, state: str) -> str:
        """Toggle a feature flag on or off. Fixes some service outages.

        Args:
            flag_id: The flag identifier.
            state: Either "on" or "off".

        Returns:
            The updated observation.
        """
        return self._apply(f"FLIP_FLAG {flag_id} {state}")

    def run_mitigation(self, mitigation_id: str) -> str:
        """Apply a runbook-defined mitigation to resolve a service outage.

        Args:
            mitigation_id: e.g. fix-schema-transform.

        Returns:
            The updated observation.
        """
        return self._apply(f"RUN_MITIGATION {mitigation_id}")

    def rerun_pipeline(self, pipeline_id: str) -> str:
        """Rerun a CI/CD pipeline. Use to apply a patch or recover a failed job.

        Args:
            pipeline_id: e.g. deploy-auth-patch-1.8.2.

        Returns:
            The updated observation.
        """
        return self._apply(f"RERUN_PIPELINE {pipeline_id}")

    def approve_exception(self, change_id: str) -> str:
        """Refresh a stale approval. Required before rerunning a blocked patch.

        Args:
            change_id: The change or deploy whose approval is stale.

        Returns:
            The updated observation.
        """
        return self._apply(f"APPROVE_EXCEPTION {change_id}")

    def scan_cve(self, dependency_id: str) -> str:
        """Scan a dependency for CVE exploitation signal.

        Args:
            dependency_id: e.g. libcurl-8.4.0.

        Returns:
            The updated observation.
        """
        return self._apply(f"SCAN_CVE {dependency_id}")

    def triage_ticket(self, ticket_id: str, priority: str) -> str:
        """Triage a customer ticket at a priority level.

        Args:
            ticket_id: e.g. T-001.
            priority: One of P1, P2, P3.

        Returns:
            The updated observation.
        """
        return self._apply(f"TRIAGE_TICKET {ticket_id} {priority}")

    def prioritize_vip(self, ticket_id: str) -> str:
        """Fast-track a VIP ticket. Use before regular triage for VIP customers.

        Args:
            ticket_id: The VIP ticket identifier.

        Returns:
            The updated observation.
        """
        return self._apply(f"PRIORITIZE_VIP {ticket_id}")

    def draft_comms(self, audience: str, message: str) -> str:
        """Send a communication to internal teams or external customers.

        Args:
            audience: Either "internal" or "external".
            message: The communication body.

        Returns:
            The updated observation.
        """
        # Guard against newlines / multi-word collisions in the command
        safe_message = message.replace("\n", " ")[:200]
        return self._apply(f"DRAFT_COMMS {audience} {safe_message}")

    def request_info(self, subject: str) -> str:
        """Query a subject for detail. Cheap: use to reduce uncertainty.

        Args:
            subject: One of services, tickets, pipelines, alerts, summary,
                scoring, graph, uncertainty, policies, triage.

        Returns:
            The updated observation.
        """
        return self._apply(f"REQUEST_INFO {subject}")

    def escalate_to_ic(self) -> str:
        """Get a one-time strategic hint from the Incident Commander.

        Returns:
            The updated observation.
        """
        return self._apply("ESCALATE_TO_IC")

    def finish_incident(self) -> str:
        """Declare the incident resolved. Call only when all issues addressed.

        Returns:
            The final observation.
        """
        return self._apply("DONE")


# --- Reward functions -------------------------------------------------

def reward_func(environments, **kwargs):
    """TRL reward function. Reads accumulated reward from each env."""
    return [env.reward for env in environments]


# --- Trainer setup ----------------------------------------------------

def build_dataset(n_prompts: int):
    """Return a dataset of identical prompts. Imports `datasets` lazily
    so that importing train.py doesn't require the training deps."""
    from datasets import Dataset
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            "A production incident is in progress. Coordinate the response. "
            "Use your tools one at a time. Finish with finish_incident when "
            "all issues are resolved."},
    ]
    return Dataset.from_dict({"prompt": [prompt] * n_prompts})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B",
                        help="Base model. Qwen3-0.6B for smoke tests.")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Total GRPO training steps.")
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Generations per prompt (GRPO group size).")
    parser.add_argument("--num-prompts", type=int, default=256,
                        help="Number of prompts in the synthetic dataset.")
    parser.add_argument("--max-completion-length", type=int, default=2048,
                        help="Total tokens across all turns in an episode.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-r", type=int, default=32,
                        help="LoRA rank. 0 disables LoRA (full fine-tune).")
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--no-vllm", action="store_true",
                        help="Disable vLLM colocate (slower but simpler).")
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    args = parser.parse_args()

    # Lazy imports so the file parses even when trl/transformers are absent
    from trl import GRPOConfig, GRPOTrainer

    grpo_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=1,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        chat_template_kwargs={"enable_thinking": False},
        log_completions=True,
        report_to=["trackio"] if os.getenv("TRACKIO_API_KEY") else [],
    )
    if not args.no_vllm:
        grpo_kwargs.update(use_vllm=True, vllm_mode="colocate")

    trainer_kwargs = dict(
        model=args.model,
        reward_funcs=reward_func,
        train_dataset=build_dataset(args.num_prompts),
        args=GRPOConfig(**grpo_kwargs),
        environment_factory=OpsTwinTrainingEnv,
    )

    # Optional LoRA
    if args.lora_r > 0:
        try:
            from peft import LoraConfig
            trainer_kwargs["peft_config"] = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_r * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
        except ImportError:
            print("peft not installed, falling back to full fine-tune.")

    trainer = GRPOTrainer(**trainer_kwargs)
    print(f"Starting GRPO training: model={args.model} steps={args.max_steps} "
          f"gens={args.num_generations} lora_r={args.lora_r} "
          f"vllm={not args.no_vllm}", flush=True)
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved final model to {args.output_dir}")


if __name__ == "__main__":
    main()
