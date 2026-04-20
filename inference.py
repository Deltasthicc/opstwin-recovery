"""
inference.py -- OpsTwin Recovery Arena
========================================
Baseline agent runner. Plays every hand-authored scenario end-to-end using
an OpenAI-compatible LLM endpoint, prints the [START]/[STEP]/[END] event log
the hackathon organizers expect.

Two connection modes (auto-detected):

  1. Local mode: If IMAGE_NAME is unset, the agent connects to
     http://localhost:8000, which must be started separately with
     `uv run server` or `python -m server.app`. Preferred for fast
     iteration on-site.

  2. Docker mode: If IMAGE_NAME is set, uses EnvClient.from_docker_image
     to launch the container, matching the hackathon submission harness.

Required env vars:
  HF_TOKEN or API_KEY        LLM API key (required)
  API_BASE_URL               LLM endpoint (default: HF router)
  MODEL_NAME                 Model id
  IMAGE_NAME                 Docker image (optional, triggers docker mode)
"""
import asyncio
import os
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI
from client import OpsTwinRecoveryEnv
from models import OpsAction


# --- Config --------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
# Default to Qwen3-8B: correct repo name (no -Instruct suffix), served by
# the HF Inference Providers router. Fallbacks if router rejects:
#   - Qwen/Qwen2.5-7B-Instruct  (older, very broadly served)
#   - meta-llama/Meta-Llama-3.1-8B-Instruct
# Override in .env with MODEL_NAME if you want a different one.
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen3-8B"
IMAGE_NAME = os.getenv("IMAGE_NAME")
LOCAL_URL = os.getenv("LOCAL_URL") or "http://localhost:8000"

BENCHMARK = "opstwin_recovery"
TASKS = ["bad_release", "security_cve", "data_pipeline_regression", "false_positive"]
MAX_STEPS = {
    "bad_release": 14,
    "security_cve": 16,
    "data_pipeline_regression": 14,
    "false_positive": 10,
}
TEMPERATURE = 0.3
MAX_TOKENS = 1024  # large budget so Qwen3 thinking doesn't truncate the command
SUCCESS_THRESHOLD = 0.70  # weighted_final score considered "success"


# --- Prompts -------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
You are an incident commander coordinating a production outage.

COMMAND FORMAT: Respond with exactly ONE command, no quotes, no explanation.

DESK NAVIGATION:
  SWITCH_DESK <INCIDENT_COMMAND|SRE|SECURITY|SUPPORT|RELEASE>
  SEND_MESSAGE <desk> <message>
  READ_MESSAGES

SRE DESK:
  RESTART_SERVICE <service_id>
  ISOLATE_SERVICE <service_id>
  ROLLBACK_DEPLOYMENT <deploy_id>
  RUN_MITIGATION <mitigation_id>

SECURITY DESK:
  QUARANTINE_SERVICE <service_id>
  BLOCK_ROLLOUT <pipeline_id>
  APPROVE_EXCEPTION <change_id>
  SCAN_CVE <dependency_id>

SUPPORT DESK:
  TRIAGE_TICKET <ticket_id> <P1|P2|P3>
  MERGE_TICKETS <tid1> <tid2>
  DRAFT_COMMS <internal|external> <message>
  PRIORITIZE_VIP <ticket_id>

RELEASE DESK:
  RERUN_PIPELINE <pipeline_id>
  CANCEL_PIPELINE <pipeline_id>
  FLIP_FLAG <flag_id> <on|off>
  PAUSE_ROLLOUT <deploy_id>

INSPECTION (reveal hidden state, small reward):
  INSPECT_RUNBOOK <service_id>
  CHECK_APPROVAL <change_id>
  VERIFY_FLAG <flag_id>
  ASSESS_BLAST_RADIUS
  REQUEST_FORECAST

INFO / CONTROL:
  REQUEST_INFO <services|tickets|pipelines|alerts|summary|scoring|audit|graph|uncertainty|policies|triage>
  ESCALATE_TO_IC  (once per episode, strategic hint)
  DONE

STRATEGY:
1. SWITCH_DESK INCIDENT_COMMAND -> ASSESS_BLAST_RADIUS to surface hidden edges.
2. REQUEST_INFO summary to understand state.
3. Switch to the correct operational desk, INSPECT_RUNBOOK on the affected service.
4. Apply the RIGHT fix (often FLIP_FLAG or RUN_MITIGATION, not ROLLBACK).
5. SWITCH_DESK SUPPORT: PRIORITIZE_VIP first, then TRIAGE_TICKET others.
6. DRAFT_COMMS (required for most scenarios -- do not forget).
7. DONE.

RULES:
- One command per response. Exact IDs from the observation.
- VIP tickets FIRST before other P1.
- Never ROLLBACK without ASSESS_BLAST_RADIUS first (policy violation).
- Never RERUN_PIPELINE with stale approval; APPROVE_EXCEPTION first.

OUTPUT FORMAT: Respond with ONLY the command, on one line. No explanation.
No <think> tags. No markdown. Just the command verb and its arguments.
/no_think
""")


# --- Logging -------------------------------------------------------

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    action_clean = action.replace("\n", " ").strip()
    done_str = str(done).lower()
    error_str = error if error else "null"
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} "
          f"done={done_str} error={error_str}", flush=True)


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.2f} rewards={rewards_str}", flush=True)


# --- Prompt builder ------------------------------------------------

def build_user_prompt(step, obs, last_reward, history):
    services = "\n".join(
        f"  {s['id']}: [{s['status']}] slo={s.get('current_slo', '?'):.4f}"
        if isinstance(s.get('current_slo'), float) else f"  {s['id']}: [{s['status']}]"
        for s in obs.services
    )
    open_tix = [t for t in obs.tickets if t.get("status", "open") == "open"]
    tickets = "\n".join(
        f"  {t['id']} [{t['priority']}]{'[VIP]' if t.get('is_vip') else ''}: "
        f"{t.get('description', '')[:60]}"
        for t in open_tix
    )
    pipelines = "\n".join(
        f"  {p['id']}: {p['status']}" for p in obs.pipelines
    )
    active_alerts = [a for a in obs.alerts if not a.get("cleared")]
    alerts = "\n".join(
        f"  {a['id']}[{a['severity']}]: {a.get('description', '')[:60]}"
        for a in active_alerts
    )
    uncertainty = "\n".join(
        f"  {u['message']}" for u in (obs.uncertainty_alerts or [])[:3]
    )
    hist = "\n".join(history[-5:]) if history else "(none)"

    return (
        f"Step {step} | Desk: {obs.active_desk or 'NONE'} | "
        f"Resolved: {obs.resolved_issues_count}/{obs.total_issues_count} | "
        f"Last reward: {last_reward:+.2f}\n\n"
        f"SERVICES:\n{services or '  (none)'}\n\n"
        f"TICKETS:\n{tickets or '  (none)'}\n\n"
        f"PIPELINES:\n{pipelines or '  (none)'}\n\n"
        f"ALERTS:\n{alerts or '  (none)'}\n\n"
        f"HIDDEN STATE:\n{uncertainty or '  (none flagged)'}\n\n"
        f"FEEDBACK: {obs.message[:400]}\n\n"
        f"HISTORY:\n{hist}\n\n"
        f"Your command:"
    )


# --- LLM wrapper ---------------------------------------------------

_consecutive_failures = 0

# Models to try in order if the primary fails with a 'model not found'
# error. Populated from MODEL_NAME then a short fallback chain of models
# known to be broadly available on the HF Inference Providers router.
_MODEL_FALLBACK_CHAIN = [
    MODEL_NAME,
    "Qwen/Qwen3-8B",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]
# De-duplicate while preserving order
_seen = set()
_MODEL_FALLBACK_CHAIN = [m for m in _MODEL_FALLBACK_CHAIN
                         if not (m in _seen or _seen.add(m))]
_active_model_idx = 0

# Some HF router backend providers reject `chat_template_kwargs`. We try it
# on the first call and flip this off permanently if it's rejected; the
# client-side <think> stripper handles the rest.
_supports_thinking_kwarg = True


def _current_model():
    return _MODEL_FALLBACK_CHAIN[_active_model_idx]


def _extract_command(text: str) -> str:
    """Robust command extraction from an LLM response.

    Handles:
      - Qwen3 thinking tokens: <think>...</think><actual response>
      - Lone <think> with no close tag (the response was truncated inside thinking)
      - Markdown code fences ``` ... ```
      - Quoted commands: "SWITCH_DESK SRE" or `SWITCH_DESK SRE`
      - Multi-line responses: first non-empty line wins
      - Prose prefixes like "I will call: SWITCH_DESK SRE"
    """
    if not text:
        return ""
    text = text.strip()

    # Strip <think>...</think> blocks (Qwen3 reasoning)
    while "<think>" in text:
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        else:
            # Truncated mid-thinking; no real command available
            return ""

    # Strip triple-backtick fences
    if text.startswith("```"):
        text = text.split("```", 2)
        text = text[1] if len(text) > 1 else ""
        # Drop a leading language tag like ```bash
        if "\n" in text:
            text = text.split("\n", 1)[1]
        text = text.rstrip("`").strip()

    # Take the first non-empty, non-comment line
    for line in text.split("\n"):
        line = line.strip().strip('"').strip("'").strip("`")
        if not line or line.startswith("#"):
            continue
        # Drop common prose prefixes
        for prefix in ("Command:", "Action:", "I will call:", "Call:",
                       "Run:", "Execute:", "Next:"):
            if line.lower().startswith(prefix.lower()):
                line = line[len(prefix):].strip().strip('"').strip("'")
        return line
    return ""


def get_model_command(client, step, obs, last_reward, history):
    global _consecutive_failures, _active_model_idx, _supports_thinking_kwarg
    prompt = build_user_prompt(step, obs, last_reward, history)

    # Build request kwargs. Only include extra_body if the backend is known
    # to accept chat_template_kwargs (checked lazily on first call).
    request_kwargs = dict(
        model=_current_model(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    if _supports_thinking_kwarg:
        # Disable Qwen3 thinking mode at the provider level when supported.
        request_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    try:
        completion = client.chat.completions.create(**request_kwargs)
        text = (completion.choices[0].message.content or "").strip()
        command = _extract_command(text)
        _consecutive_failures = 0
        return command if command else "REQUEST_INFO summary"
    except Exception as exc:
        msg = str(exc)

        # Case A: Provider rejects chat_template_kwargs. Flip the flag off
        # and retry once with the same step -- the client-side parser will
        # strip <think> blocks from the response.
        if _supports_thinking_kwarg and "chat_template_kwargs" in msg:
            _supports_thinking_kwarg = False
            print("[DEBUG] Provider does not accept chat_template_kwargs; "
                  "disabling and retrying this turn.", flush=True)
            return get_model_command(client, step, obs, last_reward, history)

        _consecutive_failures += 1

        # Case B: Model-not-found -> try next model in chain.
        # Tightened from the old over-broad "400 in msg and model in msg.lower()"
        # which was catching OUR kwarg error because the message happens to
        # contain "model" as part of the verb "model_name".
        if ("model_not_found" in msg
                or "does not exist" in msg.lower()
                or "model is not available" in msg.lower()):
            if _active_model_idx < len(_MODEL_FALLBACK_CHAIN) - 1:
                _active_model_idx += 1
                print(f"[DEBUG] Model unavailable, falling back to "
                      f"'{_MODEL_FALLBACK_CHAIN[_active_model_idx]}'", flush=True)
                _consecutive_failures = 0
                return get_model_command(client, step, obs, last_reward, history)

        # Case C: Rate-limit / credit exhaustion -> cleanly end the task.
        if any(code in msg for code in ("402", "429")) or "credit" in msg.lower():
            print("[DEBUG] API credit / rate limit. Ending task.", flush=True)
            return "DONE"

        # Case D: too many consecutive failures -> end the task.
        if _consecutive_failures >= 3:
            print("[DEBUG] 3 consecutive LLM failures. Ending task.", flush=True)
            return "DONE"

        print(f"[DEBUG] LLM error ({_consecutive_failures}/3): {exc}", flush=True)
        return "REQUEST_INFO summary"


# --- Connection management ----------------------------------------

async def open_env():
    """Return an OpsTwinRecoveryEnv either from Docker or a local URL."""
    if IMAGE_NAME:
        return await OpsTwinRecoveryEnv.from_docker_image(IMAGE_NAME)
    # Local-URL mode
    return OpsTwinRecoveryEnv(base_url=LOCAL_URL)


# --- Task runner ---------------------------------------------------

async def run_task(client: OpenAI, task_name: str) -> None:
    global _consecutive_failures
    _consecutive_failures = 0

    max_steps = MAX_STEPS.get(task_name, 15)
    env = await open_env()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.01
    success = False

    log_start(task=task_name, env=BENCHMARK, model=_current_model())

    try:
        result = await env.reset(task=task_name)
        obs = result.observation
        last_reward = 0.0

        for step_num in range(1, max_steps + 1):
            if result.done:
                break
            command = get_model_command(
                client, step_num, obs, last_reward, history)
            result = await env.step(OpsAction(command=command))
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            rewards.append(reward)
            steps_taken = step_num
            last_reward = reward
            log_step(step=step_num, action=command, reward=reward,
                     done=done, error=None)
            history.append(f"S{step_num}: {command} -> {reward:+.2f}")
            if done:
                break

        score = float(getattr(obs, "score", 0.0) or 0.0)
        score = min(max(score, 0.01), 0.99)
        success = score >= SUCCESS_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)


# --- Entry point ---------------------------------------------------

async def main() -> None:
    if not API_KEY:
        raise SystemExit(
            "ERROR: HF_TOKEN (or API_KEY) environment variable not set.\n"
            "  Windows (PowerShell): $env:HF_TOKEN = 'hf_xxxxx'\n"
            "  macOS/Linux (bash):   export HF_TOKEN='hf_xxxxx'\n"
            "  Or: copy .env.example to .env and fill HF_TOKEN.\n"
            "  Get a token at: https://huggingface.co/settings/tokens"
        )
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task in TASKS:
        await run_task(client, task)


if __name__ == "__main__":
    asyncio.run(main())
