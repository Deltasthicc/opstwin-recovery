# OpsTwin Recovery Arena — v1 Training Results

## Model
- **Base:** Qwen/Qwen3-1.7B (1.72B parameters)
- **Method:** Full fine-tune on 2774 expert trajectories
- **Hardware:** NVIDIA RTX 5090 (32 GB VRAM)
- **Wall time:** 16.4 minutes (434 steps, 2 epochs)
- **Final eval loss:** 0.02 (token accuracy 99.3%)
- **Hugging Face:** [Deltasthic/opstwin-qwen3-1.7b-sft](https://huggingface.co/Deltasthic/opstwin-qwen3-1.7b-sft)

## Results (deterministic, greedy decoding)

| Scenario | Random | Untrained LLM | **Trained** | Heuristic |
|----------|--------|---------------|-------------|-----------|
| bad_release | 0.24 | 0.18 | **0.42** | 0.99 |
| security_cve | 0.22 | 0.20 | **0.94** | 0.99 |
| data_pipeline_regression | 0.24 | 0.20 | **0.75** | 0.99 |
| false_positive | 0.24 | 0.85 | **0.84** | 0.99 |
| **Average** | **0.24** | **0.36** | **0.74** | **0.99** |

Stochastic sampling (T=0.3, 5 seeds): ~0.63 average — shown in evaluate.py chart.
Deterministic (T=0.0, greedy): ~0.74 average — closer to production behavior.

## Key takeaway
Training more than doubled baseline performance. Biggest win on
security_cve (0.20 → 0.94), demonstrating the model learned complex
multi-step policies: assess blast radius → refresh stale approvals
→ rerun pipelines → prioritize VIP tickets → draft communications.

## Known v1 limitations (target for v3)
- bad_release (0.42): model hallucinates feature-flag names instead
  of copying the exact name from the env's runbook feedback.
- No explicit entity-copying signal in training data.
- Model fills remaining steps with plausible-looking repeat actions
  after main incident is resolved.

## Reproducibility
- Training script: `train_sft_5090.py`
- Evaluation script: `evaluate.py` (patched for <think> tags)
- Detailed rollout: `capture_everything.py`
- Full rollout log: `captures/rollout_20260420_145634.log`
- Environment frozen: `requirements_v1_frozen.txt`
