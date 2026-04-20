# OpsTwin Recovery Arena — v1 Training Results

## Model
- **Base:** Qwen/Qwen3-1.7B (1.72B parameters)
- **Method:** Full fine-tune (no LoRA) on 2774 expert trajectories
- **Hardware:** NVIDIA RTX 5090 (32 GB VRAM)
- **Wall time:** 16.4 minutes (984 seconds, 434 steps, 2 epochs)
- **Final eval loss:** 0.02 (token accuracy 99.3%)
- **Hugging Face:** [Deltasthic/opstwin-qwen3-1.7b-sft](https://huggingface.co/Deltasthic/opstwin-qwen3-1.7b-sft)

## Results

| Scenario | Random | Untrained LLM | **Trained** | Heuristic |
|----------|--------|---------------|-------------|-----------|
| bad_release | 0.24 | 0.18 | **0.42** | 0.99 |
| security_cve | 0.22 | 0.20 | **0.94** | 0.99 |
| data_pipeline_regression | 0.24 | 0.20 | **0.75** | 0.99 |
| false_positive | 0.24 | 0.85 | **0.84** | 0.99 |
| **Average** | **0.24** | **0.36** | **0.74** | **0.99** |

## Training hyperparameters
- Learning rate: 1e-5 (cosine schedule, 30-step warmup)
- Batch size: 4 per device x 4 accumulation = 16 effective
- Precision: bfloat16
- Gradient checkpointing: enabled
- Eval every 25 steps, log every 5 steps

## Dataset composition
- 4 hand-authored scenarios (OPTIMAL_TRAJECTORIES)
- ~300 procedural trajectories across 3 families x 2 difficulties x 50 seeds
- Total: 2774 (observation, action) training pairs
- 95/5 train/eval split

## Known limitations (deferred to v3)
- bad_release (0.42) because model hallucinates flag names instead of
  copying the exact name from environment feedback during inspection
- Occasional action repetition after scenario completion
- No explicit curriculum, data weighting, or entity-copying signal
