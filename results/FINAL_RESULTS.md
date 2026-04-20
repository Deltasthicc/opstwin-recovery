# OpsTwin Recovery Arena — Training Results

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
- Learning rate: 1e-5 (cosine schedule)
- Batch size: 4 per device, accumulation 4 → effective 16
- Warmup: 30 steps
- Gradient checkpointing: enabled
- Precision: bfloat16
- Eval every 25 steps, log every 5 steps

## Dataset composition
- 4 hand-authored scenarios (OPTIMAL_TRAJECTORIES)
- 150 procedural scenarios × 2 difficulties × 3 families = ~300 expert trajectories
- Total: 2774 (observation, action) training pairs
- 95/5 train/eval split

## Known limitations (for v3)
- bad_release scores 0.42 because model hallucinates flag names instead of copying from env feedback
- Model sometimes repeats actions past scenario completion (fills remaining steps)
- No training data for in-context copying of rare entity names
