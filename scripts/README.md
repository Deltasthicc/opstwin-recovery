# scripts/

Helper scripts used during the V3 training run. Not required for day-to-day
use of the published HF checkpoint — those scripts exist so anyone can
reproduce the V3 training + evaluation end-to-end, and so the investigation
findings are preserved alongside the final model.

| Script | What it does | When to run |
|---|---|---|
| `plot_training_curve.py` | Reads `trainer_state.json` from the latest checkpoint of each training stage and plots a combined 3-stage loss curve to `results/training_loss_curve.png`. | After training. Requires the `sft_checkpoints_v3*/` directories to be present. |
| `push_v3_step75.py` | Loads a local checkpoint and pushes model + tokenizer to `Deltasthic/opstwin-qwen3-4b-sft-v3`. Requires `HF_TOKEN` in `.env`. | After a stage finishes and you want to publish. Adjust `CKPT` inside the script to point at the checkpoint you want to publish. |
| `diagnose_callback_vs_evaluator.py` | Runs the training-time `RolloutEvalCallback` path and the `evaluate.py` `ModelPolicy` path side-by-side on one scenario, printing the command emitted at each step by each path. Used to discover that both paths produce the same trajectory when the history format is consistent — the callback's apparent `false_positive=0.86` result during training was a liger-kernel-fused-forward artifact, not a genuinely different policy. | When the callback rollout score differs from `evaluate.py`. |
| `diagnose_v1_stack_drift.py` | Loads the V1 model (`Deltasthic/opstwin-qwen3-1.7b-sft`) from a locally-patched snapshot and rolls out `bad_release` twice — once with training-matched history (`S{step}: {cmd} -> {reward:+.2f}`) and once without. Used to confirm that V1's published 0.74 avg cannot be reproduced under transformers 4.55 regardless of eval format; the regression is stack-drift from V1's original training environment. | When comparing V1 and V3 on the current stack. |

## The tokenizer patch

`diagnose_v1_stack_drift.py` expects a local V1 snapshot at `./v1_local/` with a patched `tokenizer_config.json` (`extra_special_tokens` list → dict, required by transformers 4.55). To produce that snapshot:

```python
from huggingface_hub import snapshot_download
import json
from pathlib import Path
path = snapshot_download('Deltasthic/opstwin-qwen3-1.7b-sft', local_dir='./v1_local')
cfg = Path(path) / 'tokenizer_config.json'
d = json.load(open(cfg))
if isinstance(d.get('extra_special_tokens'), list):
    d['extra_special_tokens'] = {t.strip('<|>').replace('|', '_'): t for t in d['extra_special_tokens']}
    json.dump(d, open(cfg, 'w'), indent=2)
```

`v1_local/` is `.gitignored` (3.3 GB) — regenerate with that snippet when you need it.
