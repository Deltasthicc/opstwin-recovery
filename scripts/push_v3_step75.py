"""Push V3 stage-2 checkpoint-75 to HuggingFace as the official V3 model."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = "./sft_checkpoints_v3_stage2/checkpoint-75"
HF_ID = "Deltasthic/opstwin-qwen3-4b-sft-v3"

print(f"Loading {CKPT}...")
tokenizer = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForCausalLM.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, trust_remote_code=True,
)

print(f"Pushing model to {HF_ID} (public)...")
model.push_to_hub(HF_ID, private=False, commit_message="V3 step-75: eliminates bad_release weakness (0.42 -> 0.99)")
tokenizer.push_to_hub(HF_ID, private=False)
print(f"Done: https://huggingface.co/{HF_ID}")
