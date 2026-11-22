import os
import json
import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

stage1_dir = os.path.abspath("feasibility_workspace/stage1_cslr")
stage2_dir = os.path.abspath("feasibility_workspace/stage2_bart")

# 1. Inspect Stage 1 safetensors
weights = load_file(os.path.join(stage1_dir, "model.safetensors"))
total_params = 0
print("=== STAGE 1 TENSOR INSPECTION ===")
for k, v in weights.items():
    p_count = v.numel()
    total_params += p_count
    print(f"  {k:45s}: shape {str(list(v.shape)):20s} ({p_count:,} params, {v.dtype})")
print(f"Total Stage 1 Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

# 2. Inspect Stage 1 vocab
with open(os.path.join(stage1_dir, "vocab.json"), "r") as f:
    vocab_data = json.load(f)
token_to_id = vocab_data.get("token_to_id", vocab_data)
print(f"\nStage 1 Vocab Size: {len(token_to_id)}")
sample_tokens = list(token_to_id.items())[:20]
print("First 20 tokens:", sample_tokens)
last_tokens = list(token_to_id.items())[-10:]
print("Last 10 tokens:", last_tokens)
print("Special tokens:", {k: v for k, v in token_to_id.items() if k.startswith("<")})

# 3. Download Stage 2
print("\n=== DOWNLOADING STAGE 2 (rrrr66254/Glossa-BART) ===")
snapshot_download(repo_id="rrrr66254/Glossa-BART", local_dir=stage2_dir)
print("Stage 2 downloaded successfully.")
for f in os.listdir(stage2_dir):
    fpath = os.path.join(stage2_dir, f)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath)
        print(f"  {f}: {size} bytes ({size / (1024*1024):.2f} MB)")

