import os
import time
import json
import psutil
import torch
import numpy as np
from safetensors.torch import load_file

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

print("=== STAGE 1 FEASIBILITY TEST (manohonsy/how2sign-pose-cslr) ===", flush=True)
mem_before = get_process_memory()
t0 = time.perf_counter()

# Import the verified PoseTransformerCSLR
from test_stage1_model import PoseTransformerCSLR

model = PoseTransformerCSLR()
weights = load_file("feasibility_workspace/stage1_cslr/model.safetensors")
model.load_state_dict(weights, strict=True)
model.eval()

load_time = time.perf_counter() - t0
mem_after = get_process_memory()
total_params = sum(p.numel() for p in model.parameters())

print(f"Model loaded in: {load_time:.2f} s", flush=True)
print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)", flush=True)
print(f"RAM increase during load: {mem_after - mem_before:.2f} MB (Total process RAM: {mem_after:.2f} MB)", flush=True)

# Load Vocab
with open("feasibility_workspace/stage1_cslr/vocab.json", "r") as f:
    vdata = json.load(f)
token_to_id = vdata.get("token_to_id", vdata)
id_to_token = {v: k for k, v in token_to_id.items()}
blank_id = token_to_id["<blank>"]
pad_id = token_to_id.get("<pad>", 0)

print(f"Blank ID: {blank_id}, Pad ID: {pad_id}", flush=True)

def ctc_decode(log_probs, blank_id=1):
    # log_probs: (1, T, vocab_size)
    preds = log_probs.argmax(dim=-1)[0].tolist()
    collapsed, prev = [], None
    for p in preds:
        if p != prev:
            collapsed.append(p)
            prev = p
    gloss_ids = [p for p in collapsed if p not in (blank_id, pad_id)]
    gloss_tokens = [id_to_token[i] for i in gloss_ids if i in id_to_token]
    return gloss_ids, gloss_tokens

# Test latency across multiple sequence lengths (8 fps: 24 frames=3s, 48 frames=6s, 80 frames=10s)
test_lengths = [24, 48, 80, 120]
print("\n--- Latency Tests on CPU across Temporal Window Sizes ---", flush=True)
for T in test_lengths:
    x = torch.randn(1, T, 153)
    # Warmup
    with torch.no_grad():
        _ = model(x)
    
    # Timed run
    times = []
    for _ in range(10):
        t_start = time.perf_counter()
        with torch.no_grad():
            log_probs = model(x)
            g_ids, g_toks = ctc_decode(log_probs, blank_id)
        times.append((time.perf_counter() - t_start) * 1000)
    
    avg_lat = sum(times) / len(times)
    print(f"T={T:3d} frames (~{T/8:.1f}s at 8fps) -> Avg Latency: {avg_lat:.2f} ms (Min: {min(times):.2f} ms, Max: {max(times):.2f} ms)", flush=True)

print(f"Final Process Memory: {get_process_memory():.2f} MB", flush=True)

