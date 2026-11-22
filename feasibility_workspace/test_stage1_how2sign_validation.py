import os
import time
import json
import psutil
import torch
import numpy as np
import pandas as pd
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_process_memory():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

print("=== HOW2SIGN VALIDATION & END-TO-END BENCHMARK ===", flush=True)

# 1. Load Stage 1
from test_stage1_model import PoseTransformerCSLR

stage1_model = PoseTransformerCSLR()
weights = load_file("feasibility_workspace/stage1_cslr/model.safetensors")
stage1_model.load_state_dict(weights, strict=True)
stage1_model.eval()

with open("feasibility_workspace/stage1_cslr/vocab.json", "r") as f:
    vdata = json.load(f)
token_to_id = vdata.get("token_to_id", vdata)
id_to_token = {v: k for k, v in token_to_id.items()}
blank_id = token_to_id["<blank>"]
pad_id = token_to_id.get("<pad>", 0)

# 2. Load Stage 2
stage2_dir = os.path.abspath("feasibility_workspace/stage2_bart")
stage2_tokenizer = AutoTokenizer.from_pretrained(stage2_dir, local_files_only=True)
stage2_model = AutoModelForSeq2SeqLM.from_pretrained(stage2_dir, local_files_only=True)
stage2_model.eval()

print("Stage 1 and Stage 2 loaded successfully.", flush=True)

def ctc_decode(log_probs, blank_id=1):
    preds = log_probs.argmax(dim=-1)[0].tolist()
    collapsed, prev = [], None
    for p in preds:
        if p != prev:
            collapsed.append(p)
            prev = p
    gloss_ids = [p for p in collapsed if p not in (blank_id, pad_id)]
    gloss_tokens = [id_to_token[i] for i in gloss_ids if i in id_to_token]
    return gloss_tokens

def translate_gloss_sequence(gloss_tokens, num_beams=1):
    if not gloss_tokens:
        return "<no sign detected>", "<empty>"
    # Normalize gloss tokens
    cleaned = []
    for g in gloss_tokens:
        clean = g.replace("#", "")
        if clean in ("IX", "IX-me"):
            clean = "I"
        elif clean == "IX-you":
            clean = "YOU"
        else:
            clean = clean.replace("-", " ")
        cleaned.append(clean)
    clean_str = " ".join(cleaned)
    inputs = stage2_tokenizer(clean_str, return_tensors="pt")
    with torch.no_grad():
        out = stage2_model.generate(**inputs, max_length=64, num_beams=num_beams)
    translation = stage2_tokenizer.decode(out[0], skip_special_tokens=True)
    return clean_str, translation

# 3. Load Parquet
parquet_path = r"C:\Users\YATHARTH\.cache\huggingface\hub\datasets--ASL-Live-Video-Translation--how2sign-realigned-mediapipe\snapshots\598622fc802aae9dbfdf990b5b287aa7d99fd241\parquet\mediapipe\val-00000.parquet"
df = pd.read_parquet(parquet_path)
print(f"Testing on {len(df)} genuine How2Sign continuous clips...", flush=True)

# Test 5 representative clips
num_test = 5
results = []

for idx in range(num_test):
    row = df.iloc[idx]
    clip_id = row['sentence_id']
    pts_list = row['points'] # T frames, each has 49 pts of (x, y)
    det_list = row['detected'] # T frames, each has 3 bools
    T = len(pts_list)
    
    # Construct 153-D per frame:
    # 50 landmarks * 3 (x,y,z) + 3 masks = 153
    # We map the 49 points (7 pose + 42 hand) into (x, y, 0.0) + 1 dummy mid-hip (0.0, 0.0, 0.0) -> 50 landmarks
    features = []
    for t in range(T):
        f_pts = pts_list[t]
        f_det = det_list[t]
        vec = []
        # Pose: first 7 points
        for p in f_pts[:7]:
            vec.extend([float(p[0]), float(p[1]), 0.0])
        # 8th pose point (midpoint of hips / torso center)
        vec.extend([0.0, 0.0, 0.0])
        # Left hand: 21 points
        for p in f_pts[7:28]:
            vec.extend([float(p[0]), float(p[1]), 0.0])
        # Right hand: 21 points
        for p in f_pts[28:49]:
            vec.extend([float(p[0]), float(p[1]), 0.0])
        # 3 masks
        vec.extend([1.0 if f_det[0] else 0.0, 1.0 if f_det[1] else 0.0, 1.0 if f_det[2] else 0.0])
        features.append(vec)
    
    x = torch.tensor([features], dtype=torch.float32)
    
    # Stage 1 inference
    t0 = time.perf_counter()
    with torch.no_grad():
        log_probs = stage1_model(x)
        glosses = ctc_decode(log_probs, blank_id)
    s1_ms = (time.perf_counter() - t0) * 1000
    
    # Stage 2 translation
    t1 = time.perf_counter()
    clean_gloss, translation = translate_gloss_sequence(glosses, num_beams=1)
    s2_ms = (time.perf_counter() - t1) * 1000
    
    results.append({
        "clip_id": clip_id,
        "frames": T,
        "glosses": glosses,
        "clean_gloss": clean_gloss,
        "translation": translation,
        "s1_ms": s1_ms,
        "s2_ms": s2_ms,
        "total_ms": s1_ms + s2_ms
    })
    
    print(f"\nClip {idx+1}/{num_test} [{clip_id}] ({T} frames, ~{T/8:.1f}s):", flush=True)
    print(f"  Stage 1 Recognized Glosses: {glosses}", flush=True)
    print(f"  Stage 2 Translated English:  \"{translation}\"", flush=True)
    print(f"  Latency: Stage 1 = {s1_ms:.2f} ms | Stage 2 = {s2_ms:.2f} ms | Total = {s1_ms + s2_ms:.2f} ms", flush=True)

# Summary
avg_s1 = sum(r["s1_ms"] for r in results) / len(results)
avg_s2 = sum(r["s2_ms"] for r in results) / len(results)
print("\n================ BENCHMARK SUMMARY ================", flush=True)
print(f"Average Stage 1 Latency: {avg_s1:.2f} ms")
print(f"Average Stage 2 Latency: {avg_s2:.2f} ms")
print(f"Average Total Pipeline Latency: {avg_s1 + avg_s2:.2f} ms")
print(f"Total Process Resident Memory: {get_process_memory():.2f} MB")
print("===================================================", flush=True)

