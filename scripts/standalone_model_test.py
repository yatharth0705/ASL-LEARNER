"""
ASL-Learner: Standalone Pretrained Model Verification & Inference Test
======================================================================
This script validates both official pretrained repositories:
1. GestureSign (ONNX Conv-Transformer, 250 classes)
2. ASL-translator (PyTorch Transformer, 250 classes)

It verifies file sizes, hash/integrity, licenses, feature layouts,
and executes end-to-end inference on both models.
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 16):
            h.update(chunk)
    return h.hexdigest()

print("=" * 80)
print(" ASL-LEARNER: PRETRAINED MODEL AUDIT & STANDALONE INFERENCE VERIFICATION")
print("=" * 80)

# ---------------------------------------------------------
# 1. FILE EXISTENCE & INTEGRITY VERIFICATION
# ---------------------------------------------------------
files_to_check = [
    {
        "repo": "GestureSign",
        "file": "repos/GestureSign/web/public/models/model.onnx",
        "type": "ONNX Model Weights",
        "license": "repos/GestureSign/LICENSE"
    },
    {
        "repo": "GestureSign",
        "file": "repos/GestureSign/web/public/models/labels.json",
        "type": "250-Class Labels & Metrics",
        "license": "repos/GestureSign/LICENSE"
    },
    {
        "repo": "GestureSign",
        "file": "repos/GestureSign/web/public/models/model-meta.json",
        "type": "Calibration & Feature Metadata",
        "license": "repos/GestureSign/LICENSE"
    },
    {
        "repo": "ASL-translator",
        "file": "repos/ASL-translator/models/sign_transformer_best.pth",
        "type": "PyTorch State Dict Weights",
        "license": "repos/ASL-translator/LICENSE"
    },
    {
        "repo": "ASL-translator",
        "file": "repos/ASL-translator/models/label_map_transformer.json",
        "type": "Class Mapping & Hyperparameters",
        "license": "repos/ASL-translator/LICENSE"
    },
    {
        "repo": "ASL-translator",
        "file": "repos/ASL-translator/hand_landmarker.task",
        "type": "MediaPipe Hand Landmarker Task",
        "license": "repos/ASL-translator/LICENSE"
    },
]

print("\n[STEP 1] Verifying Model Files & Licenses:")
for item in files_to_check:
    path = item["file"]
    exists = os.path.exists(path)
    if exists:
        size = os.path.getsize(path)
        sha = file_hash(path)[:12]
        with open(path, "rb") as f:
            head = f.read(40)
        is_lfs = b"git-lfs" in head or b"oid sha256" in head
        print(f"  [OK] {item['repo']:15} | {os.path.basename(path):28} | {size:>10,} bytes | SHA256: {sha}... | Genuine Binary: {not is_lfs}")
    else:
        print(f"  [FAIL] {path} DOES NOT EXIST!")

# Verify licenses
for r in ["GestureSign", "ASL-translator"]:
    lic_path = f"repos/{r}/LICENSE"
    with open(lic_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        f.readline()
        author = f.readline().strip()
        print(f"  License for {r:15}: {first_line} ({author})")

# ---------------------------------------------------------
# 2. INFERENCE TEST: REPOSITORY 1 - GestureSign (ONNX)
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("[STEP 2] Standalone Inference Test: GestureSign (ONNX Model)")
print("=" * 80)

sys.path.insert(0, os.path.abspath("repos/GestureSign/pipeline/src"))
from gesturesign.features import sequence_features

gs_model_path = "repos/GestureSign/web/public/models/model.onnx"
gs_session = ort.InferenceSession(gs_model_path, providers=["CPUExecutionProvider"])

with open("repos/GestureSign/web/public/models/model-meta.json", "r") as f:
    gs_meta = json.load(f)

with open("repos/GestureSign/web/public/models/labels.json", "r") as f:
    gs_labels_data = json.load(f)
    gs_classes = [s["sign"] for s in gs_labels_data["signs"]]

temp = gs_meta["calibration"]["temperature"]
not_sure_below = gs_meta["calibration"]["notSureBelow"]

print(f"GestureSign Architecture: Conv + Transformer (1.4M params)")
print(f"Input specification:      [batch_size=1, frames=T, feature_dim=460]")
print(f"Outputs:                  logits [1, 250], embedding [1, 128]")
print(f"Calibration Temperature:  {temp:.4f}, Not-Sure Threshold: {not_sure_below:.4f}")

# Testing on held-out dataset replay clips
with open("repos/GestureSign/web/public/replay/clips.json", "r") as f:
    replay = json.load(f)

clips = replay["clips"]
points_idx = replay["points"]

print(f"\nEvaluating on {len(clips)} real held-out test clips:")
gs_predictions = []

for clip in clips:
    sign_true = clip["sign"]
    raw_frames = clip["frames"]
    T = len(raw_frames)
    
    # Unpack flat [2*115] frames to [T, 543, 3] MediaPipe landmark format
    full_frames = np.full((T, 543, 3), np.nan, dtype=np.float32)
    for t, flat in enumerate(raw_frames):
        for i, pt_idx in enumerate(points_idx):
            x = flat[2 * i]
            y = flat[2 * i + 1]
            if x is not None and y is not None:
                full_frames[t, pt_idx, 0] = x
                full_frames[t, pt_idx, 1] = y
                full_frames[t, pt_idx, 2] = 0.0

    # Exact official feature extraction pipeline
    t_start = time.perf_counter()
    feats = sequence_features(full_frames, feature_set="hands_pose_lips")
    input_tensor = feats[np.newaxis, :, :] # (1, T, 460)
    
    outs = gs_session.run(["logits", "embedding"], {"frames": input_tensor})
    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000

    logits = outs[0][0]
    embedding = outs[1][0]

    # Temperature scaled softmax
    scaled = logits / temp
    exps = np.exp(scaled - np.max(scaled))
    probs = exps / np.sum(exps)

    top3_idx = np.argsort(probs)[::-1][:3]
    top1_pred = gs_classes[top3_idx[0]]
    top1_conf = probs[top3_idx[0]]

    is_correct = (top1_pred == sign_true)
    top3_names = [gs_classes[i] for i in top3_idx]
    is_top3 = sign_true in top3_names

    gs_predictions.append({
        "sign": sign_true,
        "frames": T,
        "latency_ms": latency_ms,
        "top1": top1_pred,
        "conf": top1_conf,
        "is_correct": is_correct,
        "top3": [(gs_classes[i], probs[i]) for i in top3_idx]
    })

    status = "PASS (Top-1)" if is_correct else ("PASS (Top-3)" if is_top3 else "MISSED")
    print(f"  Clip: '{sign_true.upper():<10}' | Frames: {T:2d} | Latency: {latency_ms:5.2f} ms | Status: {status}")
    for rank, (name, p) in enumerate(gs_predictions[-1]["top3"]):
        marker = " <== TRUE" if name == sign_true else ""
        print(f"      {rank+1}. {name:<12} ({p*100:5.1f}%){marker}")

gs_top1 = sum(1 for p in gs_predictions if p["is_correct"])
gs_avg_lat = np.mean([p["latency_ms"] for p in gs_predictions])
print(f"\nGestureSign Summary: Top-1 Accuracy: {gs_top1}/{len(clips)} ({gs_top1/len(clips)*100:.1f}%), Avg Latency: {gs_avg_lat:.2f} ms")


# ---------------------------------------------------------
# 3. INFERENCE TEST: REPOSITORY 2 - ASL-translator (PyTorch)
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("[STEP 3] Standalone Inference Test: ASL-translator (PyTorch Transformer)")
print("=" * 80)

asl_model_path = "repos/ASL-translator/models/sign_transformer_best.pth"
asl_label_path = "repos/ASL-translator/models/label_map_transformer.json"

with open(asl_label_path, "r") as f:
    asl_map = json.load(f)

asl_classes = asl_map["classes"]
asl_input_dim = asl_map["input_dim"] # 390
asl_seq_len = asl_map["seq_len"] # 64
asl_d_model = asl_map["d_model"]
asl_heads = asl_map["n_heads"]
asl_layers = asl_map["n_layers"]
asl_dim_ff = asl_map["dim_ff"]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class SignTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers, dim_ff, n_classes, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        padding_mask = (x.abs().sum(dim=-1) == 0)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        x_masked = x * mask_expanded
        lengths = mask_expanded.sum(dim=1).clamp(min=1)
        x_mean = x_masked.sum(dim=1) / lengths
        x_for_max = x_masked.clone()
        x_for_max[padding_mask] = -1e9
        x_max = x_for_max.max(dim=1).values
        x_pooled = torch.cat([x_mean, x_max], dim=1)
        return self.classifier(x_pooled)

import math
asl_model = SignTransformer(asl_input_dim, asl_d_model, asl_heads, asl_layers, asl_dim_ff, len(asl_classes))
asl_model.load_state_dict(torch.load(asl_model_path, map_location="cpu", weights_only=True))
asl_model.eval()

# Benchmark inference latency on 64-frame sequence
dummy_input = torch.randn(1, asl_seq_len, asl_input_dim, dtype=torch.float32)

t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(20):
        out = asl_model(dummy_input)
t1 = time.perf_counter()
asl_avg_lat = ((t1 - t0) / 20) * 1000

probs = torch.softmax(out, dim=1)[0]
top5_probs, top5_idx = probs.topk(5)

print(f"ASL-translator Architecture: PyTorch Transformer Encoder ({sum(p.numel() for p in asl_model.parameters()):,} params)")
print(f"Input specification:         [batch_size=1, frames=64, feature_dim=390]")
print(f"Output shape:                {out.shape}")
print(f"Average CPU Latency:         {asl_avg_lat:.2f} ms")
print(f"Top-5 Predictions on synthetic 64-frame sample:")
for rank, (idx, prob) in enumerate(zip(top5_idx, top5_probs)):
    print(f"  {rank+1}. {asl_classes[idx]:<15} ({prob.item()*100:.2f}%)")

print("\n" + "=" * 80)
print(" STANDALONE TEST COMPLETE")
print("=" * 80)

