import os
import sys
import json
import time
import numpy as np
import onnxruntime as ort

# Add GestureSign pipeline to sys.path
sys.path.insert(0, os.path.abspath("repos/GestureSign/pipeline/src"))

from gesturesign.features import sequence_features
import gesturesign.landmarks as lm

# Load ONNX model and meta
model_path = "repos/GestureSign/web/public/models/model.onnx"
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

with open("repos/GestureSign/web/public/models/model-meta.json", "r") as f:
    meta = json.load(f)

with open("repos/GestureSign/web/public/models/labels.json", "r") as f:
    labels_data = json.load(f)
    sign_names = [s["sign"] for s in labels_data["signs"]]

temperature = meta["calibration"]["temperature"]
threshold = meta["calibration"]["notSureBelow"]

# Load replay clips
with open("repos/GestureSign/web/public/replay/clips.json", "r") as f:
    replay_data = json.load(f)

clips = replay_data["clips"]
points_idx = replay_data["points"] # 115 points stored

print(f"Loaded {len(clips)} replay clips. Testing GestureSign ONNX model inference:\n" + "="*70)

results = []

for clip in clips:
    clip_id = clip["id"]
    ground_truth = clip["sign"]
    raw_frames = clip["frames"] # list of T frames, each has 115 points [x, y, z]
    T = len(raw_frames)

    # Reconstruct full 543x3 array for each frame:
    # 543 points: 468 face, 21 left_hand, 33 pose, 21 right_hand
    # The clip stores only the 115 selected points (points_idx)
    full_frames = np.full((T, 543, 3), np.nan, dtype=np.float32)
    for t, flat in enumerate(raw_frames):
        for i, pt_idx in enumerate(points_idx):
            x = flat[2 * i]
            y = flat[2 * i + 1]
            if x is not None and y is not None:
                full_frames[t, pt_idx, 0] = x
                full_frames[t, pt_idx, 1] = y
                full_frames[t, pt_idx, 2] = 0.0

    # Extract sequence features (hands_pose_lips)
    t0 = time.perf_counter()
    feats = sequence_features(full_frames, feature_set="hands_pose_lips")
    
    # Model expects shape [1, T, 460]
    input_tensor = feats[np.newaxis, :, :]
    
    outputs = session.run(["logits", "embedding"], {"frames": input_tensor})
    t1 = time.perf_counter()
    infer_time_ms = (t1 - t0) * 1000

    logits = outputs[0][0]
    embedding = outputs[1][0]

    # Temperature-scaled softmax
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    probs = exp_logits / exp_logits.sum()

    top3_idx = np.argsort(probs)[::-1][:3]
    top1_sign = sign_names[top3_idx[0]]
    top1_prob = probs[top3_idx[0]]

    is_correct = (top1_sign == ground_truth)
    is_top3 = ground_truth in [sign_names[i] for i in top3_idx]

    top_guesses = [(sign_names[i], probs[i]) for i in top3_idx]
    
    results.append({
        "id": clip_id,
        "true": ground_truth,
        "pred": top1_sign,
        "conf": top1_prob,
        "is_correct": is_correct,
        "is_top3": is_top3,
        "time_ms": infer_time_ms,
        "top3": top_guesses
    })

    status_icon = "CORRECT" if is_correct else ("TOP-3" if is_top3 else "MISSED")
    print(f"Clip '{ground_truth.upper()}' ({T} frames, {infer_time_ms:.2f}ms): [{status_icon}]")
    for rank, (name, prob) in enumerate(top_guesses):
        marker = " <== TRUE" if name == ground_truth else ""
        print(f"   {rank+1}. {name}: {prob*100:.1f}%{marker}")
    print("-" * 70)

# Summary
correct_count = sum(1 for r in results if r["is_correct"])
top3_count = sum(1 for r in results if r["is_top3"])
print(f"\nGestureSign Replay Test Summary:")
print(f"Top-1 Accuracy: {correct_count}/{len(clips)} ({correct_count/len(clips)*100:.1f}%)")
print(f"Top-3 Accuracy: {top3_count}/{len(clips)} ({top3_count/len(clips)*100:.1f}%)")
print(f"Average Preprocessing + Inference Time: {np.mean([r['time_ms'] for r in results]):.2f} ms")
