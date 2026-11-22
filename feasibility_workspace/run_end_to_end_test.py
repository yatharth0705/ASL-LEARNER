import os
import cv2
import time
import json
import psutil
import torch
import numpy as np
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import mediapipe as mp

def get_process_memory():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

print("=== END-TO-END COMPATIBILITY & BENCHMARK TEST ===", flush=True)

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

print("Both models loaded into memory successfully.", flush=True)

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

def translate_glosses(gloss_list, num_beams=4):
    raw_gloss_str = " ".join(gloss_list)
    # Also test notation normalization: replace IX with I/YOU, strip # from loan words, replace hyphens with spaces
    cleaned_tokens = []
    for g in gloss_list:
        clean = g.replace("#", "")
        if clean in ("IX", "IX-me"):
            clean = "I"
        elif clean == "IX-you":
            clean = "YOU"
        else:
            clean = clean.replace("-", " ")
        cleaned_tokens.append(clean)
    cleaned_gloss_str = " ".join(cleaned_tokens)
    
    # Translate raw
    inputs_raw = stage2_tokenizer(raw_gloss_str, return_tensors="pt")
    with torch.no_grad():
        out_raw = stage2_model.generate(**inputs_raw, max_length=64, num_beams=num_beams)
    text_raw = stage2_tokenizer.decode(out_raw[0], skip_special_tokens=True)
    
    # Translate cleaned
    inputs_clean = stage2_tokenizer(cleaned_gloss_str, return_tensors="pt")
    with torch.no_grad():
        out_clean = stage2_model.generate(**inputs_clean, max_length=64, num_beams=num_beams)
    text_clean = stage2_tokenizer.decode(out_clean[0], skip_special_tokens=True)
    
    return raw_gloss_str, text_raw, cleaned_gloss_str, text_clean

# PART A: Test Stage 1 Vocabulary Sequences directly into Stage 2
print("\n--- PART A: Vocabulary & Notation Compatibility Checks ---", flush=True)
vocab_sample_sequences = [
    ["YOU", "HAVE", "QUESTION"],
    ["IX", "WANT", "LEARN", "SIGN"],
    ["#IF", "YOU", "NEED", "HELP", "TELL", "ME"],
    ["TODAY", "WEATHER", "VERY", "COLD"],
    ["MY", "FRIEND", "COME", "VISIT", "TOMORROW"],
    ["MAKE-SURE", "DO", "HOMEWORK"],
    ["IX-me", "FEEL", "TIRED", "TODAY"]
]

for seq in vocab_sample_sequences:
    t0 = time.perf_counter()
    raw_str, trans_raw, clean_str, trans_clean = translate_glosses(seq)
    lat = (time.perf_counter() - t0) * 1000
    print(f"Gloss Sequence: {seq}", flush=True)
    print(f"  Raw Input:       \"{raw_str}\" -> Translation: \"{trans_raw}\"", flush=True)
    print(f"  Cleaned Input:   \"{clean_str}\" -> Translation: \"{trans_clean}\"", flush=True)
    print(f"  BART Latency:    {lat:.1f} ms\n", flush=True)

# PART B: End-to-End Pipeline on Real Continuous Webcam Video
print("\n--- PART B: Real Video End-to-End Test ---", flush=True)
video_path = r"C:\Users\YATHARTH\.gemini\antigravity\brain\f808cfd0-4627-4e26-b76e-c9c82f3ba057\.user_uploaded\media_1795114088289_bcf5b2ba.mp4"

if not os.path.exists(video_path):
    print(f"Warning: Video file not found at {video_path}")
else:
    print(f"Processing real user video: {os.path.basename(video_path)}", flush=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_s = total_frames / fps
    print(f"Video Stats: {total_frames} frames, {fps:.1f} fps, duration: {duration_s:.1f} s", flush=True)

    # Effective 8 FPS sampling (stride = round(fps / 8))
    stride = max(1, round(fps / 8.0))
    print(f"Sampling every {stride} frames (effective ~8 FPS)...", flush=True)

    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    pose_indices = [11, 12, 13, 14, 15, 16, 23, 24] # 8 upper-body landmarks

    extracted_features = []
    frame_idx = 0
    t_extract_start = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            # 8 pose landmarks (24 coords)
            pose_vec = []
            has_pose = results.pose_landmarks is not None
            if has_pose:
                lms = results.pose_landmarks.landmark
                for idx in pose_indices:
                    lm = lms[idx]
                    pose_vec.extend([lm.x, lm.y, lm.z])
            else:
                pose_vec = [0.0] * 24

            # 21 left hand landmarks (63 coords)
            lh_vec = []
            has_lh = results.left_hand_landmarks is not None
            if has_lh:
                for lm in results.left_hand_landmarks.landmark:
                    lh_vec.extend([lm.x, lm.y, lm.z])
            else:
                lh_vec = [0.0] * 63

            # 21 right hand landmarks (63 coords)
            rh_vec = []
            has_rh = results.right_hand_landmarks is not None
            if has_rh:
                for lm in results.right_hand_landmarks.landmark:
                    rh_vec.extend([lm.x, lm.y, lm.z])
            else:
                rh_vec = [0.0] * 63

            # 3 detection masks
            mask_vec = [1.0 if has_pose else 0.0, 1.0 if has_lh else 0.0, 1.0 if has_rh else 0.0]

            feat_153 = pose_vec + lh_vec + rh_vec + mask_vec
            extracted_features.append(feat_153)

        frame_idx += 1

    cap.release()
    holistic.close()
    extract_time = (time.perf_counter() - t_extract_start) * 1000
    print(f"Extracted {len(extracted_features)} feature vectors (153-D) in {extract_time:.1f} ms", flush=True)

    # Run Stage 1 CSLR
    input_tensor = torch.tensor([extracted_features], dtype=torch.float32)
    t_s1_start = time.perf_counter()
    with torch.no_grad():
        log_probs = stage1_model(input_tensor)
        recognized_glosses = ctc_decode(log_probs, blank_id)
    stage1_latency = (time.perf_counter() - t_s1_start) * 1000

    # Run Stage 2 Translation
    t_s2_start = time.perf_counter()
    if recognized_glosses:
        raw_str, trans_raw, clean_str, trans_clean = translate_glosses(recognized_glosses)
    else:
        raw_str = "<empty>"
        trans_clean = "<no sign recognized>"
        clean_str = "<empty>"
    stage2_latency = (time.perf_counter() - t_s2_start) * 1000

    total_pipeline_latency = stage1_latency + stage2_latency

    print("\n================ END-TO-END VERIFICATION RESULT ================", flush=True)
    print(f"1. Input Video:             {os.path.basename(video_path)} ({duration_s:.1f}s, {len(extracted_features)} sampled frames at 8fps)")
    print(f"2. Stage 1 Recognized Gloss: {recognized_glosses if recognized_glosses else '[]'}")
    print(f"3. Stage 2 English Output:  \"{trans_clean}\"")
    print(f"4. Stage 1 Inference Time:  {stage1_latency:.2f} ms")
    print(f"5. Stage 2 Inference Time:  {stage2_latency:.2f} ms")
    print(f"6. Total ML Latency:        {total_pipeline_latency:.2f} ms")
    print(f"7. Memory Footprint:        {get_process_memory():.2f} MB")
    print("=================================================================", flush=True)

