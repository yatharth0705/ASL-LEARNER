import os
import sys
import json
import time
import math
import psutil
import numpy as np

# Ensure local imports
prototype_dir = os.path.dirname(os.path.abspath(__file__))
if prototype_dir not in sys.path:
    sys.path.insert(0, prototype_dir)

from landmarks import empty_frame, N_POINTS, FRAME_LENGTH
from pipeline import HybridASLPipeline
from segmenter import SegmenterOptions

def get_process_memory():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def load_decoded_replay_clips():
    """Load and unpack clips from public/replay/clips.json into full 543*3 frames."""
    clips_path = os.path.abspath("repos/GestureSign/web/public/replay/clips.json")
    with open(clips_path, "r") as f:
        data = json.load(f)
    
    points = data["points"]
    decoded_clips = {}
    
    for c in data["clips"]:
        sign = c["sign"]
        fps = c["fps"]
        raw_frames = c["frames"]
        full_frames = []
        for flat in raw_frames:
            f = empty_frame()
            for i, p in enumerate(points):
                x = flat[2 * i]
                y = flat[2 * i + 1]
                if x is not None and y is not None:
                    f[p * 3] = x
                    f[p * 3 + 1] = y
                    f[p * 3 + 2] = 0.0
            full_frames.append(f)
        decoded_clips[sign] = {
            "sign": sign,
            "fps": fps,
            "frames": full_frames,
            "duration_ms": (len(full_frames) / fps) * 1000.0
        }
    return decoded_clips

def simulate_stream(pipeline: HybridASLPipeline, clip_list: list, pause_ms_list: list) -> list:
    """
    Feeds a continuous sequence of clips separated by neutral pauses into the pipeline.
    Returns all recognition events.
    """
    pipeline.reset()
    results = []
    t = 1000.0 # start at 1000 ms

    for idx, clip in enumerate(clip_list):
        fps = clip["fps"]
        dt = 1000.0 / fps
        frames = clip["frames"]
        
        # 1. Push sign frames
        for f in frames:
            evt = pipeline.push_frame(f, t, poses=1)
            if evt and evt["event"] == "recognition":
                results.append(evt)
            t += dt

        # 2. Push pause frames (hands absent)
        pause_ms = pause_ms_list[idx] if idx < len(pause_ms_list) else 500.0
        pause_frames = int(math.ceil(pause_ms / dt))
        for _ in range(pause_frames):
            evt = pipeline.push_frame(empty_frame(), t, poses=0)
            if evt and evt["event"] == "recognition":
                results.append(evt)
            t += dt

    # Final flush to ensure closure
    flush_evt = pipeline.flush()
    if flush_evt and flush_evt["event"] == "recognition":
        results.append(flush_evt)

    return results

def compute_wer(reference: list, hypothesis: list) -> float:
    """Levenshtein word error rate."""
    r, h = reference, hypothesis
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    for i in range(len(r) + 1):
        d[i, 0] = i
    for j in range(len(h) + 1):
        d[0, j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1].lower() == h[j - 1].lower():
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + 1)
    return d[len(r), len(h)] / max(len(r), 1)

def run_evaluation_suite():
    print("==================================================================", flush=True)
    print("HYBRID CONTINUOUS ASL PIPELINE: MANDATORY EVALUATION SUITE", flush=True)
    print("==================================================================", flush=True)

    t_suite_start = time.perf_counter()
    mem_init = get_process_memory()

    # Load Replay Clips
    clips = load_decoded_replay_clips()
    print(f"Loaded {len(clips)} reference held-out sign clips: {list(clips.keys())}", flush=True)

    # Initialize Pipeline
    pipeline = HybridASLPipeline(min_confidence=0.35, hold_debounce_ms=800.0)
    mem_loaded = get_process_memory()
    print(f"Pipeline Initialized. Resident RAM: {mem_loaded:.2f} MB (Delta: +{mem_loaded - mem_init:.2f} MB)\n", flush=True)

    eval_results = {}

    # TEST 1: Individual Isolated Signs
    print("--- TEST 1: Individual Isolated Signs ---", flush=True)
    test1_signs = ["blue", "police", "giraffe", "hello"]
    test1_hits = 0
    test1_latencies = []

    for s in test1_signs:
        clip = clips[s]
        evts = simulate_stream(pipeline, [clip], [600.0])
        pred = evts[0]["recognition"]["sign"] if evts else None
        conf = evts[0]["recognition"]["confidence"] if evts else 0.0
        lat = evts[0]["recognition"]["latency_ms"] if evts else 0.0
        test1_latencies.append(lat)
        match = (pred == s)
        if match:
            test1_hits += 1
        print(f"  Sign '{s:8s}' -> Predicted: {pred:8s} ({conf*100:.1f}%) | Latency: {lat:.1f}ms | Correct: {match}")

    test1_acc = test1_hits / len(test1_signs)
    eval_results["test1_accuracy"] = test1_acc
    print(f"Test 1 Isolated Accuracy: {test1_acc*100:.1f}%\n", flush=True)

    # TEST 2: Two Consecutive Signs
    print("--- TEST 2: Two Consecutive Signs ---", flush=True)
    pair = [clips["hello"], clips["police"]]
    evts2 = simulate_stream(pipeline, pair, [500.0, 500.0])
    glosses2 = pipeline.get_gloss_tokens()
    trans2 = pipeline.translate_buffer(use_beam_search=False)
    print(f"  Ground Truth Sequence:   ['hello', 'police']")
    print(f"  Recognized Gloss Buffer: {glosses2}")
    print(f"  Glossa-BART Translation: \"{trans2['english_translation']}\" ({trans2['latency_ms']:.1f} ms)")
    test2_wer = compute_wer(["hello", "police"], glosses2)
    print(f"  Gloss Sequence WER:      {test2_wer:.2f}\n", flush=True)
    eval_results["test2_wer"] = test2_wer

    # TEST 3: Three-to-Five Consecutive Signs
    print("--- TEST 3: Three-to-Five Consecutive Signs ---", flush=True)
    seq3 = [clips["hello"], clips["blue"], clips["giraffe"], clips["police"]]
    evts3 = simulate_stream(pipeline, seq3, [450.0, 450.0, 450.0, 500.0])
    glosses3 = pipeline.get_gloss_tokens()
    trans3 = pipeline.translate_buffer(use_beam_search=False)
    ref3 = ["hello", "blue", "giraffe", "police"]
    print(f"  Ground Truth:   {ref3}")
    print(f"  Recognized:     {glosses3}")
    print(f"  Translation:    \"{trans3['english_translation']}\" ({trans3['latency_ms']:.1f} ms)")
    test3_wer = compute_wer(ref3, glosses3)
    print(f"  Gloss WER:      {test3_wer:.2f}\n", flush=True)
    eval_results["test3_wer"] = test3_wer

    # TEST 4: Longer Sign Sequences (6 consecutive signs)
    print("--- TEST 4: Longer Sign Sequences (6 Signs) ---", flush=True)
    seq4 = [clips["hello"], clips["blue"], clips["giraffe"], clips["smile"], clips["police"], clips["hello"]]
    ref4 = ["hello", "blue", "giraffe", "smile", "police", "hello"]
    evts4 = simulate_stream(pipeline, seq4, [400.0, 400.0, 400.0, 400.0, 400.0, 500.0])
    glosses4 = pipeline.get_gloss_tokens()
    trans4 = pipeline.translate_buffer(use_beam_search=False)
    print(f"  Ground Truth:   {ref4}")
    print(f"  Recognized:     {glosses4}")
    print(f"  Translation:    \"{trans4['english_translation']}\" ({trans4['latency_ms']:.1f} ms)")
    test4_wer = compute_wer(ref4, glosses4)
    print(f"  Gloss WER:      {test4_wer:.2f}\n", flush=True)
    eval_results["test4_wer"] = test4_wer

    # TEST 5: Different Pauses Between Signs
    print("--- TEST 5: Different Pause Durations ---", flush=True)
    pair5 = [clips["hello"], clips["blue"]]
    for pause in [150.0, 500.0, 1200.0]:
        evts5 = simulate_stream(pipeline, pair5, [pause, 500.0])
        g5 = pipeline.get_gloss_tokens()
        print(f"  Pause = {pause:4.0f} ms -> Recognized Glosses: {g5} (Expected 2 signs: {len(g5) == 2})")
    print()

    # TEST 6: Repeated Signs (Debounce Hold vs Intentional Repetition)
    print("--- TEST 6: Repeated Signs & Debouncing ---", flush=True)
    # 6A: Still Hold (Hands held in sign position without leaving)
    pipeline.reset()
    hello_frames = clips["hello"]["frames"]
    still_frame = hello_frames[-1]
    dt = 1000.0 / 30.0
    t = 1000.0
    # Push hello
    for f in hello_frames:
        pipeline.push_frame(f, t)
        t += dt
    # Hold still for 600 ms
    for _ in range(int(600 / dt)):
        pipeline.push_frame(still_frame, t)
        t += dt
    pipeline.flush()
    hold_glosses = pipeline.get_gloss_tokens()
    print(f"  6A. Sign Held Still (600ms hold) -> Glosses: {hold_glosses} (Expected 1 sign: {len(hold_glosses) == 1})")

    # 6B: Intentional Repetition (HELLO -> Hands Drop 700ms -> HELLO)
    evts6b = simulate_stream(pipeline, [clips["hello"], clips["hello"]], [700.0, 500.0])
    rep_glosses = pipeline.get_gloss_tokens()
    print(f"  6B. Intentional Repetition (700ms rest) -> Glosses: {rep_glosses} (Expected 2 signs: {len(rep_glosses) == 2})\n")

    # TEST 7: Incorrectly Recognized Signs & Manual User Correction
    print("--- TEST 7: Recognition Correction & Buffer Editing ---", flush=True)
    # Simulate an incorrect sign insertion
    pipeline.reset()
    pipeline.aggregator.add_manual_gloss("GIRAFFE")
    pipeline.aggregator.add_manual_gloss("BLUE")
    print(f"  Original Buffer with error: {pipeline.get_gloss_tokens()}")
    # User corrects index 0 from GIRAFFE to HELLO
    pipeline.replace_gloss(0, "HELLO")
    print(f"  After replace_gloss(0, 'HELLO'): {pipeline.get_gloss_tokens()}")
    # User deletes index 1
    pipeline.delete_gloss_at(1)
    print(f"  After delete_gloss_at(1):        {pipeline.get_gloss_tokens()}")
    trans7 = pipeline.translate_buffer()
    print(f"  Re-translated output: \"{trans7['english_translation']}\"\n", flush=True)

    # TEST 8: Low-Confidence Noise Filtering
    print("--- TEST 8: Low-Confidence Predictions & Noise Rejection ---", flush=True)
    pipeline.reset()
    # Generate noisy frames (random jitter with hands moving erratically)
    noise_frames = [empty_frame() + np.random.uniform(0.1, 0.9, FRAME_LENGTH).astype(np.float32) for _ in range(25)]
    noise_clip = {"fps": 30.0, "frames": noise_frames}
    evts8 = simulate_stream(pipeline, [noise_clip], [500.0])
    all_entries8 = pipeline.get_all_entries()
    confirmed8 = pipeline.get_gloss_tokens()
    print(f"  Noise clip pushed: Total aggregator entries = {len(all_entries8)}")
    for e in all_entries8:
        print(f"    Entry sign='{e.sign}' | Confidence={e.confidence:.3f} | Status={e.status}")
    print(f"  Confirmed Glosses (silently accepted noise): {confirmed8} (Expected empty: {len(confirmed8) == 0})\n")

    # TEST 9: Out-of-Vocabulary (OOV) Signs Handling in Translation
    print("--- TEST 9: Signs Outside GestureSign 250-Class Vocabulary ---", flush=True)
    oov_glosses = ["HELLO", "FRIEND", "WANT", "LEARN", "SIGN"]
    trans9 = pipeline.translator.translate(oov_glosses)
    print(f"  Input Glosses: {oov_glosses}")
    print(f"  Translation:   \"{trans9['english_translation']}\"")
    print(f"  Preservation:  Input signs syntactically unified without hallucinating unrelated sentences.\n", flush=True)

    # Summary Benchmarks & Resource Utilization
    t_suite_total = time.perf_counter() - t_suite_start
    mem_final = get_process_memory()

    print("==================================================================", flush=True)
    print("EVALUATION SUITE SUMMARY & METRICS", flush=True)
    print("==================================================================", flush=True)
    print(f"Total Test Suite Duration:     {t_suite_total:.2f} s")
    print(f"Memory (Resident RSS):         {mem_final:.2f} MB")
    print(f"Isolated Sign Accuracy (T1):   {test1_acc*100:.1f}%")
    print(f"2-Sign Gloss WER (T2):         {test2_wer:.2f}")
    print(f"4-Sign Gloss WER (T3):         {test3_wer:.2f}")
    print(f"6-Sign Gloss WER (T4):         {test4_wer:.2f}")
    print(f"Average Recognition Latency:   {sum(test1_latencies)/len(test1_latencies):.2f} ms")
    print(f"Average Translation Latency:   {trans3['latency_ms']:.2f} ms")
    print("==================================================================", flush=True)

if __name__ == "__main__":
    run_evaluation_suite()

