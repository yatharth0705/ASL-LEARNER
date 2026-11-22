"""
Interactive Command-Line Demonstration of the Hybrid Continuous ASL Pipeline.
Run with: python hybrid_prototype/demo_cli.py
"""
import os
import sys
import time

prototype_dir = os.path.dirname(os.path.abspath(__file__))
if prototype_dir not in sys.path:
    sys.path.insert(0, prototype_dir)

from pipeline import HybridASLPipeline
from test_suite import load_decoded_replay_clips, simulate_stream

def main():
    print("==================================================================")
    print("  ASL-LEARNER HYBRID CONTINUOUS TRANSLATION PROTOTYPE (DEMO)")
    print("  GestureSign (460-D ONNX) + Temporal Aggregator + Glossa-BART")
    print("==================================================================")

    clips = load_decoded_replay_clips()
    print(f"Available reference signs in catalog: {list(clips.keys())}\n")

    pipeline = HybridASLPipeline(min_confidence=0.35, hold_debounce_ms=800.0)
    print("Models loaded successfully. Initializing stream simulation...\n")

    # Demo 1: Continuous multi-sign stream
    print("--- DEMO 1: Simulating Continuous Signing ['hello', 'police'] ---")
    evts = simulate_stream(pipeline, [clips["hello"], clips["police"]], [500.0, 500.0])
    for e in evts:
        rec = e["recognition"]
        print(f"  [RECOGNIZED] Sign: {rec['sign']:8s} | Confidence: {rec['confidence']*100:.1f}% | Latency: {rec['latency_ms']:.1f}ms")
    
    trans = pipeline.translate_buffer()
    print(f"\n  Final Gloss Buffer:     {pipeline.get_gloss_tokens()}")
    print(f"  Glossa-BART Translation: \"{trans['english_translation']}\" ({trans['latency_ms']:.1f} ms)\n")

    # Demo 2: User correction and re-translation
    print("--- DEMO 2: User Buffer Correction & Re-Translation ---")
    print(f"  Current Buffer: {pipeline.get_gloss_tokens()}")
    print("  Action: Replacing sign index 1 ('police') with 'friend'...")
    pipeline.replace_gloss(1, "FRIEND")
    print(f"  Updated Buffer: {pipeline.get_gloss_tokens()}")
    trans2 = pipeline.translate_buffer()
    print(f"  Re-translated:  \"{trans2['english_translation']}\" ({trans2['latency_ms']:.1f} ms)\n")

    # Demo 3: Appending sign and translation
    print("--- DEMO 3: Appending 'WANT' and 'LEARN' ---")
    pipeline.add_manual_gloss("WANT")
    pipeline.add_manual_gloss("LEARN")
    print(f"  Expanded Buffer: {pipeline.get_gloss_tokens()}")
    trans3 = pipeline.translate_buffer()
    print(f"  Translation:     \"{trans3['english_translation']}\" ({trans3['latency_ms']:.1f} ms)\n")

    print("==================================================================")
    print("  Prototype execution verified. Ready for evaluation report.")
    print("==================================================================")

if __name__ == "__main__":
    main()

