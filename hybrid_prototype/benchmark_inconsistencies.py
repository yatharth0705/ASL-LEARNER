import os
import time
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_process_memory():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

print("=== BENCHMARK INCONSISTENCY RESOLUTION & ANALYSIS ===", flush=True)

stage2_dir = os.path.abspath("feasibility_workspace/stage2_bart")
mem_start = get_process_memory()
t0 = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(stage2_dir, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(stage2_dir, local_files_only=True)
model.eval()

load_time = time.perf_counter() - t0
mem_after_load = get_process_memory()

print(f"Model Load Time: {load_time:.2f} s", flush=True)
print(f"Process Memory after load: {mem_after_load:.2f} MB (Delta: +{mem_after_load - mem_start:.2f} MB)", flush=True)

# Warmup run
dummy_input = tokenizer("HELLO", return_tensors="pt")
with torch.no_grad():
    _ = model.generate(**dummy_input, max_length=10, num_beams=1)
print("Warm-up pass completed.", flush=True)

# 1. Direct Reproduction of Section F Scenario (Single-word input "YOU" -> "You ?")
print("\n--- SCENARIO 1: Section F Conditions (Single Word Input 'YOU') ---", flush=True)
times_single = []
input_single = tokenizer("YOU", return_tensors="pt")
for _ in range(10):
    t_start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**input_single, max_length=64, num_beams=1)
    times_single.append((time.perf_counter() - t_start) * 1000)

decoded_single = tokenizer.decode(out[0], skip_special_tokens=True)
num_tokens_single = len(out[0])
print(f"Input: 'YOU' -> Generated: '{decoded_single}'")
print(f"Output Token Count (including BOS/EOS): {num_tokens_single} tokens")
print(f"Average Latency (10 runs): {sum(times_single)/len(times_single):.2f} ms (Min: {min(times_single):.2f} ms, Max: {max(times_single):.2f} ms)")

# 2. Direct Reproduction of Section D Scenario (7 Multi-word Sentences)
print("\n--- SCENARIO 2: Section D Conditions (7 Multi-Word Sentences) ---", flush=True)
section_d_sentences = [
    "YOU GO STORE TOMORROW?",
    "I FINISH READ BOOK YESTERDAY",
    "WANT LEARN SIGN LANGUAGE",
    "WEATHER TODAY RAIN COLD",
    "TEACHER EXPLAIN MATH CONCEPT",
    "CAN YOU HELP ME PLEASE",
    "MY NAME JOHN",
]

times_multi = []
token_counts_multi = []
for sent in section_d_sentences:
    inp = tokenizer(sent, return_tensors="pt")
    t_start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inp, max_length=64, num_beams=1)
    lat = (time.perf_counter() - t_start) * 1000
    times_multi.append(lat)
    token_counts_multi.append(len(out[0]))
    dec = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"  Input: '{sent}' -> Output: '{dec}' ({len(out[0])} tokens) -> {lat:.1f} ms")

print(f"\nAverage Latency across Section D sentences: {sum(times_multi)/len(times_multi):.2f} ms")
print(f"Average Output Token Count: {sum(token_counts_multi)/len(token_counts_multi):.1f} tokens")

# 3. Systematic Scaling Analysis: Latency vs Generated Token Count
print("\n--- SYSTEMATIC ANALYSIS: Latency vs Number of Generated Tokens ---", flush=True)
target_prompts = [
    ("YOU", "Short (2 output tokens)"),
    ("YES", "Very short"),
    ("MY NAME JOHN", "Medium (6 output tokens)"),
    ("I FINISH READ BOOK YESTERDAY", "Medium-Long (7 output tokens)"),
    ("WEATHER TODAY RAIN COLD", "Long (8 output tokens)"),
    ("YOU GO STORE TOMORROW?", "Long (9 output tokens)"),
    ("WANT LEARN SIGN LANGUAGE", "Very Long (10 output tokens)")
]

scaling_results = []
for prompt, desc in target_prompts:
    inp = tokenizer(prompt, return_tensors="pt")
    latencies = []
    for _ in range(5):
        t_start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inp, max_length=64, num_beams=1)
        latencies.append((time.perf_counter() - t_start) * 1000)
    avg_l = sum(latencies) / len(latencies)
    tok_count = len(out[0])
    dec_text = tokenizer.decode(out[0], skip_special_tokens=True)
    scaling_results.append({
        "prompt": prompt,
        "desc": desc,
        "tokens": tok_count,
        "latency_ms": avg_l,
        "output": dec_text
    })
    print(f"Tokens: {tok_count:2d} | Avg Latency: {avg_l:6.2f} ms | Prompt: '{prompt:30s}' -> '{dec_text}'")

# Compute per-token step time:
# Linear regression: Latency = T_prefill + N * T_step
x_vals = [r["tokens"] for r in scaling_results]
y_vals = [r["latency_ms"] for r in scaling_results]
n = len(x_vals)
slope = (n * sum(x*y for x, y in zip(x_vals, y_vals)) - sum(x_vals) * sum(y_vals)) / (n * sum(x**2 for x in x_vals) - (sum(x_vals))**2)
intercept = (sum(y_vals) - slope * sum(x_vals)) / n

print("\n--- MATHEMATICAL CONCLUSION ---")
print(f"Encoder Prefill + Initial Step Time (Intercept): {intercept:.2f} ms")
print(f"Marginal Cost per Generated Token (Slope):        {slope:.2f} ms / token")
print(f"Model Formula: Latency ≈ {intercept:.1f} ms + ({slope:.1f} ms × Output Tokens)")
print(f"For 2 tokens ('YOU' -> 'You ?'):       Expected = {intercept + 2*slope:.1f} ms (Section F measured ~50 ms)")
print(f"For 8.6 tokens (Section D average):     Expected = {intercept + 8.6*slope:.1f} ms (Section D measured ~221 ms)")
print("=====================================================", flush=True)

