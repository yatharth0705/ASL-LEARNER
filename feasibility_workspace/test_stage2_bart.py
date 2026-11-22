import os
import time
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

print("=== STAGE 2 FEASIBILITY TEST (rrrr66254/Glossa-BART) ===", flush=True)
mem_before = get_process_memory()
t0 = time.perf_counter()

stage2_dir = os.path.abspath("feasibility_workspace/stage2_bart")
tokenizer = AutoTokenizer.from_pretrained(stage2_dir, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(stage2_dir, local_files_only=True)
model.eval()

load_time = time.perf_counter() - t0
mem_after = get_process_memory()
total_params = sum(p.numel() for p in model.parameters())

print(f"Model loaded in: {load_time:.2f} s", flush=True)
print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)", flush=True)
print(f"RAM increase during load: {mem_after - mem_before:.2f} MB (Total process RAM: {mem_after:.2f} MB)", flush=True)

# Tokenizer inspection
print("\n--- Tokenizer Inspection ---", flush=True)
print(f"Vocab size: {tokenizer.vocab_size}", flush=True)
print(f"BOS: {tokenizer.bos_token} (id={tokenizer.bos_token_id})", flush=True)
print(f"EOS: {tokenizer.eos_token} (id={tokenizer.eos_token_id})", flush=True)
print(f"PAD: {tokenizer.pad_token} (id={tokenizer.pad_token_id})", flush=True)
print(f"UNK: {tokenizer.unk_token} (id={tokenizer.unk_token_id})", flush=True)

# Test cases
test_glosses = [
    "YOU GO STORE TOMORROW?",
    "I FINISH READ BOOK YESTERDAY",
    "WANT LEARN SIGN LANGUAGE",
    "WEATHER TODAY RAIN COLD",
    "TEACHER EXPLAIN MATH CONCEPT",
    "CAN YOU HELP ME PLEASE",
    "MY NAME JOHN",
]

print("\n--- Translation Tests on CPU (Greedy vs Beam Search) ---", flush=True)
latencies_greedy = []
latencies_beam = []

for gloss in test_glosses:
    inputs = tokenizer(gloss, return_tensors="pt")
    
    # Greedy search
    t_start = time.perf_counter()
    with torch.no_grad():
        out_greedy = model.generate(**inputs, max_length=64, num_beams=1)
    lat_g = (time.perf_counter() - t_start) * 1000
    latencies_greedy.append(lat_g)
    text_greedy = tokenizer.decode(out_greedy[0], skip_special_tokens=True)
    
    # Beam search (num_beams=4)
    t_start = time.perf_counter()
    with torch.no_grad():
        out_beam = model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
    lat_b = (time.perf_counter() - t_start) * 1000
    latencies_beam.append(lat_b)
    text_beam = tokenizer.decode(out_beam[0], skip_special_tokens=True)
    
    print(f"Gloss:        \"{gloss}\"", flush=True)
    print(f"  Greedy:     \"{text_greedy}\" ({lat_g:.1f} ms)", flush=True)
    print(f"  Beam(4):    \"{text_beam}\" ({lat_b:.1f} ms)", flush=True)

print(f"\nAverage Greedy Latency: {sum(latencies_greedy)/len(latencies_greedy):.1f} ms", flush=True)
print(f"Average Beam(4) Latency: {sum(latencies_beam)/len(latencies_beam):.1f} ms", flush=True)
print(f"Final Process Memory: {get_process_memory():.2f} MB", flush=True)
