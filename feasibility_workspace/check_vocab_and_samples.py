import os
import json
import time
import psutil
import torch
import numpy as np
import pandas as pd
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Inspect Stage 1 Vocab and token types
stage1_vocab_path = "feasibility_workspace/stage1_cslr/vocab.json"
with open(stage1_vocab_path, "r") as f:
    vdata = json.load(f)
token_to_id = vdata.get("token_to_id", vdata)
id_to_token = {v: k for k, v in token_to_id.items()}
blank_id = token_to_id["<blank>"]
pad_id = token_to_id.get("<pad>", 0)

print(f"=== STAGE 1 VOCABULARY ANALYSIS ===")
print(f"Total vocabulary size: {len(token_to_id)}")
print(f"Blank ID: {blank_id}, Pad ID: {pad_id}")

tokens = list(token_to_id.keys())
hash_tokens = [t for t in tokens if t.startswith("#")]
ix_tokens = [t for t in tokens if "IX" in t]
plus_tokens = [t for t in tokens if "+" in t]
dash_tokens = [t for t in tokens if "-" in t]
print(f"Loan / fingerspelled tokens (#): {len(hash_tokens)} (e.g. {hash_tokens[:5]})")
print(f"Index / pointing tokens (IX):   {len(ix_tokens)} (e.g. {ix_tokens[:5]})")
print(f"Compound tokens (+):            {len(plus_tokens)} (e.g. {plus_tokens[:5]})")
print(f"Inflected tokens (-):           {len(dash_tokens)} (e.g. {dash_tokens[:5]})")

# 2. Inspect How2Sign sample from parquet
parquet_path = r"C:\Users\YATHARTH\.cache\huggingface\hub\datasets--ASL-Live-Video-Translation--how2sign-realigned-mediapipe\snapshots\598622fc802aae9dbfdf990b5b287aa7d99fd241\parquet\mediapipe\val-00000.parquet"
df = pd.read_parquet(parquet_path)
print(f"\nLoaded validation parquet with {len(df)} sentence clips.")

# Inspect first 3 clips
for i in range(3):
    row = df.iloc[i]
    pts = row['points']
    det = row['detected']
    print(f"Clip {i} ({row['sentence_id']}): {len(pts)} frames. Frame 0 pts count: {len(pts[0])}, det: {det[0]}")

