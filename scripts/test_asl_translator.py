import os
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn

MODEL_PATH = "repos/ASL-translator/models/sign_transformer_best.pth"
LABEL_MAP_PATH = "repos/ASL-translator/models/label_map_transformer.json"

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

class_names = label_map["classes"]
n_classes = label_map["n_classes"]
n_features = label_map["input_dim"]
seq_len = label_map["seq_len"]
d_model = label_map["d_model"]
n_heads = label_map["n_heads"]
n_layers = label_map["n_layers"]
dim_ff = label_map["dim_ff"]

print(f"ASL-translator Config: classes={n_classes}, input_dim={n_features}, seq_len={seq_len}")
print(f"Transformer dims: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, dim_ff={dim_ff}")

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
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


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

# Instantiate and load model
model = SignTransformer(n_features, d_model, n_heads, n_layers, dim_ff, n_classes)
state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Model successfully loaded! Total parameters: {total_params:,}")

# Test with a dummy sequence
dummy_seq = torch.randn(1, seq_len, n_features, dtype=torch.float32)

t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(10):
        out = model(dummy_seq)
t1 = time.perf_counter()
avg_time_ms = ((t1 - t0) / 10) * 1000

probs = torch.softmax(out, dim=1)[0]
top5_probs, top5_idx = probs.topk(5)

print(f"Inference speed: {avg_time_ms:.2f} ms per pass on CPU")
print(f"Output shape: {out.shape}")
print("Top-5 predictions on synthetic input:")
for rank, (idx, prob) in enumerate(zip(top5_idx, top5_probs)):
    print(f"  {rank+1}. {class_names[idx]} (idx {idx.item()}): {prob.item()*100:.2f}%")

