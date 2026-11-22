import os
import math
import torch
import torch.nn as nn
from safetensors.torch import load_file

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 256, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class PoseTransformerCSLR(nn.Module):
    def __init__(self, input_dim=153, d_model=256, n_heads=4, ff_dim=1024, n_layers=4, vocab_size=1354, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            norm_first=True,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, src_key_padding_mask=None):
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        logits = self.head(h)
        return logits.log_softmax(dim=-1)

model = PoseTransformerCSLR()
weights = load_file("feasibility_workspace/stage1_cslr/model.safetensors")
missing, unexpected = model.load_state_dict(weights, strict=True)
print("=== STAGE 1 WEIGHT LOAD TEST ===")
print("Model loaded with strict=True!")
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

# Test forward pass with dummy tensor (B=1, T=40, C=153)
dummy_x = torch.randn(1, 40, 153)
with torch.no_grad():
    out = model(dummy_x)
print(f"Forward pass output shape: {out.shape}")
assert out.shape == (1, 40, 1354)
print("Forward pass verified successfully!")

