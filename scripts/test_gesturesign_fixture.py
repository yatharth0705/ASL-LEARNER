import json
import numpy as np
import onnxruntime as ort

with open("repos/GestureSign/shared/fixtures/features.json", "r") as f:
    fixture = json.load(f)

print(f"Fixture keys: {list(fixture.keys())}")
print(f"Total cases: {len(fixture['cases'])}")

case0 = fixture["cases"][0]
print(f"Case 0 name: {case0.get('name')}")
print(f"Case 0 keys: {list(case0.keys())}")

expected = np.array(case0["expected"]["features"], dtype=np.float32)
print(f"Expected shape: {expected.shape}")

# Run ONNX model
session = ort.InferenceSession("repos/GestureSign/web/public/models/model.onnx", providers=["CPUExecutionProvider"])

# The model expects [1, T, 460]
input_tensor = expected[np.newaxis, :, :] # shape (1, T, 460)
outputs = session.run(["logits", "embedding"], {"frames": input_tensor})
logits, embedding = outputs[0], outputs[1]

print(f"Logits shape: {logits.shape}, range: [{logits.min():.4f}, {logits.max():.4f}]")
print(f"Embedding shape: {embedding.shape}, norm: {np.linalg.norm(embedding):.4f}")

with open("repos/GestureSign/web/public/models/labels.json", "r") as f:
    labels_data = json.load(f)

# Softmax with temperature
with open("repos/GestureSign/web/public/models/model-meta.json", "r") as f:
    meta = json.load(f)
temp = meta["calibration"]["temperature"]
scaled_logits = logits[0] / temp
probs = np.exp(scaled_logits - np.max(scaled_logits))
probs = probs / probs.sum()

top5_idx = np.argsort(probs)[::-1][:5]
print("\nTop-5 Predictions for Case 0:")
for rank, idx in enumerate(top5_idx):
    sign_info = labels_data["signs"][idx]
    print(f"  {rank+1}. {sign_info['sign']} (idx {idx}): {probs[idx]*100:.2f}% (raw logit: {logits[0][idx]:.3f})")
