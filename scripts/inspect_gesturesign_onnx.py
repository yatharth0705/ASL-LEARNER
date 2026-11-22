import onnxruntime as ort
import json
import numpy as np

# Load ONNX model
model_path = "repos/GestureSign/web/public/models/model.onnx"
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

print("--- GestureSign ONNX Model Info ---")
for inp in session.get_inputs():
    print(f"Input: name={inp.name}, shape={inp.shape}, type={inp.type}")
for out in session.get_outputs():
    print(f"Output: name={out.name}, shape={out.shape}, type={out.type}")

# Check model metadata
with open("repos/GestureSign/web/public/models/model-meta.json", "r") as f:
    meta = json.load(f)
print(f"Meta: {meta}")

with open("repos/GestureSign/web/public/models/labels.json", "r") as f:
    labels_data = json.load(f)
signs = labels_data.get('signs', [])
print(f"Total signs: {len(signs)}, First 5: {[s['sign'] for s in signs[:5]]}")

