import os
import hashlib
from huggingface_hub import hf_hub_download
import shutil

stage2_dir = os.path.abspath("feasibility_workspace/stage2_bart")
os.makedirs(stage2_dir, exist_ok=True)

print("Starting direct download of model.safetensors for rrrr66254/Glossa-BART...")
downloaded_path = hf_hub_download(
    repo_id="rrrr66254/Glossa-BART",
    filename="model.safetensors",
    local_dir=stage2_dir,
    force_download=False
)
print(f"Downloaded to: {downloaded_path}")

target_path = os.path.join(stage2_dir, "model.safetensors")
if os.path.exists(target_path):
    size = os.path.getsize(target_path)
    print(f"File exists: {target_path}, size = {size} bytes ({size / (1024*1024):.2f} MB)")
    with open(target_path, "rb") as f:
        h = hashlib.sha256()
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    print(f"SHA256: {h.hexdigest()}")
else:
    print("Warning: target_path not found!")

