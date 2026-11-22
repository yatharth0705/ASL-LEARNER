import os
import json
import hashlib
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

feasibility_dir = os.path.abspath("feasibility_workspace")
os.makedirs(feasibility_dir, exist_ok=True)
stage1_dir = os.path.join(feasibility_dir, "stage1_cslr")
stage2_dir = os.path.join(feasibility_dir, "stage2_bart")
os.makedirs(stage1_dir, exist_ok=True)
os.makedirs(stage2_dir, exist_ok=True)

api = HfApi()

print("=== 1. CHECKING STAGE 1 REPOSITORY (manohonsy/how2sign-pose-cslr) ===")
try:
    s1_info = api.model_info("manohonsy/how2sign-pose-cslr")
    print("Stage 1 Model Info:")
    print(f"  ID: {s1_info.id}")
    print(f"  Sha: {s1_info.sha}")
    print(f"  Siblings: {[s.rfilename for s in s1_info.siblings]}")
except Exception as e:
    print(f"Error fetching s1_info: {e}")

print("\n=== 2. CHECKING STAGE 2 REPOSITORY (rrrr66254/Glossa-BART vs rrrr66254/bart-gloss-finetuned) ===")
for candidate in ["rrrr66254/Glossa-BART", "rrrr66254/bart-gloss-finetuned"]:
    try:
        s2_info = api.model_info(candidate)
        print(f"Candidate {candidate}: FOUND")
        print(f"  ID: {s2_info.id}")
        print(f"  Sha: {s2_info.sha}")
        print(f"  Siblings: {[s.rfilename for s in s2_info.siblings]}")
    except Exception as e:
        print(f"Candidate {candidate}: FAILED -> {e}")

print("\n=== 3. DOWNLOADING STAGE 1 FILES ===")
snapshot_download(repo_id="manohonsy/how2sign-pose-cslr", local_dir=stage1_dir)
print(f"Stage 1 downloaded to {stage1_dir}")
for f in os.listdir(stage1_dir):
    fpath = os.path.join(stage1_dir, f)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath)
        with open(fpath, "rb") as fp:
            sha256 = hashlib.sha256(fp.read()).hexdigest()
        print(f"  {f}: {size} bytes ({size / (1024*1024):.2f} MB), SHA256: {sha256}")

print("\nScript completed preliminary download successfully.")

