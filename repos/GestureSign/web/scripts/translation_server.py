"""
translation_server.py
Lightweight local HTTP translation server running Glossa-BART on CPU (port 5005).
Endpoints:
  GET  /health
  POST /translate  {"glosses": ["HELLO", "HOW", "YOU"], "use_beam_search": false, "num_beams": 4}
"""

import os
import sys
import json
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Resolve paths robustly from script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
BART_DIR = PROJECT_ROOT / "feasibility_workspace" / "stage2_bart"

if not BART_DIR.is_dir():
    raise FileNotFoundError(f"[TranslationServer] BART_DIR directory not found: {BART_DIR}")

# Verify required model and tokenizer files
required_files = [
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]
for fname in required_files:
    file_path = BART_DIR / fname
    if not file_path.is_file():
        raise FileNotFoundError(f"[TranslationServer] Required checkpoint file missing: {file_path}")

print(f"[TranslationServer] Verified Glossa-BART directory: {BART_DIR}", flush=True)
print(f"[TranslationServer] Loading tokenizer and model (local_files_only=True)...", flush=True)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(str(BART_DIR), local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(str(BART_DIR), local_files_only=True)
model.eval()
print(f"[TranslationServer] Model successfully loaded in {time.time() - t0:.2f}s", flush=True)

class TranslationHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "online",
                "model": "rrrr66254/Glossa-BART",
                "device": "cpu"
            }).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/translate":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                req = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Invalid JSON: {e}"}).encode("utf-8"))
                return

            glosses = req.get("glosses", [])
            use_beam = req.get("use_beam_search", False)
            num_beams = req.get("num_beams", 4)

            if not glosses:
                res = {"english_translation": "", "latency_ms": 0.0, "glosses": []}
            else:
                prompt = " ".join([g.upper() for g in glosses])
                t_start = time.perf_counter()
                inputs = tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    if use_beam:
                        outputs = model.generate(
                            **inputs,
                            max_length=64,
                            num_beams=num_beams,
                            early_stopping=True,
                            no_repeat_ngram_size=2
                        )
                    else:
                        outputs = model.generate(
                            **inputs,
                            max_length=64,
                            do_sample=False
                        )
                trans = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                latency_ms = (time.perf_counter() - t_start) * 1000.0
                res = {
                    "english_translation": trans,
                    "latency_ms": round(latency_ms, 2),
                    "glosses": glosses,
                    "model": "rrrr66254/Glossa-BART"
                }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=5005):
    server = HTTPServer(("127.0.0.1", port), TranslationHandler)
    print(f"[TranslationServer] Listening on http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[TranslationServer] Shutting down.", flush=True)
        server.server_close()

if __name__ == "__main__":
    run_server()

