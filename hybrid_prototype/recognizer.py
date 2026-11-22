import os
import sys
import json
import time
from typing import Dict, Any, List, Optional
import numpy as np
import onnxruntime as ort

# Add GestureSign pipeline to sys.path
pipeline_path = os.path.abspath("repos/GestureSign/pipeline/src")
if pipeline_path not in sys.path:
    sys.path.insert(0, pipeline_path)

import gesturesign.features as gf
import gesturesign.resample as gr
from segmenter import Segment

class GestureSignRecognizer:
    def __init__(
        self,
        model_path: str = "repos/GestureSign/web/public/models/model.onnx",
        labels_path: str = "repos/GestureSign/web/public/models/labels.json",
        meta_path: str = "repos/GestureSign/web/public/models/model-meta.json"
    ):
        model_path = os.path.abspath(model_path)
        labels_path = os.path.abspath(labels_path)
        meta_path = os.path.abspath(meta_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels not found at {labels_path}")

        # Load metadata & calibration
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        self.temperature = self.meta.get("calibration", {}).get("temperature", 0.88265)
        self.not_sure_below = self.meta.get("calibration", {}).get("notSureBelow", 0.29028)

        # Load labels
        with open(labels_path, "r") as f:
            ldata = json.load(f)
        self.labels = ldata.get("signs", ldata)

        # Initialize ONNX session
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        
    def classify_segment(self, segment: Segment) -> Dict[str, Any]:
        """Process segment frames through GestureSign exact 460-D pipeline and ONNX model."""
        T = len(segment.frames)
        if T == 0:
            raise ValueError("Segment has no frames")

        # Convert list of 1629-element frames to [T, 543, 3] array
        raw_seq = np.array(segment.frames, dtype=np.float32).reshape((T, 543, 3))

        # 1. Resample to 30 FPS
        src_fps = segment.fps if segment.fps > 0 else 30.0
        resampled = gr.resample(raw_seq, src_fps, 30.0)

        # 2. Ensure minimum 8 frames (stretch if needed)
        if len(resampled) < 8 and len(resampled) > 1:
            positions = np.linspace(0, len(resampled) - 1, 8, dtype=np.float64)
            resampled = gr.interpolate(resampled, positions)

        # 3. Fit length (cap at maxFrames = 64)
        fitted = gr.fit_length(resampled, 64)

        # 4. Extract sequence features (460 dimensions per frame)
        t_feat = time.perf_counter()
        features = gf.sequence_features(fitted, 'hands_pose_lips')
        feat_time = (time.perf_counter() - t_feat) * 1000

        # 5. Run ONNX inference
        input_tensor = features[np.newaxis, :].astype(np.float32)
        t_infer = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: input_tensor})
        infer_time = (time.perf_counter() - t_infer) * 1000

        raw_logits = outputs[0][0] # [250]
        embedding = outputs[1][0] if len(outputs) > 1 else np.zeros(128, dtype=np.float32)

        # Calibrated Softmax
        scaled_logits = raw_logits / self.temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probabilities = exp_logits / np.sum(exp_logits)

        # Top-3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]
        top3 = []
        for rank, idx in enumerate(top_indices):
            top3.append({
                "rank": rank + 1,
                "index": int(idx),
                "sign": self.labels[idx]["sign"],
                "label": self.labels[idx].get("label", self.labels[idx]["sign"]),
                "probability": float(probabilities[idx]),
                "raw_logit": float(raw_logits[idx])
            })

        top1 = top3[0]
        return {
            "sign": top1["sign"],
            "confidence": top1["probability"],
            "raw_logit": top1["raw_logit"],
            "top3": top3,
            "embedding": embedding,
            "latency_ms": infer_time,
            "feature_time_ms": feat_time,
            "total_ms": feat_time + infer_time,
            "frames": len(fitted),
            "raw_frames": T,
            "fps": src_fps,
            "duration_ms": segment.duration_ms
        }

    def classify_frames(self, frames: List[np.ndarray], fps: float = 30.0) -> Dict[str, Any]:
        duration_ms = (len(frames) / fps) * 1000.0 if fps > 0 else 1000.0
        timestamps = [i * (1000.0 / fps) for i in range(len(frames))]
        seg = Segment(frames=frames, timestamps=timestamps, fps=fps, duration_ms=duration_ms)
        return self.classify_segment(seg)

