# ASL-Learner: Real-Time Sign Language Education

**ASL-Learner** is a web-based American Sign Language (ASL) learning and recognition application built with **Google Antigravity**. It runs real-time sign recognition, interactive practice scoring, an ASL alphabet reference, and an isolated-sign sentence builder—all executing **100% locally in your browser** using WebAssembly, with zero video data leaving your device.

---

## ⚡ Important Model Decision: Pretrained Models Only

* **NO Model Training, Fine-Tuning, or Re-Training**: The application strictly uses verified pretrained deep learning weights directly from official GitHub sources.
* **NO Randomly Initialized Weights or Mock Predictions**: All predictions are computed by the pretrained deep learning model.
* **Permissive Open-Source Licensing**: All components adhere to the MIT License and Creative Commons attribution guidelines.

---

## 🔬 Pretrained Model Audit & Verification

We cloned and evaluated both official candidate repositories:
1. **GestureSign** (`shashank11yadav/GestureSign`)
2. **ASL-translator** (`aymenwnia/ASL-translator`)

### Model Comparison Table

| Metric / Attribute | GestureSign (Selected) | ASL-translator (Evaluated) |
| :--- | :--- | :--- |
| **Model Weights Artifact** | `web/public/models/model.onnx` (5.62 MB) | `models/sign_transformer_best.pth` (9.85 MB) |
| **Model Architecture** | Depthwise-Conv + Transformer (1.4M params) | Transformer Encoder (2.4M params) |
| **Pretrained Classes** | **250 isolated ASL signs** (Kaggle ASL Signs) | **250 isolated ASL signs** (Kaggle ASL Signs) |
| **Inference Runtime** | **ONNX Runtime Web (WASM SIMD)** | PyTorch (Requires desktop Python) |
| **Inference Latency (i5 CPU)** | **~4.4 ms** per sequence | **~5.8 ms** (model only, excl. 3 detectors) |
| **Feature Dimension** | **460 features** (115 Holistic points) | **390 features** (42 hands, 76 face, 12 pose) |
| **Webcam Invariance** | ✅ Face ratio aspect-correction & handedness canonicalization | ❌ Raw unnormalized pixel coordinates |
| **Confidence Calibration** | ✅ Calibrated temp (0.8827) & "Not sure" threshold (0.2903) | ❌ Raw uncalibrated softmax |
| **Metric Embeddings** | ✅ 128-d cosine embedding head for practice scoring | ❌ No metric embedding head |
| **License** | **MIT License** (Shashank Yadav) | **MIT License** (AWAINIA Aymen) |

### Why GestureSign Was Chosen as Primary:
1. **Browser Native & Privacy First**: Executes inside WebAssembly in the visitor's browser tab; zero video streaming to servers.
2. **Webcam Aspect Ratio Normalization**: Normalizes vertical proportions using face geometry (eye-to-mouth over ear-to-ear distance) to cancel landscape webcam stretching compared to portrait mobile training videos.
3. **Automatic Hand Dominance**: Detects dominant signing hand (by visibility and wrist travel) and mirrors if needed, supporting left- and right-handed signers seamlessly.
4. **Interactive Practice Scoring**: Produces normalized 128-d cosine embeddings that enable quantitative similarity scoring against target signs.

---

## 🚀 Application Architecture & Features

### Phase 1: Application Foundation
* **Responsive Single-Page Application (SPA)** built with React 19, TypeScript, Vite, and Tailwind CSS.
* **Multi-Page Navigation**:
  * **Landing Page**: Feature highlights, hardware benchmarks, and quick-start actions.
  * **Dashboard**: Practice streak counter, vocabulary mastery percentage, featured "Sign of the Day", and category decks.
  * **ASL Alphabet Guide**: Interactive 26-letter fingerspelling guide (A-Z) with mnemonics, finger positions, common pitfalls, and palm orientations.
  * **Sign Recognition**: Live webcam and replay recognition with real-time skeletal overlays.
  * **Practice Studio**: Target sign flashcards, dictionary links, and similarity scoring.
  * **Sentence Builder**: Isolated sign sequence accumulator with hold deduplication.
  * **Progress & Analytics**: Complete practice history log and category mastery breakdown.
  * **Technical About**: Full model audit report, architecture details, and open-source licenses.

### Phase 2: Real-Time Isolated Sign Recognition
* Loads `model.onnx` (1.4M parameters), `labels.json` (250 signs), and `model-meta.json`.
* MediaPipe Holistic tracker running inside a background Web Worker.
* Outputs **Top-3 model predictions** with animated confidence percentile bars.
* Calibrated **"Not sure"** rejection threshold (0.2903) to prevent false positives when hands are resting.
* Built-in **Replay Simulator Mode** with 6 genuine held-out signer clips (`blue`, `police`, `quiet`, `giraffe`, `smile`, `hello`) for testing without a camera.

### Phase 3: Interactive Learning & Practice
* Search and choose from all 250 signs or filter by thematic categories (Essentials, Animals, Food, Objects, Family, Colors, Body, Actions).
* Instant attempt evaluation via `practiceScore` (temperature-calibrated cosine similarity against target sign).
* Immediate visual feedback: "Passed (+100 XP)", "Close (model saw X, check hand orientation)", or "Try Again".
* Direct one-click link to ASL Dictionary video demonstrations.
* Auto-saves session streaks and attempts to `localStorage`.

### Phase 4: Isolated-Sign Sentence Builder
* **Prominent Linguistic Distinction**: Clearly documented as an isolated-sign sequence builder, **NOT** continuous ASL translation.
* **Hold Deduplication**: Prevents duplicate additions when a user holds the same sign without lowering hands.
* **Controls**: Add space, delete last word (backspace), clear canvas, pause/resume live feed, copy to clipboard, and **Speak Aloud** via the Web Speech Synthesis API.

### Phase 5: Verification & Quality Assurance
* **17 test suites (143 automated tests)** covering feature extraction, segmenter logic, camera handling, prototype matching, progress persistence, and alphabet data.

---

## 💻 System Requirements

* **Operating System**: Windows 10/11, macOS, or Linux.
* **Processor**: Intel Core i5 (e.g. i5-1155G7) or equivalent (CPU-friendly, ~4.4 ms per inference).
* **RAM**: 4 GB+ (tested on 16 GB).
* **Webcam**: Standard 720p or 1080p laptop camera (or use built-in Replay Mode).
* **Node.js**: v20+ (tested on v20.17.0).
* **Python** *(optional, for standalone model audits)*: Python 3.10+ with `torch` and `onnxruntime`.

---

## 🛠️ Quick Start & Execution

### 1. Install Dependencies
```bash
# From workspace root
npm --prefix repos/GestureSign/web install
```

### 2. Run the Development Server
```bash
npm run dev
```
Open **`http://localhost:5330`** in your browser.

### 3. Run Production Build
```bash
npm run build
```

### 4. Run Test Suite
```bash
npm test
```

### 5. Run Python Pretrained Model Audit Test
```bash
python scripts/standalone_model_test.py
```

---

## 📜 Attributions & Licenses

* **GestureSign**: MIT License, Copyright (c) 2026 Shashank Yadav. Source: [shashank11yadav/GestureSign](https://github.com/shashank11yadav/GestureSign)
* **ASL-translator**: MIT License, Copyright (c) 2026 AWAINIA Aymen. Source: [aymenwnia/ASL-translator](https://github.com/aymenwnia/ASL-translator)
* **Kaggle Google ASL Signs Dataset**: CC BY 4.0 International, Deaf Professional Arts Network (DPAN).
* **MediaPipe**: Apache License 2.0, Google LLC.

