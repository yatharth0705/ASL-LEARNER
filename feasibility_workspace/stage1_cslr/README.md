---
license: mit
library_name: pytorch
tags:
- sign-language
- cslr
- continuous-sign-language-recognition
- asl
- how2sign
- pose-estimation
- ctc
---

# How2Sign Pose-Based CSLR

Pose-based Continuous Sign Language Recognition (ASL video -> gloss) trained
on the How2Sign dataset with LLM-generated pseudo-glosses.

## Architecture

| component       | spec                                                                    |
|-----------------|-------------------------------------------------------------------------|
| Input           | 153-D per frame: 150-D MediaPipe Holistic keypoints + 3-D presence mask |
| Pose layout     | 8 pose + 21 left-hand + 21 right-hand landmarks, xyz each               |
| Frame rate      | 8 fps effective (stride 3 over ~24 fps source video)                    |
| Stem            | Linear(153 -> 256) + LayerNorm + GELU + Dropout                         |
| Position        | Sinusoidal positional encoding                                          |
| Encoder         | 4 x TransformerEncoderLayer (d=256, heads=4, FF=1024, pre-norm GELU)    |
| Head            | Linear(256 -> 1354)                                         |
| Loss            | CTC (blank id = 1)                                         |
| Parameters      | ~3.5M                                                    |

## Training data

- **Source:** How2Sign realigned train split, ~31k sentence clips.
- **Pose extraction:** MediaPipe Holistic (8 pose + 42 hand landmarks per frame).
- **Glosses:** **LLM-generated pseudo-glosses**, not human-annotated. Noisier
  supervision than e.g. PHOENIX14T.
- **Vocabulary:** 1354 tokens (full vocab, no `<unk>` cap).
- **Internal val:** ~10% of training **videos** held
  out (group-by-video, seed=13). Official How2Sign val/test
  stay reserved for end-to-end pipeline evaluation since they have no glosses.

## Evaluation

| metric                                                       | value                              |
|--------------------------------------------------------------|------------------------------------|
| Internal val WER (overall, full vocab)                       | 95.24%  |
| Internal val WER (sentences using only top-500 tokens) | 93.90%  |

The overall WER is high because ~10k of the 18k vocab tokens occur <=3 times
in train (long fingerspelled words, rare proper nouns). The head-only WER is
the representative number for whether the pipeline learns the common signs.

## Usage

```python
import json, numpy as np, torch
from huggingface_hub import hf_hub_download
# Requires the PoseTransformerCSLR class from the training notebook.
from your_pkg import PoseTransformerCSLR

model = PoseTransformerCSLR.from_pretrained("manohonsy/how2sign-pose-cslr").eval()

vocab = json.load(open(hf_hub_download("manohonsy/how2sign-pose-cslr", "vocab.json")))["token_to_id"]
id_to_token = {i: t for t, i in vocab.items()}
blank_id = vocab["<blank>"]

# Run on a .npz produced by how2sign_prep (or your own MediaPipe Holistic pipeline)
data = np.load("your_sample.npz")
features, mask = data["features"].astype("float32"), data["mask"].astype(bool)

# (apply normalize_features + concat mask as in the training notebook)
# x: torch.FloatTensor (1, T, 153)
log_probs, lens = model(x, torch.tensor([x.shape[1]]))

preds = log_probs.argmax(-1)[0].tolist()
collapsed, prev = [], None
for p in preds:
    if p != prev: collapsed.append(p); prev = p
gloss_ids = [p for p in collapsed if p != blank_id]
print(" ".join(id_to_token[i] for i in gloss_ids))
```

## Caveats

- **Pseudo-gloss supervision.** Glosses are LLM-generated. Reported WER
  reflects fit to pseudo-glosses, not to ground-truth ASL gloss.
- **Long-tail vocabulary.** ~10k tokens occur <=3 times. Effectively
  unlearnable from this corpus size. Head-only WER is the meaningful signal.
- **Pose-only input.** Misses appearance cues (fine handshape, non-manual
  markers). Trade-off documented in ADR-003.
- **MediaPipe sensitivity.** Landmark quality drops in low light, partial
  occlusion, or non-frontal camera angles.

## License

MIT.

## Related

Implements **ADR-003: Pose-Based CSLR Front-End** of the project. See
ADR-001 (gloss-based pipeline) and ADR-002 (mBART-50 translator) for the
broader architecture.
