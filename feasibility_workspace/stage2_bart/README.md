---
tags:
  - text2text-generation
  - transformers
  - english
  - bart
  - sign-language
library_name: transformers
language:
- en
metrics:
- bertscore
- bleu
- rouge
base_model:
- facebook/bart-base
---

# Model Card for Model ID

This model is a fine-tuned version of `facebook/bart-base`, trained to convert American Sign Language (ASL) gloss sequences into fluent English sentences. It is designed to assist in research, education, and accessibility applications involving gloss-based ASL interpretation. 
The model was trained using high-quality aligned pairs of gloss annotations and English translations, and evaluated using BERTScore.

## Model Details

### Model Description

This is the model card of a transformers model that has been pushed on the Hub. This model card has been automatically generated.

- **Developed by:** Dongjun Kim
- **Model type:** Text2Text Generation, Gloss2Eng
- **Language(s) (NLP):** English


## Intended Uses

This model is fine-tuned for translating American Sign Language (ASL) gloss input sequences into natural, grammatically correct English sentences. It can be used for:

- Building real-time sign language interpretation systems
- Research in sign language understanding and low-resource language translation
- Educational tools for ASL learners to see gloss-to-English transformation
- Data augmentation for multimodal ASL translation tasks

## Out-of-Scope Uses

The model is **not** suitable for:

- Translating from ASL videos or images directly (no visual input is processed)
- Formal legal or medical translation without human validation
- General-purpose translation outside ASL gloss context
- Languages other than English

## How to Use

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("rrrr66254/bart-gloss-finetuned")
model = AutoModelForSeq2SeqLM.from_pretrained("rrrr66254/bart-gloss-finetuned")

gloss_input = "YOU GO STORE TOMORROW?"
inputs = tokenizer(gloss_input, return_tensors="pt")
output = model.generate(**inputs)
print(tokenizer.decode(output[0], skip_special_tokens=True))
# Expected output is "Are you going to the store tomorrow?"
```

## Bias, Risks, and Limitations

This model is trained on American Sign Language (ASL) glosses mapped to natural English sentences. As such, it may inherit several limitations:

- **Data bias**: If the training data overrepresents certain sentence structures, cultural expressions, or gloss forms, the model may produce outputs that lack variety or inclusivity.
- **Limited linguistic scope**: The model only understands **ASL gloss** as input and **English** as output. It does not cover other sign languages or spoken/written languages.
- **Context loss**: ASL gloss does not encode facial expressions, spatial grammar, or non-manual signals, which are essential in ASL. The model may misrepresent meaning as a result.
- **Generalization risk**: The model may not generalize well to gloss styles or sentence structures it wasn’t trained on.

Outputs should not be used in **critical settings** (e.g., legal, medical, or emergency interpreting) without human review.

### Recommendations

- Human-in-the-loop: Always have a fluent signer or linguist verify model outputs in any production or educational setting.
- Data expansion: Consider fine-tuning with more diverse gloss datasets that include different dialects or informal structures.
- Downstream use: If used as part of a larger translation or accessibility pipeline, include disclaimers about potential misinterpretation due to a lack of non-manual signals.

## Training Details

### Training Data

The model was fine-tuned on a custom dataset of 1:1 pairs of ASL gloss and fluent English sentences.
The glosses are structured representations of ASL without punctuation, articles, or verb conjugation. Each gloss sentence is paired with a corresponding English sentence that captures its intended meaning.
The dataset was cleaned to remove non-English outputs, duplicates, and ill-formed pairs using custom filters.

### Training Procedure

The training used the Hugging Face `Trainer` API with a sequence-to-sequence objective.
The training leveraged a BART-based architecture (facebook/bart-base) to learn a mapping from gloss to fluent English sentences.

#### Preprocessing [optional]

- Input text was trimmed and normalized
- Tokenizer: Pretrained BART tokenizer
- Special tokens: `[INST]` and `[/INST]` were used to delimit gloss input and output reference



#### Training Hyperparameters

#### Training Hyperparameters

- **Base model**: `facebook/bart-base`
- **Epochs**: 3
- **Learning rate**: 5e-5
- **Batch size**: 4 per device (both train and eval)
- **Gradient accumulation**: Not used
- **Weight decay**: 0.01
- **Learning rate scheduler**: Linear (default in Trainer)
- **Precision**: Mixed precision (fp16=True)
- **Evaluation strategy**: Per epoch
- **Save strategy**: Per epoch (with `save_total_limit=2`)
- **Logging frequency**: Every 50 steps
- **Early stopping**: Custom callback based on BERTScore with patience = 2
- **Evaluation metric**: BERTScore (F1), computed with `microsoft/deberta-xlarge-mnli`

#### Factors

This model does not explicitly disaggregate results by demographic group, signer identity, or domain. However, the training data may implicitly reflect distributional biases present in publicly available gloss datasets.


#### Metrics

- **Primary metric**: BERTScore (F1), BLEU, and ROUGE
- **Model selection**: Best checkpoint based on highest validation BERTScore-F1
- BERTScore is used to evaluate semantic alignment, while BLEU and ROUGE provide additional insight into surface-level n-gram overlap. All metrics were evaluated using the same held-out set of 500 gloss-reference pairs.

### Results

After 2 epochs of training, the model achieved the following on the 500-pair evaluation set:

| Metric           | Score  |
| ---------------- | ------ |
| **BERTScore-F1** | 0.7191 |
| **BERTScore-P**  | 0.7399 |
| **BERTScore-R**  | 0.6983 |
| **BLEU-1**       | 0.7063 |
| **BLEU-2**       | 0.6175 |
| **BLEU-3**       | 0.5479 |
| **BLEU-4**       | 0.4821 |
| **ROUGE-1**      | 0.7587 |
| **ROUGE-2**      | 0.5874 |
| **ROUGE-L**      | 0.7312 |

- Qualitative inspection shows that most model outputs are fluent and contextually accurate. Common errors include omission of function words and minor verb tense mismatches.


#### Summary

This model demonstrates strong potential for gloss-to-English translation, with near-human fluency in many cases. However, further work is needed to improve generalization to informal gloss styles and integrate non-manual features.

## Model Card Authors

- Dongjun Kim

## Model Card Contact

- rrrr66254@gmail.com