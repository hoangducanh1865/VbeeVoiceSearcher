# VbeeVoiceSearcher

Vietnamese TTS annotation classifier — dự đoán 5 trục (Ngữ điệu, Valence, Energy, Tốc độ, Chất giọng). Hỗ trợ backend **LLM** (zero-shot NLI) và **ML** (TF-IDF + classical classifiers).

## Setup

```bash
pip install -r requirements.txt
```

`.env`:
```
GEMINI_API="your_gemini_api_key"
```

## Cấu trúc

```
src/
  model_llm/      ← LLM backend (zero-shot NLI via HuggingFace)
  model_ml/       ← ML backend (SVM / Logistic Regression / Naive Bayes)
  dataset/        ← Gemini dataset generator
data/
  meta_data/      ← định nghĩa 5 trục chú thích
  user_input/     ← user_input.jsonl (text + labels, dùng để infer tự do)
  user_input_predict/  ← LLM predictions (per model subfolder)
  dataset/        ← train.jsonl / test.jsonl (generated)
  model_ml/       ← trained ML artifacts
experiment/
  compare_models.py    ← so sánh LLM / ML models
```

## Workflow ML

```bash
# 1. Generate dataset (~300 Gemini calls, ~5-10 phút)
python -m src.dataset.generator

# 2. Train
python main.py --mode ml --action train --model-ml svm
python main.py --mode ml --action train --model-ml logistic_regression
python main.py --mode ml --action train --model-ml naive_bayes

# 3. Evaluate trên test set
python main.py --mode ml --action evaluate --model-ml svm

# 4. Infer 1 story từ data/user_input/user_input.jsonl
python main.py --mode ml --action infer --model-ml svm --story story_03
```

## LLM mode

```bash
# Infer tất cả stories trong user_input.jsonl
python main.py --mode llm [--model <hf_model_id>]
```

## So sánh models

```bash
python experiment/compare_models.py           # chạy cả LLM + ML
python experiment/compare_models.py --mode llm   # → compare_input_llm.png
python experiment/compare_models.py --mode ml    # → compare_input_ml.png
```

## Google Colab

Thêm `HF_TOKEN` và `GEMINI_API` vào **Secrets** (🔑), sau đó:

```python
!pip install -r requirements.txt
!python -m src.dataset.generator
!python main.py --mode ml --action train --model-ml svm
!python experiment/compare_models.py --mode ml
```

## LLM Models

| Short name   | Model ID                                                       | Size   |
|--------------|----------------------------------------------------------------|--------|
| XLM-R-L      | joeddav/xlm-roberta-large-xnli                                 | ~1.1GB |
| XLM-R-B      | symanto/xlm-roberta-base-snli-mnli-anli-xnli                   | ~500MB |
| mDeBERTa-v3  | MoritzLaurer/mDeBERTa-v3-base-mnli-xnli                        | ~280MB |
| mDeBERTa-2M7 | MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7      | ~280MB |
| MiniLM-L6    | MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli                | ~120MB |
| MiniLM-L12   | MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli               | ~230MB |
| DeBERTa-v3-L | cross-encoder/nli-deberta-v3-large                             | ~900MB |

## ML Models

Features: TF-IDF char_wb n-grams (2-4), 50k features, sublinear TF.

| Name                | Classifier        |
|---------------------|-------------------|
| svm                 | LinearSVC         |
| logistic_regression | LogisticRegression|
| naive_bayes         | ComplementNB      |

Style (multi-label): tất cả dùng `OneVsRestClassifier`.
