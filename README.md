# VbeeVoiceSearcher

Vietnamese TTS annotation classifier — dự đoán 5 trục chú thích (Ngữ điệu, Valence, Energy, Tốc độ, Chất giọng) từ đoạn text tiếng Việt. Hỗ trợ hai backend: **LLM** (zero-shot NLI) và **ML** (classical classifiers trained on synthetic data).

## Setup

```bash
pip install -r requirements.txt
```

Thêm file `.env` ở root:
```
GEMINI_API="your_gemini_api_key"
```

## Cấu trúc project

```
src/
  label_model/        ← LLM backend (zero-shot NLI)
    config.py         ← cấu hình model, path, threshold
    classifier.py     ← TextClassifier (HuggingFace pipeline)
    evaluator.py      ← tính accuracy/precision/recall/F1
    formatter.py      ← render bảng + JSONL output
    metadata_loader.py
    text_loader.py
    validator.py
  dataset/
    generator.py      ← generate dataset qua Gemini API
  ml_model/           ← ML backend
    base.py           ← BaseMLClassifier ABC
    features.py       ← TF-IDF vectorizer
    trainer.py        ← train/evaluate/infer orchestration
    svm/model.py      ← LinearSVC + OneVsRest
    logistic_regression/model.py
    naive_bayes/model.py
data/
  meta_data/          ← định nghĩa 5 trục chú thích
  user_input/         ← user_input.jsonl (dùng để infer tự do)
  user_input_label/   ← ground truth cho 20 stories
  user_input_predict/ ← LLM predictions (per model subfolder)
  dataset/
    train.jsonl       ← generated: 240 samples
    test.jsonl        ← generated: 60 samples (dùng để evaluate)
  ml_model/           ← trained ML artifacts (model.pkl per model)
experiment/
  compare_models.py   ← so sánh 7 LLM models trên test set
```

## Workflow

### Bước 1 — Generate dataset (chỉ cần chạy 1 lần)

```bash
python -m src.dataset.generator          # ~300 Gemini API calls, ~5-10 phút
# → data/dataset/train.jsonl (240 samples)
# → data/dataset/test.jsonl  (60 samples)
```

### Bước 2 — Train ML models

```bash
python main.py --mode ml --action train --ml-model svm
python main.py --mode ml --action train --ml-model logistic_regression
python main.py --mode ml --action train --ml-model naive_bayes
# → data/ml_model/<model>/model.pkl
```

### Bước 3 — Evaluate ML trên test set

```bash
python main.py --mode ml --action evaluate --ml-model svm
```

### Bước 4 — Infer ML trên 1 story

```bash
python main.py --mode ml --action infer --ml-model svm --story story_03
```

### LLM mode (zero-shot, không cần train)

```bash
# Infer tất cả stories trong user_input.jsonl
python main.py --mode llm

# Chọn model khác
python main.py --mode llm --model MoritzLaurer/mDeBERTa-v3-base-mnli-xnli

# So sánh 7 LLM models + metrics heatmap trên test set
python experiment/compare_models.py

# Chọn story hiển thị trong chart sections 1-3
python experiment/compare_models.py --sample gen_005
```

## Google Colab

Thêm `HF_TOKEN` và `GEMINI_API` vào **Secrets** (biểu tượng 🔑), sau đó:

```python
!pip install -r requirements.txt
!python -m src.dataset.generator
!python main.py --mode ml --action train --ml-model svm
!python main.py --mode ml --action evaluate --ml-model svm
```

> HF_TOKEN được tự động đọc từ Colab Secrets — không cần set thủ công.

## Output

**Terminal — ML infer:**
```
── story_03 (ML: svm) ──
| Trục (VI)                    | Loại nhãn | Giá trị dự đoán          |
|------------------------------|-----------|--------------------------|
| Ngữ điệu                     | Đơn nhãn  | Trung cao                |
| Mức độ tích cực / tiêu cực   | Đơn nhãn  | Tích cực (vui, hài lòng) |
| Độ mạnh năng lượng (Arousal) | Đơn nhãn  | Nhẹ nhàng (low)          |
| Tốc độ                       | Đơn nhãn  | Bình thường              |
| Chất giọng                   | Đa nhãn   | Ngọt ngào, Ôn hoà        |
```

**Terminal — ML evaluate:**
```
── Evaluation Metrics ──
Axis                   Acc      Prec       Rec        F1
─────────────────────────────────────────────────────────
Intonation           0.750     0.712     0.738     0.721
Valence              0.833     0.821     0.810     0.813
...
Overall                                            0.756
```

**LLM Predictions** — `data/user_input_predict/<model>/user_input_predict.jsonl`:
```jsonl
{"id": "story_01", "model": "XLM-R-L", "Intonation": "Trầm", "Valence": "Tiêu cực (buồn, giận)", "Energy": "Nhẹ nhàng (low)", "Temporal": "Chậm rãi", "Style": ["Sâu lắng", "Buồn bã"]}
```

**Chart** — `experiment/compare_input.png` (4 sections):
- Section 1: bar chart 4 trục đơn nhãn (confidence per model)
- Section 2: grouped bar chart Style (10 nhãn × 7 model)
- Section 3: top-3 Style labels per model
- Section 4: metrics heatmap (F1 + Accuracy, 7 model × 5 trục + Overall)

## LLM Models (zero-shot NLI)

| Short name   | Model ID                                                       | Size   |
|--------------|----------------------------------------------------------------|--------|
| XLM-R-L      | joeddav/xlm-roberta-large-xnli                                 | ~1.1GB |
| XLM-R-B      | symanto/xlm-roberta-base-snli-mnli-anli-xnli                   | ~500MB |
| mDeBERTa-v3  | MoritzLaurer/mDeBERTa-v3-base-mnli-xnli                        | ~280MB |
| mDeBERTa-2M7 | MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7      | ~280MB |
| MiniLM-L6    | MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli                | ~120MB |
| MiniLM-L12   | MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli               | ~230MB |
| DeBERTa-v3-L | cross-encoder/nli-deberta-v3-large                             | ~900MB |

## ML Models (TF-IDF + classical classifiers)

| Name                 | Single-label          | Style (multi-label)            | Artifact                             |
|----------------------|-----------------------|--------------------------------|--------------------------------------|
| svm                  | LinearSVC             | OneVsRest(LinearSVC)           | data/ml_model/svm/model.pkl          |
| logistic_regression  | LogisticRegression    | OneVsRest(LogisticRegression)  | data/ml_model/logistic_regression/   |
| naive_bayes          | ComplementNB          | OneVsRest(ComplementNB)        | data/ml_model/naive_bayes/           |

Features: TF-IDF char_wb n-grams (2-4), max 50k features, sublinear TF.
