# VbeeVoiceSearcher

Zero-shot Vietnamese TTS annotation classifier — dự đoán 5 trục chú thích (Ngữ điệu, Valence, Energy, Tốc độ, Chất giọng) từ đoạn text tiếng Việt.

## Setup

```bash
pip install -r requirements.txt
```

## Cấu trúc project

```
src/label_model/       ← toàn bộ source code
  config.py            ← cấu hình model, path, threshold
  classifier.py        ← TextClassifier (zero-shot NLI)
  metadata_loader.py   ← đọc meta_data.jsonl
  text_loader.py       ← load stories JSONL + ground truth
  evaluator.py         ← tính accuracy/precision/recall/F1
  formatter.py         ← render bảng + JSONL output
  validator.py         ← kiểm tra nhãn hợp lệ
data/
  meta_data/           ← định nghĩa 5 trục chú thích
  user_input/          ← user_input.jsonl (20 stories đầu vào)
  user_input_label/    ← user_input_label.jsonl (ground truth)
  user_input_predict/
    XLM-R-L/           ← predictions của từng model
    mDeBERTa-v3/
    ...
experiment/
  compare_models.py    ← so sánh 7 model, xuất chart PNG
```

## Chạy

```bash
# Dự đoán 20 stories với model mặc định → data/user_input_predict/XLM-R-L/user_input_predict.jsonl
python main.py

# Chọn model khác
python main.py --model MoritzLaurer/mDeBERTa-v3-base-mnli-xnli

# So sánh 7 model + đánh giá metrics + xuất chart PNG
python experiment/compare_models.py

# Chọn story hiển thị trong sections 1-3
python experiment/compare_models.py --sample story_05
```

## Google Colab

Thêm `HF_TOKEN` vào **Secrets** (biểu tượng 🔑 bên trái), sau đó:

```python
!pip install -r requirements.txt

# Single model, 20 stories
!python main.py

# So sánh 7 model + xuất chart
!python experiment/compare_models.py
```

> HF_TOKEN được tự động đọc từ Colab Secrets — không cần set thủ công.

## Output

**Terminal** — bảng per story:

```
── story_01 ──
| Trục (VI)                    | Loại nhãn | Giá trị dự đoán   | Confidence |
|------------------------------|-----------|-------------------|------------|
| Ngữ điệu                     | Đơn nhãn  | Trầm              | 0.72       |
| Mức độ tích cực / tiêu cực   | Đơn nhãn  | Tiêu cực          | 0.61       |
| Độ mạnh năng lượng (Arousal) | Đơn nhãn  | Nhẹ nhàng (low)   | 0.81       |
| Tốc độ                       | Đơn nhãn  | Chậm rãi          | 0.85       |
| Chất giọng                   | Đa nhãn   | Sâu lắng, Buồn bã | 0.82, 0.79 |
```

**Predictions** — `data/user_input_predict/<model>/user_input_predict.jsonl`:
```jsonl
{"id": "story_01", "model": "XLM-R-L", "Intonation": "Trầm", "Valence": "Tiêu cực (buồn, giận)", "Energy": "Nhẹ nhàng (low)", "Temporal": "Chậm rãi", "Style": ["Sâu lắng", "Buồn bã"]}
```

**Chart** — `experiment/compare_input.png` (4 sections):
- Section 1: bar chart 4 trục đơn nhãn (confidence per model)
- Section 2: grouped bar chart Style (10 nhãn × 7 model)
- Section 3: top-3 Style labels per model
- Section 4: metrics heatmap (F1 + Accuracy, 7 model × 5 trục + Overall)

## Models

| Short name   | Model ID                                                       | Size   |
|--------------|----------------------------------------------------------------|--------|
| XLM-R-L      | joeddav/xlm-roberta-large-xnli                                 | ~1.1GB |
| XLM-R-B      | symanto/xlm-roberta-base-snli-mnli-anli-xnli                   | ~500MB |
| mDeBERTa-v3  | MoritzLaurer/mDeBERTa-v3-base-mnli-xnli                        | ~280MB |
| mDeBERTa-2M7 | MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7      | ~280MB |
| MiniLM-L6    | MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli                | ~120MB |
| MiniLM-L12   | MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli               | ~230MB |
| DeBERTa-v3-L | cross-encoder/nli-deberta-v3-large                             | ~900MB |
