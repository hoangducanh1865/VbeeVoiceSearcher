# VbeeVoiceSearcher

Zero-shot Vietnamese TTS annotation classifier — dự đoán 5 trục chú thích (Ngữ điệu, Valence, Energy, Tốc độ, Chất giọng) từ đoạn text tiếng Việt.

## Setup

```bash
pip install -r requirements.txt
```

## Chạy

```bash
# Dự đoán một file, kết quả append vào data/input_label.jsonl
python main.py ./data/user_input/input1.txt

# So sánh nhiều model, xuất ảnh PNG ra experiment/
python experiment/compare_models.py ./data/user_input/input1.txt
```

## Google Colab

Thêm `HF_TOKEN` vào **Secrets** (biểu tượng 🔑 bên trái), sau đó:

```python
!pip install -r requirements.txt

# Single model
!python main.py ./data/user_input/input1.txt

# So sánh 3 model + xuất chart
!python experiment/compare_models.py ./data/user_input/input1.txt
```

> HF_TOKEN được tự động đọc từ Colab Secrets — không cần set thủ công.

## Output

**Terminal** — bảng markdown:

| Trục (VI)                    | Loại nhãn | Giá trị dự đoán       | Confidence |
|------------------------------|-----------|-----------------------|------------|
| Ngữ điệu                     | Đơn nhãn  | Trầm                  | 0.72       |
| Mức độ tích cực / tiêu cực   | Đơn nhãn  | Tiêu cực (buồn, giận) | 0.89       |
| Độ mạnh năng lượng (Arousal) | Đơn nhãn  | Nhẹ nhàng (low)       | 0.81       |
| Tốc độ                       | Đơn nhãn  | Chậm rãi              | 0.85       |
| Chất giọng                   | Đa nhãn   | Sâu lắng, Buồn bã     | 0.82, 0.79 |

**File** — `data/input_label.jsonl` (append mỗi lần chạy):
```jsonl
{"source": "input1.txt", "axis_en": "Valence", "axis_vi": "Mức độ tích cực / tiêu cực", "loai_nhan": "Đơn nhãn", "gia_tri": "Tiêu cực (buồn, giận)", "confidence": 0.89}
{"source": "input1.txt", "axis_en": "Style", "axis_vi": "Chất giọng", "loai_nhan": "Đa nhãn", "gia_tri": ["Sâu lắng", "Buồn bã"], "confidence": [0.82, 0.79]}
```

**Chart** — `experiment/compare_input1.png` (3 model × 5 trục):
- Hàng trên: bar chart 4 trục đơn nhãn
- Hàng dưới: heatmap Style với ★ đánh dấu nhãn ≥ 0.50

## Models (so sánh)

| Short name    | Model ID                                              | Size   |
|---------------|-------------------------------------------------------|--------|
| XLM-R-Large   | joeddav/xlm-roberta-large-xnli                        | ~1.1GB |
| mDeBERTa-v3   | MoritzLaurer/mDeBERTa-v3-base-mnli-xnli               | ~280MB |
| mDeBERTa-2M7  | MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 | ~280MB |
