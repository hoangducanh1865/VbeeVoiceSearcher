# VbeeVoiceSearcher

Zero-shot Vietnamese TTS annotation classifier — dự đoán 5 trục chú thích (Ngữ điệu, Valence, Energy, Tốc độ, Chất giọng) từ đoạn text tiếng Việt.

## Setup

```bash
pip install -r requirements.txt
```

> Lần đầu chạy sẽ tải model `joeddav/xlm-roberta-large-xnli` (~1.1 GB) về cache.

## Chạy

```bash
python main.py ./data/user_input/input1.txt
```

## Output mẫu

| Trục (VI)                    | Loại nhãn | Giá trị dự đoán       | Confidence |
|------------------------------|-----------|-----------------------|------------|
| Ngữ điệu                     | Đơn nhãn  | Trầm                  | 0.72       |
| Mức độ tích cực / tiêu cực   | Đơn nhãn  | Tiêu cực (buồn, giận) | 0.89       |
| Độ mạnh năng lượng (Arousal) | Đơn nhãn  | Nhẹ nhàng (low)       | 0.81       |
| Tốc độ                       | Đơn nhãn  | Chậm rãi              | 0.85       |
| Chất giọng                   | Đa nhãn   | Sâu lắng, Buồn bã     | 0.82, 0.79 |

## Google Colab

```python
!pip install -r requirements.txt
!python main.py ./data/user_input/input1.txt
```
