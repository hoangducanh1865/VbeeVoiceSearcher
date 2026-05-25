# Audio Feature Extraction untuk PromptTTS2

## 📋 Tổng quan

Script này trích xuất **12 loại đặc trưng acoustic** từ file audio, được thiết kế để align với các attributes của PromptTTS2:
- **Gender** (male/female)
- **Speed** (fast/normal/slow)  
- **Volume** (high/normal/low)
- **Pitch** (high/normal/low)
- **Emotion** (happy/angry/sad/neutral)
- **Accent** (american/british)
- **Age** (adult/old)

## 🎵 Các đặc trưng trích xuất

| Đặc trưng | Mô tả | Liên quan tới |
|-----------|-------|---------------|
| `mel_spectrogram` | Time-frequency representation (80 mel bands) | TTS backbone input |
| `mfcc` | Mel-Frequency Cepstral Coefficients (13-dim) | Gender, emotion, accent |
| `f0_*` | Fundamental frequency (mean/median/std/min/max) | **Gender**, **pitch**, age |
| `rms_*` | RMS energy / loudness | **Volume** |
| `spectral_centroid_*` | Brightness/timbre của giọng nói | Emotion, gender |
| `spectral_rolloff_*` | High-frequency content | Voice quality |
| `zcr_*` | Zero-crossing rate (voicing indicator) | Speech/silence |
| `chroma_*` | 12-dimensional chroma features | Pitch content |
| `spectral_flux_*` | Temporal spectral changes | **Emotion**, speech dynamics |
| `onset_strength_*` | Temporal energy peaks | **Speed** (speech rate) |
| `duration` | Độ dài file audio (seconds) | **Speed** estimation |
| `frame_rate` | Frames per second | Speech rate proxy |

## 🚀 Cách sử dụng

### Step 1: Chuẩn bị dữ liệu

Chạy script để tạo thư mục và xem hướng dẫn tải dữ liệu:
```bash
conda activate voice_search
cd baseline/PromptTTS2
python3 setup_test_data.py
```

Script sẽ in ra hướng dẫn chi tiết để tải audio từ:
- **LJSpeech**: Single female speaker, tiếng Anh chuẩn
- **CMU Arctic**: Multiple speakers (male/female)
- **TIMIT**: Diverse speakers & accents
- **VoxCeleb1**: Real-world speech
- **LibriSpeech**: Read speech, clean audio

### Step 2: Tải audio

Chọn một option từ hướng dẫn và tải vài file vào `audio_data/` folder.

**Ví dụ nhanh (LJSpeech)**:
```bash
# Từ directory khác, download LJSpeech
wget https://data.keithito.com/data/LJ-Speech-1.1.tar.bz2
tar -xvf LJ-Speech-1.1.tar.bz2

# Copy vài file vào audio_data
cp LJ-Speech-1.1/wavs/LJ001-000[1-5].wav audio_data/
```

### Step 3: Trích xuất đặc trưng

```bash
python3 extract_features.py \
  --input_dir audio_data \
  --output_dir features \
  --sr 22050 \
  --output_csv features_stats.csv
```

**Options:**
- `--input_dir`: Thư mục input (default: audio_data)
- `--output_dir`: Thư mục output (default: features)
- `--sr`: Sampling rate (default: 22050 Hz)
- `--output_csv`: Tên file CSV thống kê (default: features_stats.csv)

### Step 4: Kiểm tra output

```bash
ls -lh features/
# Kết quả:
# - *.npy: Mel-spectrogram (80 x T)
# - *.npz: Tất cả đặc trưng (dict format)
# - features_stats.csv: Bảng thống kê scalar values
```

## 📊 Cấu trúc Output

### 1. `.npy` files (Mel-spectrogram)
```python
import numpy as np
mel_spec = np.load('audio_001.npy')  # Shape: (80, time_steps)
```

### 2. `.npz` files (Full features)
```python
import numpy as np
features_dict = np.load('audio_001.npz')
mel = features_dict['mel_spectrogram']  # (80, T)
mfcc = features_dict['mfcc']            # (13, T)
f0_mean = features_dict['f0_mean']      # scalar
rms_mean = features_dict['rms_mean']    # scalar
```

### 3. `.csv` file (Statistics)
```
filename,duration,f0_mean,f0_std,rms_mean,rms_std,spectral_centroid_mean,...
audio_001.wav,5.23,145.2,23.5,0.087,0.012,2850.3,...
audio_002.wav,4.91,120.1,18.2,0.092,0.015,2750.1,...
```

## 🔬 Phân tích kết quả

### Cách phân loại từ đặc trưng

```python
import pandas as pd
import numpy as np

# Load statistics
stats = pd.read_csv('features/features_stats.csv')

# 1. Dự đoán GENDER (từ F0)
stats['predicted_gender'] = stats['f0_mean'].apply(
    lambda x: 'female' if x > 200 else 'male'
)

# 2. Dự đoán VOLUME (từ RMS energy)
stats['predicted_volume'] = pd.cut(
    stats['rms_mean'], 
    bins=[0, 0.06, 0.10, 1.0],
    labels=['low', 'normal', 'high']
)

# 3. Dự đoán PITCH (từ F0 variation)
stats['predicted_pitch'] = pd.cut(
    stats['f0_mean'],
    bins=[0, 130, 170, 500],
    labels=['low', 'normal', 'high']
)

# 4. Dự đoán SPEED (từ frame rate)
stats['predicted_speed'] = pd.cut(
    stats['frame_rate'],
    bins=[0, 50, 70, 300],
    labels=['slow', 'normal', 'fast']
)

print(stats[['filename', 'predicted_gender', 'predicted_volume', 
             'predicted_pitch', 'predicted_speed']])
```

## ⚙️ Dependencies

```
librosa >= 0.10
numpy >= 1.20
scipy >= 1.6
tqdm >= 4.6
```

Cài đặt:
```bash
conda activate voice_search
pip install librosa numpy scipy tqdm
```

## 🐛 Troubleshooting

### Lỗi: "No module named 'librosa'"
```bash
conda activate voice_search
pip install librosa
```

### Lỗi: "PySoundFile failed. Trying audioread instead"
Bình thường, librosa sẽ fallback to audioread. Nếu cần soundfile:
```bash
conda install -c conda-forge libsndfile
pip install soundfile
```

### Lỗi: "No audio files found"
Kiểm tra:
1. File có trong `audio_data/` không?
2. Định dạng phải là `.wav`, `.mp3`, `.flac`, hoặc `.ogg`
3. Chạy `ls audio_data/` để verify

## 📚 Tham khảo

- [Librosa Documentation](https://librosa.org/)
- [Speech Features Extraction](https://en.wikipedia.org/wiki/Speech_recognition)
- [PromptTTS2 Paper](https://arxiv.org/abs/2309.08145)

## 💡 Tips để cải thiện

1. **Tăng số lượng audio**: Dùng ít nhất 50-100 file cho training tốt hơn
2. **Đa dạng speakers**: Mix multiple gender, age, accent
3. **Normalize features**: Sử dụng standard scaling trước khi training
4. **Feature selection**: Chọn 5-10 features quan trọng nhất theo use case
5. **Augmentation**: Thêm variations (pitch shift, time stretch, noise)

---

**Last updated**: April 2026 | **Version**: 1.0
