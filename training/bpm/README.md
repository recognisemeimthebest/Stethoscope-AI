# BPM — 온디바이스 심박수 검출 파이프라인

ESP32에서 직접 심박수(BPM)를 계산하기 위한 경량 모델입니다.  
Shannon 에너지 기반 라벨링 → CNN 학습 → TFLite 변환 → C 헤더 추출까지의 전체 파이프라인입니다.

---

## 파이프라인

```
01. Shannon 에너지로 심음 피크 자동 라벨링
        ↓
02. 경량 CNN beat detector 학습
        ↓
03. TFLite 변환 / C 헤더 가중치 추출
        ↓
04. ESP32에서 BPM 계산 평가
```

---

## 파일 설명

### 파이프라인 (순서대로)
| 파일 | 단계 | 역할 |
|------|------|------|
| `01_shannon_labeling.py` | 라벨링 | Shannon 에너지로 S1/S2 피크 자동 검출 |
| `01b_verify_labels.py` | 검증 | 자동 라벨 품질 확인 |
| `02_train_beat_detector.py` | 학습 | PyTorch CNN beat detector 학습 |
| `02b_train_tf.py` | 학습 | TensorFlow 버전 학습 |
| `03_convert_tflite.py` | 변환 | TFLite 포맷으로 변환 |
| `03_export_c_weights.py` | 변환 | C 헤더 파일로 가중치 추출 (ESP32 직접 탑재) |
| `04_evaluate_bpm.py` | 평가 | BPM 정확도 평가 |

### 경량 모델
| 파일 | 역할 |
|------|------|
| `model.py` | 경량 CNN 모델 정의 |
| `train.py` | 학습 스크립트 |
| `prepare_data.py` | 데이터 전처리 |
| `quantize.py` | INT8 양자화 (ESP32 최적화) |
| `bpm_test.py` | BPM 계산 테스트 |
