# Server — AI 추론 서버 (Production)

Raspberry Pi 5에서 실행되는 최종 추론 코드입니다.  
학습 과정은 [`training/`](../training/)을 참고하세요.

---

## 최종 모델 성능

### 심음 분류 — ResNet+CBAM + XGBoost (v17)

3-class: **Normal / Abnormal / Unknown**

| 지표 | Normal | Abnormal | Unknown | 평균 |
|------|--------|----------|---------|------|
| Precision | 82% | 85% | 100% | 89% |
| Recall | 86% | 83% | 77% | 82% |
| F1 | 84% | 84% | 87% | 85% |

- **MACC: 84.1%** (파일 단위 분할, 데이터 누수 없음)
- 전처리: 4kHz, 5초 윈도우(2초 stride), 64-mel spectrogram, PeakNorm
- Unknown 임계값: 30% 이상이면 Unknown 판정
- 모델 크기: ResNet 4.8MB + XGBoost 709KB

### 호흡음 분류 — ResNet18 + RFN (v3)

4-class: **Crackle / Normal / Unknown / Wheeze**

| 설정 | 정확도 |
|------|--------|
| Cross-validation | ~91-93% |
| 앙상블 (ResNet18+MobileNetV2) | +2-3%p |
| 최종 단일 모델 (ResNet18+RFN) | ~91% |

- 전처리: 8kHz, 2초 고정, 128-mel PCEN, 3채널 (mel + delta + delta²)
- RFN (Relaxed Frequency Normalization): 주파수 축 정규화
- Unknown 임계값: 30% 이상이면 Unknown 판정
- Focal Loss (γ=2.0) 로 희소 클래스 학습 강화

### 심음 디노이즈 — TU-Net v2 (TFLite)

| 지표 | 값 |
|------|-----|
| SNR 개선 | ~+5.5 dB |
| SI-SNR | ~12.7 |
| 추론 속도 | CRNN 대비 2-3x 빠름 |
| 모델 형식 | TFLite (INT8 호환) |

- 입력: 4kHz, 5초 청크, overlap-add 재조합
- 아키텍처: TCN U-Net + SE attention + residual learning
- 손실함수: 90% MSE + 10% SI-SNR

---

## 파일 구성

| 파일 | 역할 | 의존 모델 |
|------|------|----------|
| `server.py` | MQTT 수신 → 추론 → 결과 발행 | 아래 3개 모듈 |
| `predict_heart.py` | 심음 3-class 분류 | `resnet_cbam_best.pth` + `heart_xgb_model.json` |
| `predict_lung.py` | 호흡음 4-class 분류 | `stetho_resnet18_lung.pth` |
| `predict_denoise.py` | 심음 노이즈 제거 | `tunet_v2_best.tflite` |
| `convert_to_tflite.py` | PyTorch → TFLite 변환 유틸 | — |

---

## 추론 파이프라인

```
WAV 수신 (MQTT)
    │
    ├─ HEART ──→ TU-Net v2 디노이즈 ──→ ResNet+CBAM feature ──→ XGBoost 분류
    │                                                              ↓
    │                                         Normal / Abnormal / Unknown
    │
    └─ LUNG ───→ 8kHz PCEN mel+delta ──→ ResNet18+RFN 분류
                                              ↓
                                  Crackle / Normal / Unknown / Wheeze
```

### Unknown 판정 로직

두 모델 모두 동일한 방식:
1. Unknown 확률 ≥ 30% → **무조건 Unknown** (argmax 무시)
2. Unknown 확률 < 30% → Unknown 제외하고 나머지 클래스 중 argmax

Unknown으로 판정되면 → BLE로 `AI:{TYPE}:RECHECK:0` 전송 → ESP32에서 "다시 청진해 주세요" 표시

---

## 실행 환경

```bash
# Raspberry Pi 5 (server_models/ 에 모델 가중치 배치 필요)
cd ~/Desktop/audio_records
python server.py
```

필수 패키지:
```
torch torchvision librosa soundfile scipy numpy
scikit-learn xgboost paho-mqtt tflite-runtime
```
