# Denoise — CRNN 심음 노이즈 제거

Conv1D + Bidirectional LSTM 기반 디노이즈 모델입니다.  
합성 데이터 10,000쌍(clean + noisy)으로 학습했습니다.

---

## 아키텍처

```
입력 (5초, 4kHz = 20000 samples)
    ↓
Conv1D Feature Extraction
    ↓
Bidirectional LSTM (128→64 units)
    ↓
Dense → 출력 (denoised waveform)
```

- **Masking 변형**: 0-1 마스크를 예측해 원본에 곱하는 방식도 실험
- 복원율: ~54%, SNR 개선: +5.22 dB

> 이후 더 깊은 구조(LU-Net, TU-Net)를 [`denoise_lunet/`](../denoise_lunet/)에서 실험하여 최종 TU-Net v2를 채택했습니다.

---

## 파일 설명

### 학습
| 파일 | 역할 |
|------|------|
| `train1.py` | 기본 CRNN 학습 (Conv1D → BiLSTM → Dense) |
| `train_masking_crnn.py` | 마스킹 방식 변형 학습 |
| `engine.py` | 학습 엔진 (epoch loop, validation, early stopping) |

### 데이터
| 파일 | 역할 |
|------|------|
| `generate_full_dataset.py` | 합성 노이즈 데이터셋 10,000쌍 생성 |
| `X_Y.py` | 입출력 데이터 전처리 |
| `dataset_info.py` | 데이터셋 통계 확인 |
| `dataset_info.json` | 데이터셋 메타 정보 |

### 평가
| 파일 | 역할 |
|------|------|
| `evaluate_model.py` | 복원율/SNR 평가 |
| `denoising_test.py` | 디노이즈 결과 청취 테스트 |
| `preprocessed_mix_test.py` | 전처리 혼합 테스트 |
| `convert_image.py` | 스펙트로그램 이미지 변환 |

### 시각화
| 파일 | 역할 |
|------|------|
| `generate_denoise_charts.py` | 디노이즈 성능 차트 |
| `generate_report.py` | 학습 리포트 생성 |
