# Demo — 샘플 오디오 & 디노이즈 결과

테스트용 심장음 샘플��� 디노이즈 모델별 비교 결과물입니다.

---

## 샘플 심장음

| 파일 | 설명 |
|------|------|
| `raw_heart_sound.wav` | ICS43434로 녹음한 원본 심장음 (8kHz, mono) |
| `raw_heart_sound_x4.wav` | 4배속 재생용 |

---

## 디노이즈 비교

동일한 심장음에 대해 각 모델의 결과를 비교할 수 있습니다.

| 파일 | 모델 | 설명 |
|------|------|------|
| `1_clean.wav` | �� | 원본 clean 심장음 |
| `1_clean_4k.wav` | — | 4kHz 리샘플링 버전 |
| `2_noisy.wav` | — | 노이즈��� 섞�� 입력 |
| `2_noisy_4k.wav` | — | 4kHz 리샘플링 버전 |
| `3_denoised_v1.wav` | CRNN v1 | 기본 CRNN 결과 |
| `3_denoised_lunet_v1.wav` | LU-Net v1 | U-Net+BiLSTM 결과 |
| `4_denoised_lunet_v2.wav` | LU-Net v2 | +SE attention, 잔차 학습 |
| `5_denoised_tunet_v1.wav` | TU-Net v1 | TCN U-Net 결과 |
| `6_denoised_tunet_v2.wav` | **TU-Net v2** | **최종 채택 모델** |

---

## 청취 비교 포인트

1. **2_noisy → 6_denoised (TU-Net v2)**: 최종 모델의 노이즈 제거 효과
2. **3_denoised_v1 → 6_denoised**: CRNN에서 TU-Net v2로의 품질 개선
3. **1_clean vs 6_denoised**: 원본 대비 복원 충실도
