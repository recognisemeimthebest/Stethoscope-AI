# Denoise Comparison — 디노이즈 모델 비교

CRNN, LU-Net, TU-Net 각 버전의 디노이즈 성능을 비교한 자료입니다.

---

## 코드 → 이미지 매핑

| 스크립트 | 생성 이미지 | 설명 |
|----------|------------|------|
| `generate_denoise_all.py` | `denoise_all_comparison.png` | 전체 모델 파형 비교 (한 장 요약) |
| `generate_denoise_all.py` | `denoise_LUNet_v2.png` | LU-Net v2 개별 결과 |
| `generate_denoise_all.py` | `denoise_TUNet_v1.png` | TU-Net v1 개별 결과 |
| `generate_denoise_all.py` | `denoise_TUNet_v2.png` | TU-Net v2 개별 결과 (최종 채택) |
| `generate_denoise_lunet_v1.py` | `denoise_LUNet_v1.png` | LU-Net v1 개별 결과 |
| `generate_denoise_viz.py` | `denoise_v1_performance.png` | CRNN v1 성능 차트 |

---

## 모델별 성능

| 모델 | SNR 개선 | SI-SNR | 비고 |
|------|---------|--------|------|
| CRNN v1 | +5.22 dB | — | 스펙트로그램 도메인 |
| LU-Net v1 | +5.94 dB | 9.82 | 시간 도메인 U-Net+BiLSTM |
| LU-Net v2 | +5.59 dB | 12.76 | +SE attention, 잔차 학습 |
| TU-Net v1 | ~비슷 | ~비슷 | BiLSTM→TCN 교체 |
| **TU-Net v2** | **~+5.5 dB** | **~12.7** | **최종 채택 — 2-3배 빠름** |

> 청취 비교는 [`demo/`](../../demo/)에서 WAV 파일로 확인 가능합니다.
