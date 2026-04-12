# Denoise LU-Net/TU-Net — 고급 디노이즈 실험

CRNN보다 깊은 U-Net 기반 아키텍처를 탐색한 실험입니다.  
최종적으로 **TU-Net v2**가 채택되어 서버에 배포되었습니다.

---

## 모델 비교

| 모델 | SNR 개선 | SI-SNR | 속도 | 특징 |
|------|---------|--------|------|------|
| CRNN (baseline) | +5.22 dB | — | 1x | 스펙트로그램 도메인 |
| **LU-Net v1** | +5.94 dB | 9.82 | 1x | 5단 U-Net + BiLSTM skip |
| **LU-Net v2** | +5.59 dB | **12.76** | 1x | +SE attention, 잔차 학습, SI-SNR 손실 |
| TU-Net v1 | ~비슷 | ~비슷 | 2x | BiLSTM → TCN(dilated conv) 교체 |
| **TU-Net v2** | ~+5.5 dB | ~12.7 | **2-3x** | LU-Net v2 개선 + TCN 속도 |

### 왜 TU-Net v2를 선택했나?

- LU-Net v2와 비슷한 품질이지만 **CPU 추론 2-3배 빠름**
- TFLite 변환 후 Raspberry Pi에서 실시간 처리 가능
- SI-SNR 12.7로 파형 충실도 우수

---

## 파일 설명

### 학습
| 파일 | 역할 |
|------|------|
| `train lunet.py` | LU-Net v1 학습 |
| `train lunet v2.py` | LU-Net v2 학습 (SE + 잔차 + SI-SNR) |
| `train tunet v1.py` | TU-Net v1 학습 (TCN 교체) |
| `train tunet v2.py` | TU-Net v2 학습 (최종 채택) |
| `train_lunet_2s.py` | 2초 청크 LU-Net 실험 |

### 데이터
| 파일 | 역할 |
|------|------|
| `generate dataset.py` | 합성 노이즈 데이터셋 생성 (5초) |
| `generate_dataset_2s.py` | 2초 청크 데이터셋 생성 |

### 평가
| 파일 | 역할 |
|------|------|
| `evaluate_model.py` | 모델 평가 |
| `evaluate 2sec.py` | 2초 청크 평가 |
| `evaluate v2.py` | v2 모델 평가 |
| `restoration rate.py` | 복원율 측정 |
| `benchmark_speed.py` | 모델별 추론 속도 벤치마크 |
| `denoise_compare.py` | 모델 간 비교 |
| `denoise compare fixed.py` | 수정된 비교 스크립트 |
| `denoise listen all.py` | 전체 모델 청취 비교 |

### 시각화
| 파일 | 역할 |
|------|------|
| `visualize_architecture.py` | 아키텍처 다이어그램 |
| `visualize_denoise_results.py` | 디노이즈 결과 시각화 |
| `generate_performance_chart.py` | 성능 비교 차트 |
| `generate_preprocessing_pipeline.py` | 전처리 파이프라인 |
| `generate_speed_benchmark.py` | 속도 벤치마크 차트 |
