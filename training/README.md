# Training — 모델 개선 과정

AI 청진기 프로젝트의 ML 모델 학습 및 개선 이력입니다.  
각 폴더는 독립적인 실험 영역이며, 최종 추론 코드는 [`server/`](../server/)에 있습니다.

---

## heart/ — 심음 분류 (Normal / Abnormal / Unknown)

### 아키텍처: ResNet+CBAM + XGBoost Ensemble

| 버전 | MACC | 주요 변경 |
|------|------|----------|
| **v1** | 52% | ResNet-18 + ImageNet pretrain, 1kHz, 224×224 mel, 2-class |
| **v8** | 78% | 5초 슬라이딩 윈도우 (2초 stride), 64×64 mel, CBAM 어텐션 추가, XGBoost 앙상블 도입 |
| **v14-v15** | ~95% ❌ | 3-class (Unknown 추가, ESC-50), **그러나 데이터 누수 발견** — 세그먼트 랜덤 분할로 같은 원본의 겹치는 구간이 train/val에 동시 포함 |
| **v16** | 82.4% | **파일 단위 분할**로 누수 제거, 4kHz 리샘플링, SpecAugment |
| **v17 (최종)** | **84.1%** | PeakNorm 전처리, ICS43434 디바이스 노이즈 + BLE 녹음 Unknown 데이터 보강 |

**v14→v16에서 MACC가 떨어진 이유:** 데이터 누수를 수정했기 때문입니다. v14-v15의 95%는 오버랩된 윈도우가 train/val에 동시에 들어가서 발생한 허수였고, v16부터 파일 단위 분할로 정직한 평가를 시작했습니다.

### 주요 파일

| 파일 | 역할 |
|------|------|
| `preprocessing.py` | WAV → 5초 윈도우 → 64-mel 스펙트로그램 |
| `CBAM+RESNET+XG.py` | ���종 아키텍처 (ResNet+CBAM → 256dim → XGBoost) |
| `XGBClassifier.py` | 초기 XGBoost 단독 버전 |
| `XGBClassifier_Full.py` | ���체 파이프라인 통합 학습 |
| `external_data.py` | 소아 데이터셋 통합 |
| `check_overfitting.py` | v14-v15 데이터 누수 검증 |
| `compare_v14_v15.py` | 누수 전/후 성능 비교 |
| `generate_v1_v8_v17_comparison.py` | 버전별 성능 시각화 |
| `generate_macc_evolution_v17.py` | MACC 진화 그래프 |

---

## lung/ — 폐음 분류 (Crackle / Normal / Unknown / Wheeze)

### 아키텍처 진화

| 단계 | 모델 | 해상도 | 손실함수 | 비고 |
|------|------|--------|---------|------|
| v1 | MobileNetV2 | 128×128 | CrossEntropy | 기본 baseline |
| v2 | MobileNetV2 | **224×224** | **Focal Loss** (γ=2) | 해상도 증가로 Crackle 세밀 패턴 포착 |
| v3 | ResNet18 + RFN | 224×224 | Focal Loss | **RFN**(주파수 정규화) 추가, SpecAugment + Mixup |
| v4 | ResNet18 + CBAM + RFN | 224×224 | Focal Loss | CBAM 어텐션 결합 |
| Frame | FrameCNN | 64×41 | Focal Loss | 100ms 단위 세밀 분석 실험 |

**최종 선택: ResNet18 + RFN** — 앙상블(ResNet18+MobileNetV2)이 단일 모델 대비 2-3%p 향상되었으나, 라즈베리파이 추론 속도를 고려해 ResNet18+RFN 단일 모델 채택.

### 주요 파일

| 파일 | 역할 |
|------|------|
| `data_balance.py` | 클래스 균형 맞추기 (Normal 1000개 cap, 희소 클래스 augment) |
| `train_MobileNetV2.py` | v1 baseline 학습 |
| `focal224.py` | v2 Focal Loss + 224 학습 (주력) |
| `train_variants.py` | ResNet18/CBAM/RFN 변형 비교 실험 |
| `cross_eval.py` | ���차 검증 |
| `full_eval.py` | SPRSound/ICBHI 외부 데이터 평가 |
| `pipeline_eval.py` | 전체 파이프라인 통합 평가 |
| `external.py` | 외부 데이터셋 변환 |
| `frame_train.py` | 100ms 프레임 단위 학습 실험 |

---

## denoise/ — 심음 노이즈 제거 (CRNN)

Conv1D + Bidirectional LSTM 기반 디��이즈 모델.  
합성 데이터 10,000쌍 (clean + noisy)으로 학습.

| 파일 | 역할 |
|------|------|
| `train1.py` | 기본 CRNN 학습 (Conv1D → BiLSTM → Dense) |
| `train_masking_crnn.py` | 마스킹 방식 변형 (0-1 마스크 예측) |
| `engine.py` | 학습 엔진 (epoch loop, validation) |
| `generate_full_dataset.py` | 합성 노이즈 데이터셋 생성 |
| `evaluate_model.py` | 복원율/SNR 평가 |

---

## denoise_lunet/ — LU-Net / TU-Net 디노이즈 실험

CRNN 대비 더 깊은 아키텍처를 탐색한 실험들입니다.

### 모델 비교

| 모델 | SNR 개선 | SI-SNR | 특징 |
|------|---------|--------|------|
| CRNN (baseline) | +5.22 dB | — | 스펙트로그램 도메인, 복원율 54.1% |
| **LU-Net v1** | +5.94 dB | 9.82 | 5단 U-Net + BiLSTM skip, 시간 도메인 직접 처리 |
| **LU-Net v2** | +5.59 dB | **12.76** | SE 어텐션 추가, 잔차 학습 (노이즈 예측), SI-SNR 손실 |
| TU-Net v1 | — | — | BiLSTM → TCN(dilated conv) 교체, 구조 동일 |
| TU-Net v2 | 비슷 | 비슷 | LU-Net v2�� 동일 개선, **2-3배 빠른 추론** |

**최종 선택: TU-Net v2** — LU-Net v2와 비슷한 품질이지만 CPU 추론 속도가 2-3배 빠름. TFLite 변환 후 라즈베리파이에서 실시간 처리 가능.

### 주요 파일

| 파일 | 역할 |
|------|------|
| `train lunet.py` | LU-Net v1 학습 |
| `train lunet v2.py` | LU-Net v2 학습 (SE + 잔차 + SI-SNR) |
| `train tunet v1.py` | TU-Net v1 학습 |
| `train tunet v2.py` | TU-Net v2 학습 (최종 채택) |
| `benchmark_speed.py` | 모델별 추론 속도 벤치마크 |
| `restoration rate.py` | 복원율 측정 |
| `generate dataset.py` | 합성 노이즈 데이터셋 생성 |

---

## bpm/ — 온디바이스 BPM 검출

ESP32에서 직접 심박수를 계산하기 위한 경량 모델 파이프라인.

| 파일 | 역할 |
|------|------|
| `01_shannon_labeling.py` | Shannon 에너지 기반 심음 피크 자동 라벨링 |
| `01b_verify_labels.py` | 라벨 검증 |
| `02_train_beat_detector.py` | beat detector 학습 |
| `02b_train_tf.py` | TensorFlow 버전 학습 |
| `03_convert_tflite.py` | TFLite 변환 |
| `03_export_c_weights.py` | C 헤더 형태로 가중치 추출 (ESP32용) |
| `04_evaluate_bpm.py` | BPM 정확도 평가 |
| `model.py` | 경량 CNN 모델 정의 |
| `quantize.py` | INT8 양자화 |

---

## lunet_ablation/ — LU-NET Ablation Study

| 파일 | 역할 |
|------|------|
| `evaluate ablation.py` | LU-Net 구성 요소별 기여도 분석 |
