# Lung — 폐음 분류 모델 (Crackle / Normal / Unknown / Wheeze)

MobileNetV2에서 시작해 ResNet18+RFN까지 진화한 폐음 분류 모델입니다.  
Focal Loss와 RFN(주파수 정규화)이 핵심 개선 포인트입니다.

---

## 아키텍처 진화

```
v1 MobileNetV2       →  v2 +Focal Loss    →  v3 ResNet18+RFN  →  최종 선택
128x128, CE Loss        224x224 해상도 업     +SpecAugment        ResNet18+RFN
기본 baseline           Crackle 세밀 포착     +Mixup              (~91% acc)
                        gamma=2.0             CBAM 실험 포함
```

---

## 파일 설명

### 데이터 전처리
| 파일 | 역할 |
|------|------|
| `data_balance.py` | 클래스 균형화 (Normal 1000개 cap, 희소 클래스 augment) |
| `Data_set.py` | 데이터셋 로더 |
| `external.py` | 외부 데이터셋 (SPRSound, ICBHI) 변환 |
| `merge.py` | 다중 데이터셋 병합 |
| `disting.py` | 클래스 구분 분석 |

### 학습
| 파일 | 역할 |
|------|------|
| `train_MobileNetV2.py` | v1 baseline (128x128, CrossEntropy) |
| `focal224.py` | v2 주력 모델 (224x224, Focal Loss gamma=2) |
| `train_variants.py` | ResNet18/CBAM/RFN 변형 비교 실험 |
| `frame_train.py` | 100ms 프레임 단위 FrameCNN 실험 |
| `frame_data_gen.py` | 프레임 학습용 데이터 생성 |

### 평가
| 파일 | 역할 |
|------|------|
| `cross_eval.py` | 교차 검증 |
| `full_eval.py` | SPRSound/ICBHI 외부 데이터 평가 |
| `pipeline_eval.py` | 전체 파이프라인 통합 평가 |
| `224_focal_test.py` | Focal Loss 모델 테스트 |
| `model_test.py` | 모델 단위 테스트 |

### 시각화
| 파일 | 역할 |
|------|------|
| `visualize_lung_classification.py` | 분류 결과 시각화 |
