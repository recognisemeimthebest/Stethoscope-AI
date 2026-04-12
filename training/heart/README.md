# Heart — 심음 분류 모델 (Normal / Abnormal / Unknown)

ResNet+CBAM + XGBoost 앙상블로 심장음을 3-class 분류합니다.  
v1(52%)에서 v17(84.1%)까지 17번의 반복 개선을 거쳤습니다.

---

## 버전 진화

```
v1 (52%)  →  v8 (78%)  →  v14-15 (95%❌)  →  v16 (82%)  →  v17 (84.1%✅)
 기본        윈도우+CBAM    데이터누수 발견     파일단위분할     PeakNorm+디바이스노이즈
 ResNet-18   +XGBoost앙상블  세그먼트랜덤분할→   누수제거후       Unknown 데이터 보강
 224x224     64x64 mel      오버랩이 train/val   정직한 평가     ESC-50+ICS43434
 2-class     3-class        에 동시 포함                        BLE 녹음
```

---

## 파일 설명

### 학습 코드
| 파일 | 역할 |
|------|------|
| `preprocessing.py` | WAV → 5초 윈도우(2초 stride) → 64-mel 스펙트로그램 |
| `CBAM+RESNET+XG.py` | 최종 아키텍처 학습 (ResNet+CBAM 256dim → XGBoost) |
| `XGBClassifier.py` | 초기 XGBoost 단독 실험 |
| `XGBClassifier_Full.py` | 전체 파이프라인 통합 학습 |
| `external_data.py` | 소아 데이터셋 통합 스크립트 |

### 분석/검증
| 파일 | 역할 |
|------|------|
| `check_overfitting.py` | v14-15 데이터 누수 검증 (세그먼트 오버�� 분석) |
| `compare_v14_v15.py` | 누수 수정 전/후 성능 비교 |

### 시각화
| 파일 | 역할 |
|------|------|
| `generate_v1_v8_v15_comparison.py` | v1/v8/v15 3버전 비교 |
| `generate_v1_v8_v16_comparison.py` | v1/v8/v16 비교 |
| `generate_v1_v8_v17_comparison.py` | v1/v8/v17 최종 비교 |
| `generate_macc_evolution.py` | MACC 진화 그래프 |
| `generate_macc_evolution_v16.py` | v16 포함 진화 그래프 |
| `generate_macc_evolution_v17.py` | v17 최종 진화 그래프 |
| `generate_comparison_tables.py` | 버전별 비교 테이블 생성 |
| `generate_pipeline_diagram.py` | 파이프라인 다이어그램 |
| `generate_v15_charts.py` | v15 성능 차트 |
| `visualize_classification.py` | 분류 결과 시각화 |
| `visualize_raw_comparison.py` | 원본 데이터 비교 |
