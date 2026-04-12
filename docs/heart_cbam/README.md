# Heart CBAM — 어텐션 메커니즘 시각화

ResNet+CBAM 아키텍처의 어텐션 효과를 분석한 자료입니다.

---

## 코드 → 이미지 매핑

| 스크립트 | 생성 이미지 | 설명 |
|----------|------------|------|
| `generate_resnet_cbam_viz.py` | `cbam_attention_map.png` | CBAM이 멜 스펙트로그램에서 주목하는 영역 |
| `generate_resnet_cbam_viz.py` | `resnet_cbam_pipeline.png` | ResNet+CBAM+XGBoost 전체 파이프라인 구조도 |
| `generate_cbam_effect.py` | `cbam_effect_comparison.png` | CBAM 적용 전/후 특징 맵 비교 |
| `generate_cbam_vs_resnet_cbam.py` | `cbam_vs_resnet_cbam.png` | 기본 ResNet vs ResNet+CBAM 성능 비교 |
| `generate_resnet_power.py` | `resnet_residual_power.png` | 잔차 연결(Residual)이 학습에 미치는 효과 |

---

## 핵심 인사이트

- CBAM은 **S1/S2 심음 피크 영역**에 어텐션을 집중시킴
- Channel Attention: 심음에 중요한 주파수 대역 강조
- Spatial Attention: 시간축에서 심음 이벤트 위치 강조
- CBAM 추가로 MACC **+3-5%p** 향상
