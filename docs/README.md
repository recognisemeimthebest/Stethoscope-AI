# Docs — 시각화 자료

모델 아키텍처, 성능 비교, 전처리 파이프라인을 설명하는 이미지와 생성 스크립트입니다.

---

## images/ — 시각화 이미지

### 심음 분���

| 이미지 | 설명 |
|--------|------|
| `resnet_cbam_pipeline.png` | ResNet+CBAM+XGBoost 전체 파이프라인 |
| `cbam_attention_map.png` | CBAM 어텐션 맵 시각화 |
| `cbam_effect_comparison.png` | CBAM 적용 전/후 비교 |
| `cbam_vs_resnet_cbam.png` | 기본 ResNet vs ResNet+CBAM 성능 비교 |
| `resnet_residual_power.png` | ResNet 잔차 연결의 효과 |
| `mel_spectrogram_comparison.png` | 멜 스펙트로그램 버전별 비교 |
| `v8_v17_mel_comparison.png` | v8 vs v17 전처�� 차이 |
| `v8_v17_improvement.png` | v8→v17 성능 향상 요약 |

### 디노이즈

| 이미지 | 설명 |
|--------|------|
| `denoise_all_comparison.png` | 전체 모델 비교 (CRNN, LUNet, TUNet) |
| `denoise_LUNet_v1.png` | LU-Net v1 결과 |
| `denoise_LUNet_v2.png` | LU-Net v2 결과 |
| `denoise_TUNet_v1.png` | TU-Net v1 결과 |
| `denoise_TUNet_v2.png` | TU-Net v2 결과 (최종) |
| `denoise_v1_performance.png` | CRNN v1 성능 |

---

## scripts/ — 이미지 생성 스크립트

| 스크립트 | 생성 이미지 |
|----------|------------|
| `generate_resnet_cbam_viz.py` | ResNet+CBAM 파이프라인 |
| `generate_cbam_effect.py` | CBAM 효과 비교 |
| `generate_cbam_vs_resnet_cbam.py` | ResNet vs ResNet+CBAM |
| `generate_resnet_power.py` | 잔차 연결 효과 |
| `generate_mel_comparison.py` | 멜 스펙트로그램 비교 |
| `generate_v8_v17_mel.py` | v8/v17 전처리 비교 |
| `generate_v8_v17_why.py` | v8→v17 개선 이유 |
| `generate_denoise_all.py` | 전체 디노이즈 비교 |
| `generate_denoise_viz.py` | 디노이즈 성능 차트 |
| `generate_denoise_lunet_v1.py` | LU-Net v1 시각화 |
