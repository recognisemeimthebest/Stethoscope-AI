# Heart Evolution — 심음 모델 버전별 성능 비교

v1 → v8 → v17로 이어지는 심음 분류 모델의 개선 과정을 시각화한 자료입니다.

---

## 코드 → 이미지 매핑

| 스크립트 | 생성 이미지 | 설명 |
|----------|------------|------|
| `generate_mel_comparison.py` | `mel_spectrogram_comparison.png` | 버전별 멜 스펙트로그램 전처리 차이 |
| `generate_v8_v17_mel.py` | `v8_v17_mel_comparison.png` | v8(64x64) vs v17(64x79) 스펙트로그램 비교 |
| `generate_v8_v17_why.py` | `v8_v17_improvement.png` | v8→v17에서 무엇이 바뀌었는지 요약 |

---

## 핵심 변경사항

| 항목 | v8 | v17 |
|------|-----|------|
| 샘플링 | 1kHz | 4kHz |
| 멜 크기 | 64×64 | 64×79 |
| 정규화 | 없음 | PeakNorm |
| 데이터 분할 | 세그먼트 랜덤 | 파일 단위 |
| Unknown 데이터 | ESC-50만 | +ICS43434 노이즈 +BLE 녹음 |
| **MACC** | **78%** | **84.1%** |
