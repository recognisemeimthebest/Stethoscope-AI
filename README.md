# Stethoscope-AI

> **AI 기반 전자 청진기 — 간호학과 학생이 설계하고 개발한 임상 보조 시스템**

심장음/호흡음을 실시간 수집하고, BLE로 전송하여 AI가 분류합니다.  
ESP32 펌웨어 + 모바일 웹 브릿지 + Raspberry Pi ML 서버의 3계층 구조입니다.

---

## 시스템 구조

```
┌─────────────┐    BLE     ┌──────────────┐   MQTT(WS)   ┌──────────────┐
│  ESP32 +    │ ────────── │  Web Bridge  │ ──────────── │ Raspberry Pi │
│  ICS43434   │  Audio/Cmd │  (Chrome)    │   WAV 전송   │  AI Server   │
│  TFT LCD    │ <───────── │              │ <──────────── │  (Python)    │
└─────────────┘  AI Result └──────────────┘   AI Result  └──────────────┘
```

---

## 프로젝트 구조

```
Stethoscope-AI/
│
├── firmware/          ESP32 펌웨어 — BLE 오디오 전송, TFT LCD UI
├── web/               웹 블루투스 브릿지 — BLE↔MQTT 중계
├── server/            ML 추론 서버 — 심음/폐음 분류, 디노이즈
│
├── training/          모델 학습 코드 (개선 과정 포함)
│   ├── heart/         심음 분류 v1→v17 (52%→84.1% MACC)
│   ├── lung/          폐음 분류 MobileNetV2→ResNet18+RFN
│   ├── denoise/       CRNN 디노이즈 학습
│   ├── denoise_lunet/ LU-Net/TU-Net 디노이즈 실험
│   ├── bpm/           온디바이스 BPM 파이프라인
│   └── lunet_ablation/LU-NET 어블레이션 스터디
│
├── docs/              시각화 자료
│   ├── images/        모델 비교/분석 이미지
│   └── scripts/       이미지 생성 Python 스크립트
│
├── demo/              샘플 오디오 + 디노이즈 데모
├── final_submission/  최종 제출 코드 + 실행 가이드
│
├── CLAUDE.md          프로젝트 기술 명세
└── CHANGELOG.txt      변경 이력
```

---

## AI 모델 성능 (최종)

| 모델 | 클래스 | 성능 | 아키텍처 |
|------|--------|------|----------|
| **심음 분류** | Normal / Abnormal / Unknown | MACC 84.1% | ResNet+CBAM + XGBoost |
| **폐음 분류** | Crackle / Normal / Unknown / Wheeze | ~91% acc | ResNet18 + RFN |
| **심음 디노이즈** | — | +5.5dB SNR | TU-Net v2 (TFLite) |

자세한 성능 지표는 [server/README.md](server/README.md),  
개선 과정은 [training/README.md](training/README.md)를 참고하세요.

---

## 빠른 시작

### 1. AI 서버 (Raspberry Pi 5)
```bash
cd ~/Desktop/audio_records
python server.py
```

### 2. 펌웨어 (ESP32)
Arduino IDE에서 `firmware/2026-04-05-2130.h` 업로드

### 3. 웹 브릿지 (Chrome)
`web/2026-03-26.html`을 로컬 서버로 열기

> 상세 실행 방법은 [final_submission/README.md](final_submission/README.md)를 참고하세요.

---

## 기술 스택

| 계층 | 기술 |
|------|------|
| **Hardware** | ESP32 + ICS43434 I2S MEMS Mic + ST7789 TFT LCD |
| **통신** | BLE 5.0 + MQTT over WebSocket |
| **ML** | PyTorch, XGBoost, TFLite, librosa |
| **서버** | Python 3.11+, Raspberry Pi 5 |
| **개발 도구** | Claude Code |

---

## 개발자

**간호학과 학생 개발자** — 임상 지식과 기술을 연결하는 다리를 만들고 있습니다.

---

## License

MIT License
