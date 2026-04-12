# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered electronic stethoscope system built on ESP32 with ICS43434 I2S microphone. Captures heart/lung sounds, transmits audio over BLE to a mobile web bridge, runs ML inference (ResNet+XGBoost / MobileNetV2) on a Raspberry Pi 5 backend, and displays results on TFT LCD.

---

## AUTHORITATIVE FILE VERSIONS

항상 이 파일들을 기준으로 작업할 것. 타임스탬프가 오래된 파일들은 구버전임.

| Layer | 현재 파일 | 줄 수 |
|-------|----------|------|
| **Firmware** | `stethoscope-software/2026-04-05-2130.h` | 1880줄 |
| **Web Bridge** | `Stetho_Web/2026-03-26.html` | ~1236줄 |
| **Server** | `server.py` | 344줄 |
| **Heart 추론** | `server_models/predict_heart.py` | 189줄 |
| **Lung 추론** | `server_models/predict_lung.py` | 121줄 |
| **Denoise 추론** | `server_models/predict_denoise.py` | 150줄 |

---

## Architecture

3-layer 시스템:

**1. Hardware Firmware** (`stethoscope-software/`)
- Arduino C++, ESP32 타겟, JSF++ 규칙 준수 (constexpr, nullptr, snprintf 등)
- 최신 파일: `2026-03-26-2310.h` (1865줄) — 구버전인 `2026-03-25-1442.h` (652줄)과 혼동 금지
- ICS43434 I2S 마이크: 16kHz 입력 → 8kHz 데시메이션 후 BLE 전송
- TFT LCD 디스플레이 (135×240, TFT_eSPI 라이브러리)
- Key pins: I2S_WS=25, I2S_SD=33, I2S_SCK=26, BTN_L_MODE=35, BTN_R_RECORD=0, TFT_BL=4

**2. Mobile Web Bridge** (`Stetho_Web/`)
- `2026-03-26.html`: Vanilla JS + Web Bluetooth API + MQTT.js 4.3.7
- MQTT 브로커: `ws://114.206.167.99:9001` (병원 내부망 — 하드코딩됨)
- BLE로부터 받은 오디오를 WAV(8kHz, 16-bit, mono)로 변환 → MQTT 업로드
- AI 결과를 MQTT에서 받아 BLE로 `AI:HEART:Normal:87` 형식으로 전송

**3. Python ML Server** (Raspberry Pi 5)
- `server.py` (344줄): MQTT 수신 → AI 추론 → 결과 발행
- 실행 경로: `/home/leejongwan/Desktop/audio_records/`
- EMR CSV: `patients_emr.csv` (patient_id, name, age, gender, ward, status, audio_path, ai_result)

---

## BLE UUIDs

```
Service:  19b10000-e8f2-537e-4f6c-d104768a1214
Audio:    19b10001-e8f2-537e-4f6c-d104768a1214
Command:  19b10002-e8f2-537e-4f6c-d104768a1214
```

---

## BLE Protocol

**Firmware → Web (Notify):**
- 오디오: 80 samples × int16 raw binary (8kHz)
- `REC_DONE` — 5초 녹음 정상 완료
- `REC_FAIL:DISCONNECTED` — BLE 끊김으로 녹음 실패 (주의: 접두사로 비교해야 함, `===` 사용 금지)
- `PAT:P001|김철수|301호,P002|이영희|302호,...` — 환자 목록

**Web → Firmware (Write):**
- `AI:HEART:Normal:87` — AI 결과 (scan_type:label:confidence%)
- `AI:LUNG:RECHECK:0` — Unknown 판정 시 (confidence=0)
- label 값: `"Normal"`, `"Abnormal"`, `"RECHECK"`, 또는 폐음 클래스명

---

## Firmware State Machine (`2026-03-26-2310.h`)

```cpp
enum AppState { ACTIVE, SLEEP, POPUP, PAIRING, AI_RESULT };
enum MeasureMode { MODE_HEART=1, MODE_LUNG=2, MODE_PATIENT=3 };
```

**AI 결과 관련 핵심 로직:**
- 녹음 완료(`stopRecording()`) → `isWaitingAI=true`, 점 애니메이션 시작
- `". / .. / ..."` 애니메이션 (500ms 간격, 노란색) — HEART/LUNG 화면 환자번호 아래
- BLE로 `AI:` 수신 → `aiResultPending=true` → loop()에서 `changeState(AI_RESULT)`
- `AI_RESULT` 상태: 전체화면 팝업 (버튼 L 또는 R로 닫기)
  - Normal=초록, Abnormal=빨강, RECHECK=주황 + "다시 청진해 주세요"
- 30초 타임아웃 시 자동으로 RECHECK 팝업 (`AI_WAIT_TIMEOUT_MS=30000`)
- 팝업 닫힌 후: `[Normal 87%]` 미니 표시 (환자번호 아래)
- `isWaitingAI` 중에는 파형/BPM/RR 그리기 중단

**주요 상수:**
```cpp
static constexpr int SAMPLE_RATE     = 16000;   // I2S 입력
static constexpr int TARGET_RATE     = 8000;    // BLE 전송 (DECIMATION=2)
static constexpr int BLE_BUFFER_SIZE = 80;      // 80 samples per BLE packet
static constexpr unsigned long RECORD_DURATION_MS = 5000UL; // 5초 녹음
static constexpr unsigned long AI_WAIT_TIMEOUT_MS = 30000UL; // 30초 타임아웃
static constexpr unsigned long AI_DOT_ANIM_MS     = 500UL;
static constexpr int DSP_PEAK_REFRACTORY = 2400; // 심음 피크 간격 (@ 8kHz)
static constexpr int RR_REFRACTORY       = 12000; // 호흡 감지 간격
```

**NVS 저장:** currentMode, selectedPatient, isBleOn (재부팅 후 복원)
**WDT:** 10초 timeout (audioTask, loop() 등록)

---

## MQTT Topics

**Server 구독:**
- `stethoscope/wav/{patient_id}/{HEART|LUNG}` — WAV 바이너리 수신 → AI 분석
- `stethoscope/req_patients` — 대기 환자 목록 요청
- `stethoscope/req_all_patients` — 전체 환자 EMR 요청
- `stethoscope/log` — 웹 로그 출력

**Server 발행:**
- `stethoscope/result` — AI 결과 JSON
- `stethoscope/denoised/{patient_id}/HEART` — 디노이즈 WAV 바이너리
- `stethoscope/res_patients` — 대기 환자 목록 JSON 배열
- `stethoscope/res_all_patients` — 전체 환자 EMR JSON (age, gender, diagnosis, status, ai_result 포함)

---

## AI Result JSON (server → web via MQTT)

```json
{
  "patient_id": "P001",
  "patient_name": "김철수",
  "scan_type": "HEART",
  "label": "Normal",
  "prob_normal": 85.32,
  "prob_abnormal": 12.45,
  "prob_unknown": 2.23,
  "segments": 4,
  "denoised": true,
  "timestamp": "20260326_153422",

  // Unknown 판정 시 추가 필드:
  "action": "RE_AUSCULTATE",
  "message": "유효한 청진음이 감지되지 않았습니다. 다시 청진해주세요."
}
```

**폐음 결과:**
```json
{
  "label": "Normal",
  "probs": {
    "Complex": 5.12, "Crackle": 8.34, "Normal": 78.92,
    "Stridor": 3.21, "Unknown": 2.50, "Wheeze": 1.91
  }
}
```

---

## ML Models

### Heart Sound (3-class)
- **클래스:** Normal(0), Abnormal(1), Unknown(2)
- **Unknown 데이터:** ESC-50 환경음 2000개 사용
- **아키텍처:** ResNet+CBAM(256-dim feature) + XGBoost 앙상블
- **전처리:** 4kHz 리샘플링, 5초 윈도우 2초 스트라이드, 64-mel 스펙트로그램
- **모델 파일:**
  - `Heart_binary_classification/resnet_cbam_best.pth` (4.8MB) — 학습용
  - `Heart_binary_classification/heart_xgb_model.json` (709KB) — 학습용
  - `server_models/Heart_binary_classification/` — 서버 추론용 사본
- **추론:** `server_models/predict_heart.py` → `load_models()`, `predict(wav_path, resnet, xgb)`

### Lung Sound (6-class)
- **클래스:** Complex, Crackle, Normal, Stridor, Unknown, Wheeze
- **Unknown 데이터:** ESC-50 환경음 2000개 사용
- **아키텍처:** MobileNetV2 (ImageNet 사전학습) + Focal Loss
- **전처리:** 16kHz 리샘플링, 2초 고정 윈도우, 224×224 mel-spectrogram PNG (magma colormap)
- **모델 파일:**
  - `Lung_classification/stetho_mobilenetv2_224_focal.pth` (8.8MB) — 학습용
  - `server_models/Lung_classification/` — 서버 추론용 사본
- **추론:** `server_models/predict_lung.py` → `load_model()`, `predict(wav_path, model)`

### Denoising (Heart only)
- **아키텍처:** Conv1D + Bidirectional LSTM (CRNN), TFLite 변환
- **학습 데이터:** 합성 10,000 샘플 (`LSTM_first/Dataset_10000/`)
- **입력/출력:** 5초 윈도우 (20000 samples @ 4kHz), overlap-add 방식
- **모델 파일:** `server_models/tunet_v2_best.tflite` (TFLite)
- **추론:** `server_models/predict_denoise.py` → `load_model()`, `denoise(wav_path, model)`

---

## Web Bridge 주요 JS 함수 (`2026-03-26.html`)

| 함수 | 역할 |
|------|------|
| `connectMQTT()` | 병원 MQTT 브로커 연결 |
| `connectBLE()` | ESP32 BLE 스캔·연결 |
| `startRec(pat, type)` | 5초 녹음 시작 |
| `stopRec()` | 녹음 완료 → WAV 컴파일 |
| `parseBleCommand(data)` | 펌웨어 알림 파싱 (REC_DONE, REC_FAIL, PAT 등) |
| `publishResult(patient_id, scan_type)` | WAV → MQTT 업로드 |
| `handleAiResult(r)` | AI 결과 UI 표시 + BLE로 결과 전송 |
| `sendResultToDevice(r)` | `AI:TYPE:LABEL:CONF` 포맷으로 BLE Write |
| `render()` | Canvas 파형 애니메이션 루프 |

**Unknown/RECHECK 처리 (웹):**
- `r.label==='Unknown'` 또는 `r.action==='RE_AUSCULTATE'` → 노란색 경고 배너, 재청진 지시
- BLE 전송: `AI:${r.scan_type}:RECHECK:0`
- EMR 상태는 "대기중" 유지 (서버에서 update_emr 호출 안 함)

---

## Training Workflow

```bash
# 학습 전: G:\stetho_ai\_misc\datasets\ 경로 데이터 확인 필요
# 직접 Python 경로 사용 (conda env 이슈 방지):
C:/Users/dwd00/anaconda3/envs/stetho_ai/python.exe

# 심음 분류
cd Heart_binary_classification/
python preprocessing.py       # x_data.npy, y_data.npy 생성
python XGBClassifier_Full.py  # 모델 학습 및 저장

# 폐음 분류
cd Lung_classification/
python data_balance.py        # 스펙트로그램 PNG 생성 (ESC-50 Unknown 포함)
python focal224.py            # MobileNetV2 학습 (50 epochs, focal loss)

# 디노이즈
cd LSTM_first/
python train1.py              # CRNN 학습

# 학습 후: 모델 파일을 server_models/ 하위 폴더에 복사
```

**주요 데이터 경로 (Windows 개발 PC):**
- 심음 데이터: `G:\stetho_ai\_misc\datasets\classification-of-heart-sound-recordings\`
- 소아 데이터: `G:\stetho_ai\_misc\datasets\pediatric\`
- 폐음 데이터: `G:\stetho_ai\_misc\lung\classification\LUNG_SOUND\`
- ESC-50: `G:\stetho_ai\_misc\datasets\ESC-50-master\ESC-50-master\audio\`
- 학습용 스펙트로그램: `G:\stetho_ai\_misc\lung\classification\LUNG_SOUND\Spectrograms_Balanced\`

---

## Python Dependencies

```
torch torchvision tensorflow librosa soundfile scipy numpy pandas
scikit-learn xgboost tqdm paho-mqtt matplotlib pillow tflite-runtime
```

---

## Hardware Build

- Arduino IDE 또는 PlatformIO, ESP32 타겟
- Required libraries: `TFT_eSPI`, `ESP32 BLE Arduino`, I2S driver
- 펌웨어 파일은 `.h` 헤더 형식, 타임스탬프로 버전 관리
- **신규 작업 시 반드시 최신 타임스탬프 파일 기준으로 작성**

---

## Key Notes

- 펌웨어 파일명 규칙: `YYYY-MM-DD-HHMM.h` — 가장 큰 타임스탬프가 현행 버전
- MQTT 서버 주소·환자 데이터 포맷은 3개 레이어(펌웨어·웹·서버)가 강하게 결합됨 — 변경 시 3곳 동시 수정
- WAV 포맷: 8kHz, 16-bit, mono (2026-03-26 2kHz→8kHz 통일 완료)
- 테스트용 샘플 오디오: `sample_sound/raw_heart_sound.wav`
- `REC_FAIL` 비교는 반드시 `.startsWith("REC_FAIL")` 사용 (펌웨어가 `REC_FAIL:DISCONNECTED` 전송)
- conda 환경 대신 직접 Python 경로 사용: `C:/Users/dwd00/anaconda3/envs/stetho_ai/python.exe`

---

# Git Commit Rules

- 파일 변경/추가 시 자동으로 커밋하지 말 것
- 변경사항을 분석해서 커밋 메시지를 먼저 보여줄 것
- 커밋 메시지 형식:
  - Summary: 한 줄 요약 (영문, 50자 이내)
  - Description: 변경 내용 bullet point로 정리
- 내가 "커밋해" 또는 "푸시해"라고 할 때만 실행할 것
- conventional commit 형식 사용 (feat:, fix:, docs: 등)

# Change Log Rules

- 파일을 생성/수정/삭제할 때마다 `G:\stetho_ai\CHANGELOG.txt`에 변경 내역을 **추가(append)** 할 것
- 파일을 새로 만들지 말고, 하나의 `CHANGELOG.txt`에 계속 기록할 것
- 기록 형식:
  ```
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [YYYY-MM-DD HH:MM] 작업 제목
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  파일: 변경된 파일 경로 (여러개면 줄바꿈)
  유형: 생성 | 수정 | 삭제
  내용:
    - 변경/개선 내용 bullet point
  ```
- 한 작업에서 여러 파일이 변경되면 하나의 블록에 모아서 기록
- 이전에 로그를 남기지 않은 변경사항도 소급하여 기록할 것
