# AI Stethoscope - 실행 가이드

> ESP32 전자 청진기 + BLE + Flutter 앱 + Raspberry Pi AI 서버

---

## 목차

1. [시스템 구성](#시스템-구성)
2. [제출 파일 목록](#제출-파일-목록)
3. [환경 요구사항](#환경-요구사항)
4. [1단계: AI 서버 실행 (Raspberry Pi 5)](#1단계-ai-서버-실행-raspberry-pi-5)
5. [2단계: 펌웨어 업로드 (ESP32)](#2단계-펌웨어-업로드-esp32)
6. [3단계: 모바일 앱 실행 (Flutter)](#3단계-모바일-앱-실행-flutter)
7. [4단계: 웹 브릿지 실행 (선택)](#4단계-웹-브릿지-실행-선택)
8. [전체 동작 흐름](#전체-동작-흐름)
9. [AI 모델 학습 (재현)](#ai-모델-학습-재현)
10. [트러블슈팅](#트러블슈팅)

---

## 시스템 구성

```
┌─────────────┐    BLE     ┌──────────────┐   MQTT(WS)   ┌──────────────┐
│  ESP32 +    │ ────────── │  Flutter 앱  │ ──────────── │ Raspberry Pi │
│  ICS43434   │  Audio/Cmd │  (Android)   │   WAV 전송   │  AI Server   │
│  TFT LCD    │ ◄───────── │              │ ◄──────────── │  (Python)    │
└─────────────┘  AI 결과   └──────────────┘   AI 결과     └──────────────┘
```

| 계층 | 역할 | 통신 |
|------|------|------|
| **ESP32 펌웨어** | 청진음 수집, TFT 표시, BLE 전송 | BLE (Notify/Write) |
| **Flutter 앱** | BLE↔MQTT 브릿지, 대시보드 UI | BLE + MQTT over WebSocket |
| **AI 서버** | 노이즈 제거, 심음/호흡음 분류, EMR 관리 | MQTT (localhost:1883) |

---

## 제출 파일 목록

### 실행 코드

| 파일 | 설명 |
|------|------|
| `firmware_stethoscope.h` | ESP32 펌웨어 (Arduino C++, JSF++ 준수) |
| `stetho_app/` | Flutter 모바일 앱 (BLE 브릿지 + 대시보드) |
| `web_bridge_2026-03-26.html` | 웹 브릿지 (Web Bluetooth, 브라우저용 대안) |
| `server.py` | AI 추론 서버 (MQTT 수신 → 분류 → 결과 발행) |
| `predict_heart.py` | 심음 분류 추론 (ResNet+CBAM → XGBoost) |
| `predict_lung.py` | 호흡음 분류 추론 (MobileNetV2 + Focal Loss) |
| `predict_denoise.py` | 심음 노이즈 제거 추론 (TUNet-v2, TFLite) |

### 학습 코드

| 파일 | 설명 |
|------|------|
| `heart_preprocessing.py` | 심음 데이터 전처리 (4kHz, 5s 윈도우, 64-mel) |
| `heart_XGBClassifier_Full.py` | 심음 분류 모델 학습 (ResNet+CBAM + XGBoost) |
| `lung_data_balance.py` | 호흡음 스펙트로그램 생성 + 데이터 균형화 |
| `lung_focal224.py` | 호흡음 분류 모델 학습 (MobileNetV2, Focal Loss) |
| `denoise_train1.py` | 노이즈 제거 모델 학습 (Conv1D + BiLSTM → TFLite) |

---

## 환경 요구사항

### AI 서버 (Raspberry Pi 5)

- **OS:** Raspberry Pi OS (64-bit)
- **Python:** 3.11+
- **MQTT 브로커:** Mosquitto

```
pip install torch torchvision tflite-runtime librosa soundfile scipy \
            numpy pandas scikit-learn xgboost paho-mqtt matplotlib pillow
```

### ESP32 펌웨어

- **보드:** ESP32-DevKitC (또는 호환 보드)
- **센서:** ICS43434 I2S MEMS 마이크
- **디스플레이:** ST7789 TFT LCD (135x240)
- **IDE:** Arduino IDE 2.x 또는 PlatformIO
- **라이브러리:** `TFT_eSPI`, `ESP32 BLE Arduino`

### Flutter 앱

- **Flutter SDK:** 3.11.4+
- **대상:** Android 12+ (BLE 권한 필요)

### 웹 브릿지 (선택)

- **브라우저:** Chrome 89+ (Web Bluetooth API 지원)

---

## 1단계: AI 서버 실행 (Raspberry Pi 5)

### 1-1. Mosquitto MQTT 브로커 설치 및 실행

```bash
sudo apt install mosquitto mosquitto-clients
```

`/etc/mosquitto/conf.d/websocket.conf` 파일 생성:

```
listener 1883
listener 9001
protocol websockets
allow_anonymous true
```

```bash
sudo systemctl restart mosquitto
```

### 1-2. 디렉토리 구조 구성

```bash
mkdir -p ~/Desktop/audio_records
cd ~/Desktop/audio_records

# 서버 파일 배치
cp server.py predict_heart.py predict_lung.py predict_denoise.py ./

# 모델 파일 배치 (학습 완료된 가중치)
mkdir -p server_models/Heart_binary_classification
mkdir -p server_models/Lung_classification

# 심음 모델 복사
cp resnet_cbam_best.pth server_models/Heart_binary_classification/
cp heart_xgb_model.json server_models/Heart_binary_classification/

# 호흡음 모델 복사
cp stetho_mobilenetv2_224_focal.pth server_models/Lung_classification/

# 디노이즈 모델 복사
cp tunet_v2_best.tflite server_models/
```

> **참고:** `predict_*.py` 파일들은 `server_models/` 디렉토리에도 복사해야 합니다.
> `server.py`가 `server_models/` 경로를 `sys.path`에 추가하여 import합니다.

```bash
cp predict_heart.py predict_lung.py predict_denoise.py server_models/
```

### 1-3. 서버 실행

```bash
cd ~/Desktop/audio_records
python server.py
```

정상 실행 시 출력:

```
============================================================
 AI 모델 로드 중...
============================================================
[OK] 심음 분류 모델 로드 완료
[OK] 호흡음 분류 모델 로드 완료
[OK] 노이즈 제거 모델 로드 완료
============================================================
 모델 로드 완료: 3/3
============================================================

[SERVER] AI EMR 서버 가동 중!
[SERVER] 구독 완료: log / wav/# / req_patients / req_all_patients
```

---

## 2단계: 펌웨어 업로드 (ESP32)

### 2-1. 핀 연결

| 기능 | ESP32 핀 |
|------|----------|
| I2S WS (Word Select) | GPIO 25 |
| I2S SD (Serial Data) | GPIO 33 |
| I2S SCK (Clock) | GPIO 26 |
| 왼쪽 버튼 (모드 전환) | GPIO 35 |
| 오른쪽 버튼 (녹음) | GPIO 0 |
| TFT 백라이트 | GPIO 4 |

### 2-2. Arduino IDE 설정

1. **보드 매니저**에서 `esp32 by Espressif` 설치
2. 보드 선택: `ESP32 Dev Module`
3. 라이브러리 설치: `TFT_eSPI`, `ESP32 BLE Arduino`
4. `TFT_eSPI` User_Setup.h에서 ST7789 드라이버 활성화

### 2-3. 업로드

1. `firmware_stethoscope.h` 내용을 `.ino` 파일의 `#include`로 포함
2. 컴파일 및 업로드 (115200 baud)

### 2-4. 펌웨어 동작

- **전원 ON** → TFT에 초기 화면 표시, BLE 광고 시작
- **왼쪽 버튼**: 모드 전환 (심음 → 폐음 → 환자선택 → 심음 ...)
- **오른쪽 버튼**: 5초 녹음 시작
- 녹음 완료 → BLE로 `REC_DONE` 전송 후 AI 대기 애니메이션 (`. / .. / ...`)
- AI 결과 수신 → 전체화면 팝업 (Normal=초록 / Abnormal=빨강 / RECHECK=주황)
- 30초 타임아웃 시 자동 RECHECK 표시

---

## 3단계: 모바일 앱 실행 (Flutter)

### 3-1. 프로젝트 빌드

```bash
cd stetho_app/
flutter pub get
flutter run        # 디버그 빌드
# 또는
flutter build apk  # APK 생성
```

> **주의:** Release 빌드는 `ws://` (cleartext WebSocket)을 차단합니다.
> 현재 MQTT 브로커가 `ws://`를 사용하므로 **debug APK로 배포**해야 합니다.

### 3-2. 앱 사용법

1. 앱 실행 → **BLE 연결** 버튼으로 ESP32 스캔·연결
2. **MQTT 연결** 버튼으로 AI 서버 브로커 연결
3. ESP32에서 녹음 시작(오른쪽 버튼) → 앱이 BLE 오디오 수신
4. 녹음 완료(`REC_DONE`) → 앱이 WAV 변환 후 MQTT로 서버 전송
5. AI 결과 수신 → 대시보드에 표시 + BLE로 ESP32에 결과 전달

### 3-3. MQTT 브로커 주소 변경

앱의 `mqtt_service.dart`에서 브로커 IP를 환경에 맞게 수정:

```dart
// 라즈베리파이 IP 주소로 변경
final String brokerUrl = 'ws://라즈베리파이_IP:9001';
```

---

## 4단계: 웹 브릿지 실행 (선택)

Flutter 앱 대신 Chrome 브라우저에서 Web Bluetooth로 동작하는 대안입니다.

```bash
# 로컬 서버로 실행 (HTTPS 필요 시 mkcert 사용)
python -m http.server 8080
# 브라우저에서 http://localhost:8080/web_bridge_2026-03-26.html 접속
```

> **참고:** Web Bluetooth는 Chrome에서만 지원되며, HTTPS 또는 localhost에서만 동작합니다.

---

## 전체 동작 흐름

```
1. [ESP32] ICS43434로 청진음 캡처 (16kHz → 8kHz 데시메이션)
       ↓ BLE Notify (80 samples × int16)
2. [Flutter 앱] BLE 오디오 수신 → 5초간 버퍼링
       ↓ REC_DONE 수신
3. [Flutter 앱] WAV 인코딩 (8kHz, 16-bit, mono)
       ↓ MQTT: stethoscope/wav/{patient_id}/{HEART|LUNG}
4. [AI 서버] WAV 수신 → 저장
       ↓
5. [AI 서버] 심음인 경우: TUNet-v2 디노이즈 → 디노이즈 WAV 전송
       ↓
6. [AI 서버] ResNet+CBAM+XGBoost (심음) 또는 MobileNetV2 (호흡음) 분류
       ↓ MQTT: stethoscope/result (JSON)
7. [Flutter 앱] AI 결과 대시보드 표시
       ↓ BLE Write: "AI:HEART:Normal:87"
8. [ESP32] TFT LCD에 결과 팝업 표시
```

---

## AI 모델 학습 (재현)

### 심음 분류 (3-class: Normal / Abnormal / Unknown)

```bash
# 1. 전처리: WAV → 64-mel 스펙트로그램 (4kHz, 5s 윈도우, 2s 스트라이드)
python heart_preprocessing.py
# → x_data.npy, y_data.npy 생성

# 2. 모델 학습: ResNet+CBAM 특징 추출 → XGBoost 앙상블
python heart_XGBClassifier_Full.py
# → resnet_cbam_best.pth, heart_xgb_model.json 생성
```

### 호흡음 분류 (6-class: Complex / Crackle / Normal / Stridor / Unknown / Wheeze)

```bash
# 1. 스펙트로그램 생성 + ESC-50 Unknown 클래스 추가
python lung_data_balance.py
# → Spectrograms_Balanced/ 디렉토리에 224x224 PNG 생성

# 2. MobileNetV2 학습 (ImageNet pretrained, Focal Loss, 50 epochs)
python lung_focal224.py
# → stetho_mobilenetv2_224_focal.pth 생성
```

### 노이즈 제거 (심음 전용)

```bash
# Conv1D + Bidirectional LSTM (CRNN) 학습 → TFLite 변환
python denoise_train1.py
# → tunet_v2_best.tflite 생성
```

> **데이터셋 경로는 학습 스크립트 상단에서 수정 필요** (현재 `G:\stetho_ai\_misc\datasets\` 기준)

---

## BLE 프로토콜 요약

### UUID

| 용도 | UUID |
|------|------|
| Service | `19b10000-e8f2-537e-4f6c-d104768a1214` |
| Audio (Notify) | `19b10001-e8f2-537e-4f6c-d104768a1214` |
| Command (Write) | `19b10002-e8f2-537e-4f6c-d104768a1214` |

### 메시지 포맷

| 방향 | 메시지 | 설명 |
|------|--------|------|
| ESP32 → 앱 | `(raw binary)` | 80 samples × int16 오디오 |
| ESP32 → 앱 | `REC_DONE` | 녹음 완료 |
| ESP32 → 앱 | `REC_FAIL:DISCONNECTED` | 녹음 중 BLE 끊김 |
| ESP32 → 앱 | `PAT:P001\|김철수\|301호,...` | 환자 목록 |
| 앱 → ESP32 | `AI:HEART:Normal:87` | AI 결과 |
| 앱 → ESP32 | `AI:LUNG:RECHECK:0` | 재청진 지시 |

---

## MQTT 토픽 요약

| 방향 | 토픽 | 페이로드 |
|------|------|----------|
| 앱 → 서버 | `stethoscope/wav/{id}/{HEART\|LUNG}` | WAV 바이너리 |
| 앱 → 서버 | `stethoscope/req_patients` | (empty) |
| 앱 → 서버 | `stethoscope/req_all_patients` | (empty) |
| 서버 → 앱 | `stethoscope/result` | AI 결과 JSON |
| 서버 → 앱 | `stethoscope/denoised/{id}/HEART` | 디노이즈 WAV |
| 서버 → 앱 | `stethoscope/res_patients` | 환자 목록 JSON |
| 서버 → 앱 | `stethoscope/res_all_patients` | 전체 EMR JSON |

---

## 트러블슈팅

### 서버에서 모델 로드 실패

```
[WARN] 심음 분류 모델 로드 실패: ...
```

- `server_models/` 디렉토리에 모델 파일이 존재하는지 확인
- `predict_*.py` 파일이 `server_models/`에 복사되었는지 확인
- `tflite-runtime` 설치 여부 확인 (디노이즈 모델용)

### BLE 연결 안 됨

- ESP32가 BLE 광고 중인지 확인 (TFT에 BLE 아이콘 표시)
- Android 12+: 위치 권한 + 블루투스 스캔 권한 모두 필요
- 다른 기기와 이미 페어링된 경우 해제 후 재시도

### MQTT 연결 실패

- Mosquitto가 실행 중인지 확인: `sudo systemctl status mosquitto`
- WebSocket 리스너(9001번 포트)가 설정되었는지 확인
- 방화벽에서 1883, 9001 포트 허용 확인
- Release APK에서는 `ws://` 차단됨 → **debug APK 사용** 필요

### 녹음은 되는데 AI 결과가 안 옴

- 서버 콘솔에서 `[REC] 수신` 로그 확인
- WAV 크기가 44바이트 미만이면 무시됨 (헤더만 있는 빈 파일)
- AI 타임아웃 30초 후 ESP32에서 자동 RECHECK 표시

### Unknown 판정이 계속 나옴

- 청진기 흡착 부위를 확인 (피부 밀착 필요)
- 주변 소음이 심한 환경에서는 Unknown 판정 빈도 증가
- RMS 에너지가 임계값(0.005) 이하면 무음 판정
