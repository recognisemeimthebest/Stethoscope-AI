# Web — 모바일 웹 블루투스 브릿지

Chrome 브라우저에서 Web Bluetooth API로 ESP32에 연결하고,  
MQTT를 통해 AI 서버와 통신하는 중계 역할을 합니다.

---

## 파일 목록

| 파일 | 설명 |
|------|------|
| **`2026-03-26.html`** | **최종 버전** — BLE↔MQTT 브릿지 + 녹음/결과 UI |
| `2026-03-25-1442.html` | 이전 버전 (참고용) |
| `presentation.html` | 발표용 미러링 디스플레이 (LCD 화면 재현) |

---

## 동작 흐름

```
Chrome 브라우저
    │
    ├── Web Bluetooth ──→ ESP32 BLE 연결
    │       ↓ Audio Notify (80 samples x int16)
    │       ↓ REC_DONE / REC_FAIL 수신
    │
    ├── MQTT.js ──→ ws://서버IP:9001 연결
    │       ↑ WAV 업로드: stethoscope/wav/{patient_id}/{HEART|LUNG}
    │       ↓ AI 결과 수신: stethoscope/result (JSON)
    │
    └── BLE Write ──→ ESP32에 "AI:HEART:Normal:87" 전송
```

---

## 주요 함수

| 함수 | 역할 |
|------|------|
| `connectBLE()` | ESP32 BLE 스캔 ��� 연결 |
| `connectMQTT()` | MQTT 브로커 WebSocket 연결 |
| `startRec()` | 5초 녹음 시작 |
| `stopRec()` | 녹음 완료 → WAV 컴파일 |
| `publishResult()` | WAV를 MQTT로 서버 전송 |
| `handleAiResult()` | AI 결과 UI 표시 + BLE 전달 |
| `render()` | Canvas 파형 애니메이션 |

---

## 실행

```bash
# 로컬 서버 (HTTPS 또는 localhost 필요)
python -m http.server 8080
# Chrome에서 http://localhost:8080/2026-03-26.html
```

> Web Bluetooth는 Chrome 89+ 에서만 지원됩니다.
