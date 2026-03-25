# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered electronic stethoscope system built on ESP32 with ICS43434 I2S microphone. It captures heart/lung sounds, transmits audio over BLE to a mobile web bridge, and runs ML inference for abnormal sound detection.

## Architecture

Three-layer system:

**1. Hardware Firmware** (`stethoscope-software/`)
- Arduino C++ targeting ESP32
- Latest build: `stethoscope-software/2026-03-25-1442.h`
- ICS43434 I2S microphone at 16kHz/16-bit; TFT LCD for patient list UI
- BLE GATT with two characteristics: audio stream + command/patient list
- Key pins: I2S_WS=25, I2S_SD=33, I2S_SCK=26, BTN_MODE=35, BTN_RECORD=0

**2. Mobile Web Bridge** (`Stetho_Web/`)
- Single HTML file (`2026-03-25-1442.html`) using vanilla JS, Web Bluetooth API, MQTT.js
- Connects to hospital MQTT server (`ws://114.206.167.99:9001`), retrieves patient list, pushes it to the device over BLE as `PAT:P001|Kim|301,...`
- Converts received BLE audio bytes to WAV (8kHz, 16-bit) and publishes to MQTT topic `stethoscope/wav/{patient_id}/{record_type}`

**3. ML Models** (Python, PyTorch/TensorFlow)
- **Heart classification** (`Heart_binary_classification/`): ResNet+CBAM (`resnet_cbam_best.pth`) + XGBoost ensemble (`heart_xgb_model.json`). Input: 64-mel spectrograms, 5s windows, 2s stride.
- **Lung classification** (`Lung_classification/`): MobileNetV2 with focal loss (`stetho_mobilenetv2_224_focal.pth`). Input: 224×224 spectrograms.
- **Denoising** (`LSTM_first/`): Conv1D + Bidirectional LSTM (CRNN). Trained on 10,000 synthetic samples.

## BLE UUIDs

```
Service:  19b10000-e8f2-537e-4f6c-d104768a1214
Audio:    19b10001-e8f2-537e-4f6c-d104768a1214
Command:  19b10002-e8f2-537e-4f6c-d104768a1214
```

## Running ML Training

Data paths are **hardcoded** in training scripts pointing to `G:\stetho_ai\...`. Update these before running locally.

```bash
# Heart sound classification
cd Heart_binary_classification/
python preprocessing.py           # Generate mel-spectrograms from .wav
python XGBClassifier_Full.py      # Train XGBoost ensemble

# Lung sound classification
cd Lung_classification/
python train_MobileNetV2.py       # 50 epochs, batch 32

# Denoising
cd LSTM_first/
python train1.py                  # CRNN training
```

## Python Dependencies

```
torch torchvision tensorflow librosa soundfile scipy numpy pandas scikit-learn xgboost tqdm
```

## Hardware Build

Compiled via Arduino IDE or PlatformIO targeting ESP32. Required libraries: `TFT_eSPI` (display), ESP32 BLE Arduino (BLE), I2S driver.

## Key Notes

- The firmware files are `.h` header files (not `.ino`), named by timestamp — the newest timestamp is the authoritative source.
- MQTT server endpoint and patient data format are tightly coupled between the firmware, web bridge, and backend — changes must be coordinated across all three layers.
- Sample audio for testing: `sample_sound/raw_heart_sound.wav`.

# Git Commit Rules

- 파일 변경/추가 시 자동으로 커밋하지 말 것
- 변경사항을 분석해서 커밋 메시지를 먼저 보여줄 것
- 커밋 메시지 형식:
  - Summary: 한 줄 요약 (영문, 50자 이내)
  - Description: 변경 내용 bullet point로 정리
- 내가 "커밋해" 또는 "푸시해"라고 할 때만 실행할 것
- conventional commit 형식 사용 (feat:, fix:, docs: 등)