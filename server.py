"""
================================================================
AI Stethoscope Server (Raspberry Pi 5)
================================================================
MQTT 수신 → TUNet-v2 노이즈 제거 → 심음/호흡음 분류 → 결과 전송

MQTT Topics:
  구독:
    stethoscope/wav/{patient_id}/{HEART|LUNG}  - WAV 수신
    stethoscope/req_patients                    - 환자 목록 요청
    stethoscope/log                             - 웹 로그

  발행:
    stethoscope/result          - AI 분류 결과 JSON
    stethoscope/denoised/{patient_id}/{type}  - 디노이즈 WAV
    stethoscope/res_patients    - 환자 목록 응답
================================================================
"""

import paho.mqtt.client as mqtt
import datetime
import os
import sys
import glob
import io
import json
import traceback
import numpy as np
import scipy.io.wavfile as wavfile
import pandas as pd

# ── 모델 경로 (server_models/ 디렉토리 기준) ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(SCRIPT_DIR, "server_models")
sys.path.insert(0, MODEL_DIR)

# ==========================================
# 설정
# ==========================================
BASE_DIR       = "/home/leejongwan/Desktop/audio_records"
EMR_FILE       = os.path.join(BASE_DIR, "patients_emr.csv")
ANALYSE_DIR    = os.path.join(BASE_DIR, "analyse")
ANALYSE_HEART  = os.path.join(ANALYSE_DIR, "HEART")
ANALYSE_LUNG   = os.path.join(ANALYSE_DIR, "LUNG")
DENOISED_DIR   = os.path.join(BASE_DIR, "denoised")
DENOISED_HEART = os.path.join(DENOISED_DIR, "HEART")

for folder in [ANALYSE_HEART, ANALYSE_LUNG, DENOISED_HEART]:
    os.makedirs(folder, exist_ok=True)


# ==========================================
# EMR 관리
# ==========================================
def init_emr():
    if not os.path.exists(EMR_FILE):
        df = pd.DataFrame([
            {"patient_id": "P001", "name": "김철수", "age": 65, "gender": "M",
             "ward": "301호", "status": "대기중", "audio_path": "", "ai_result": ""},
            {"patient_id": "P002", "name": "이영희", "age": 52, "gender": "F",
             "ward": "302호", "status": "대기중", "audio_path": "", "ai_result": ""},
            {"patient_id": "P003", "name": "박지민", "age": 28, "gender": "F",
             "ward": "응급실", "status": "대기중", "audio_path": "", "ai_result": ""},
        ])
        df.to_csv(EMR_FILE, index=False, encoding="utf-8-sig")
        print("[EMR] patients_emr.csv 생성 완료")


def get_waiting_patients():
    df = pd.read_csv(EMR_FILE, encoding="utf-8-sig")
    waiting = df[df["status"] == "대기중"]
    return waiting[["patient_id", "name", "ward"]].to_dict(orient="records")


def update_emr(patient_id, audio_path, ai_result):
    df = pd.read_csv(EMR_FILE, encoding="utf-8-sig")
    idx = df.index[df["patient_id"] == patient_id].tolist()
    if idx:
        df.at[idx[0], "status"] = "청진완료"
        df.at[idx[0], "audio_path"] = audio_path
        df.at[idx[0], "ai_result"] = ai_result
        df.to_csv(EMR_FILE, index=False, encoding="utf-8-sig")
        print(f"[EMR] {patient_id} 청진완료")


def get_patient_name(patient_id):
    df = pd.read_csv(EMR_FILE, encoding="utf-8-sig")
    p = df[df["patient_id"] == patient_id]
    return p["name"].values[0] if not p.empty else "알수없음"


def manage_storage(target_dir, max_files=100):
    try:
        files = sorted(glob.glob(os.path.join(target_dir, "*.wav")),
                       key=os.path.getmtime)
        while len(files) >= max_files:
            os.remove(files.pop(0))
    except Exception:
        pass


# ==========================================
# AI 모델 로드 (서버 시작 시 1회)
# ==========================================
print("=" * 60)
print(" AI 모델 로드 중...")
print("=" * 60)

# ── 심음 분류 모델 (ResNet+CBAM + XGBoost) ──
try:
    from predict_heart import load_models as load_heart_models
    from predict_heart import predict as predict_heart_fn
    from predict_heart import wav_to_mel_segments

    heart_resnet, heart_xgb = load_heart_models()
    print("[OK] 심음 분류 모델 로드 완료")
    HEART_MODEL_READY = True
except Exception as e:
    print(f"[WARN] 심음 분류 모델 로드 실패: {e}")
    HEART_MODEL_READY = False

# ── 호흡음 분류 모델 (MobileNetV2) ──
try:
    from predict_lung import load_model as load_lung_model
    from predict_lung import predict as predict_lung_fn

    lung_model = load_lung_model()
    print("[OK] 호흡음 분류 모델 로드 완료")
    LUNG_MODEL_READY = True
except Exception as e:
    print(f"[WARN] 호흡음 분류 모델 로드 실패: {e}")
    LUNG_MODEL_READY = False

# ── 노이즈 제거 모델 (TUNet-v2) ──
try:
    from predict_denoise import load_model as load_denoise_model
    from predict_denoise import denoise as denoise_fn

    denoise_model = load_denoise_model()
    print("[OK] 노이즈 제거 모델 로드 완료")
    DENOISE_MODEL_READY = True
except Exception as e:
    print(f"[WARN] 노이즈 제거 모델 로드 실패: {e}")
    DENOISE_MODEL_READY = False

print("=" * 60)
model_count = sum([HEART_MODEL_READY, LUNG_MODEL_READY, DENOISE_MODEL_READY])
print(f" 모델 로드 완료: {model_count}/3")
print("=" * 60)


# ==========================================
# AI 추론
# ==========================================
def run_heart_analysis(wav_path, client, patient_id):
    """심음: 디노이즈 → 분류"""
    result = {"scan_type": "HEART", "denoised": False}

    # 1) 디노이즈
    denoised_path = wav_path
    if DENOISE_MODEL_READY:
        try:
            import soundfile as sf
            denoised_audio, sr = denoise_fn(wav_path, denoise_model)
            denoised_path = wav_path.replace(".wav", "_denoised.wav")
            sf.write(denoised_path, denoised_audio, sr)
            result["denoised"] = True
            print(f"  [DENOISE] 완료: {os.path.basename(denoised_path)}")

            # 디노이즈 WAV를 MQTT로 전송
            with open(denoised_path, "rb") as f:
                denoised_bytes = f.read()
            topic = f"stethoscope/denoised/{patient_id}/HEART"
            client.publish(topic, denoised_bytes)
            print(f"  [MQTT] 디노이즈 WAV 전송 → {topic}")
        except Exception as e:
            print(f"  [DENOISE] 실패: {e}")

    # 2) 분류
    if HEART_MODEL_READY:
        try:
            classify_result = predict_heart_fn(denoised_path, heart_resnet, heart_xgb)
            result["label"]         = classify_result["label"]
            result["prob_normal"]   = classify_result["prob_normal"]
            result["prob_abnormal"] = classify_result["prob_abnormal"]
            result["segments"]      = classify_result["segments"]
            print(f"  [CLASSIFY] 심음: {result['label']} "
                  f"(정상 {result['prob_normal']}% / 비정상 {result['prob_abnormal']}%)")
        except Exception as e:
            print(f"  [CLASSIFY] 심음 분류 실패: {e}")
            result["label"] = "ERROR"
            result["prob_normal"] = 0
            result["prob_abnormal"] = 0
    else:
        result["label"] = "MODEL_NOT_LOADED"

    return result


def run_lung_analysis(wav_path, client, patient_id):
    """호흡음: 분류 (디노이즈 없음 — 심음 전용 모델)"""
    result = {"scan_type": "LUNG", "denoised": False}

    if LUNG_MODEL_READY:
        try:
            classify_result = predict_lung_fn(wav_path, lung_model)
            result["label"] = classify_result["label"]
            result["probs"] = classify_result["probs"]
            print(f"  [CLASSIFY] 호흡음: {result['label']}")
            for cls, prob in sorted(result["probs"].items(), key=lambda x: -x[1]):
                print(f"    {cls}: {prob:.1f}%")
        except Exception as e:
            print(f"  [CLASSIFY] 호흡음 분류 실패: {e}")
            result["label"] = "ERROR"
            result["probs"] = {}
    else:
        result["label"] = "MODEL_NOT_LOADED"

    return result


# ==========================================
# MQTT 콜백
# ==========================================
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        init_emr()
        print("\n[SERVER] AI EMR 서버 가동 중!")
        client.subscribe("stethoscope/log")
        client.subscribe("stethoscope/wav/#")
        client.subscribe("stethoscope/req_patients")
    else:
        print(f"[ERROR] 브로커 연결 실패 (rc={rc})")


def on_message(client, userdata, message):
    try:
        topic = message.topic

        # ── 환자 목록 요청 ──
        if topic == "stethoscope/req_patients":
            patients = get_waiting_patients()
            client.publish("stethoscope/res_patients", json.dumps(patients))
            print("[MQTT] 환자 목록 전송")

        # ── 웹 로그 ──
        elif topic == "stethoscope/log":
            print(f"[WEB] {message.payload.decode('utf-8').strip()}")

        # ── WAV 수신 → AI 분석 ──
        elif topic.startswith("stethoscope/wav/"):
            wav_data = message.payload
            parts = topic.split("/")
            patient_id = parts[-2]
            scan_type  = parts[-1].upper()

            if len(wav_data) < 44:
                return

            print(f"\n{'='*50}")
            print(f"[REC] 수신: {patient_id} / {scan_type} ({len(wav_data)} bytes)")

            # WAV 저장
            analyse_target = ANALYSE_HEART if scan_type == "HEART" else ANALYSE_LUNG
            manage_storage(analyse_target)

            now_str  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"record_{patient_id}_{scan_type}_{now_str}.wav"
            filepath = os.path.join(analyse_target, filename)

            rate, data = wavfile.read(io.BytesIO(wav_data))
            if len(data.shape) == 2:
                data = data[:, 0]
            data = data - np.mean(data)
            max_val = np.max(np.abs(data))
            if max_val > 0:
                data = (data / max_val) * 32700
            data_mono = data.astype(np.int16)
            wavfile.write(filepath, rate, data_mono)
            print(f"[SAVE] {filename}")

            # AI 분석
            print("[AI] 분석 시작...")
            if scan_type == "HEART":
                ai_result = run_heart_analysis(filepath, client, patient_id)
            else:
                ai_result = run_lung_analysis(filepath, client, patient_id)

            # 결과 JSON 구성
            p_name = get_patient_name(patient_id)
            result_data = {
                "patient_id":   patient_id,
                "patient_name": p_name,
                "scan_type":    scan_type,
                "label":        ai_result.get("label", "UNKNOWN"),
                "denoised":     ai_result.get("denoised", False),
                "timestamp":    now_str,
            }

            if scan_type == "HEART":
                result_data["prob_normal"]   = ai_result.get("prob_normal", 0)
                result_data["prob_abnormal"] = ai_result.get("prob_abnormal", 0)
                result_data["segments"]      = ai_result.get("segments", 0)
                ai_str = f"{ai_result.get('label','?')} (정상 {ai_result.get('prob_normal',0):.1f}%)"
            else:
                result_data["probs"] = ai_result.get("probs", {})
                top_prob = max(ai_result.get("probs", {}).values(), default=0)
                ai_str = f"{ai_result.get('label','?')} ({top_prob:.1f}%)"

            # EMR 업데이트
            update_emr(patient_id, filepath, ai_str)

            # 결과 MQTT 발행
            client.publish("stethoscope/result", json.dumps(result_data))
            print(f"[MQTT] AI 결과 전송: {p_name} → {ai_str}")
            print(f"{'='*50}\n")

    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()


# ==========================================
# 메인
# ==========================================
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect("localhost", 1883)
    client.loop_forever()
except Exception as e:
    print(f"[FATAL] 서버 구동 실패: {e}")
