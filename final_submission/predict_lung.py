"""
호흡음 분류 추론 스크립트 (MobileNetV2, 224x224 focal)
사용법: python predict_lung.py <wav_file>

클래스: Complex / Crackle / Normal / Stridor / Wheeze  (알파벳 순)
"""

import sys
import os
import io
import numpy as np
import librosa
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 경로 설정 ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Lung_classification", "stetho_mobilenetv2_224_focal.pth")

DEVICE = torch.device("cpu")

# 전처리 파라미터 (data_balance.py 기준)
SR          = 16000
FIXED_SEC   = 2            # 2초로 고정
N_MELS      = 128
FMIN        = 50
FMAX        = 2000

# ImageFolder 알파벳 정렬 순서
CLASSES = ["Complex", "Crackle", "Normal", "Stridor", "Unknown", "Wheeze"]
UNKNOWN_IDX = CLASSES.index("Unknown")

# Unknown 판정 임계값
UNKNOWN_THRESHOLD = 30.0

# 이미지 정규화 (ImageNet 기준)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── 모델 로드 ──────────────────────────────────────────────
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.last_channel, len(CLASSES)),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


# ── WAV → mel spectrogram 이미지 (PIL) ────────────────────
def wav_to_pil(wav_path):
    y, _ = librosa.load(wav_path, sr=SR)
    y = librosa.util.fix_length(y, size=FIXED_SEC * SR)

    S    = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(3, 3))
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    plt.imshow(S_dB, aspect="auto", origin="lower", cmap="magma")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ── 추론 ───────────────────────────────────────────────────
def predict(wav_path, model):
    pil_img = wav_to_pil(wav_path)
    tensor  = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)   # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(tensor)                              # (1, 5)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    # Unknown 우선 판정
    if probs[UNKNOWN_IDX] * 100 >= UNKNOWN_THRESHOLD:
        label = "Unknown"
    else:
        # Unknown 제외하고 나머지 클래스만 비교
        probs_no_unk = probs.copy()
        probs_no_unk[UNKNOWN_IDX] = 0.0
        top_idx = int(np.argmax(probs_no_unk))
        label = CLASSES[top_idx]
    result = {
        "label": label,
        "probs": {cls: round(float(p) * 100, 2) for cls, p in zip(CLASSES, probs)},
    }
    return result


# ── 메인 ───────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python predict_lung.py <wav_file>")
        sys.exit(1)

    wav_path = sys.argv[1]
    if not os.path.exists(wav_path):
        print(f"파일 없음: {wav_path}")
        sys.exit(1)

    print("모델 로드 중...")
    model = load_model()

    print(f"추론 중: {wav_path}")
    result = predict(wav_path, model)

    print("\n===== 호흡음 분류 결과 =====")
    print(f"  판정: {result['label']}")
    if result['label'] == "Unknown":
        print("  ⚠️ 호흡음으로 인식되지 않음 — 다시 청진해주세요.")
    print()
    for cls, prob in sorted(result["probs"].items(), key=lambda x: -x[1]):
        bar = "#" * int(prob / 5)
        print(f"  {cls:<10} {prob:>6.2f}%  {bar}")
