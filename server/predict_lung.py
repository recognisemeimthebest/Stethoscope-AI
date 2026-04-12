"""
호흡음 분류 추론 스크립트 (ResNet18, npy 기반)
사용법: python predict_lung.py <wav_file>

클래스: Crackle / Normal / Unknown / Wheeze (알파벳 순)
"""

import sys
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ── RFN (Relaxed Frequency-wise Instance Normalization) ───────
class RFN(nn.Module):
    def __init__(self, num_features, eps=1e-5, p_init=0.5):
        super().__init__()
        self.eps = eps
        self.p = nn.Parameter(torch.full((1, 1, num_features, 1), p_init))

    def forward(self, x):
        mu = x.mean(dim=3, keepdim=True)
        std = x.std(dim=3, keepdim=True) + self.eps
        x_norm = (x - mu) / std
        p = torch.sigmoid(self.p)
        return p * x + (1 - p) * x_norm


class ResNet18_RFN(nn.Module):
    def __init__(self, num_classes, n_mels=128):
        super().__init__()
        self.rfn = RFN(num_features=n_mels)
        resnet = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(resnet.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.rfn(x)
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

# ── 경로 설정 ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Lung_classification", "stetho_resnet18_lung.pth")

DEVICE = torch.device("cpu")

# 전처리 파라미터 (data_balance.py 기준 — 8kHz PCEN)
SR          = 8000
FIXED_SEC   = 2
N_MELS      = 128
FMIN        = 50
FMAX        = 3500
HOP_LENGTH  = 80     # 10ms @ 8kHz
N_FFT       = 512    # 64ms @ 8kHz

# 클래스 (알파벳 순)
CLASSES = ["Crackle", "Normal", "Unknown", "Wheeze"]
UNKNOWN_IDX = CLASSES.index("Unknown")
UNKNOWN_THRESHOLD = 30.0


# ── 모델 로드 ──────────────────────────────────────────────
def load_model():
    model = ResNet18_RFN(num_classes=len(CLASSES), n_mels=N_MELS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


# ── WAV → 3채널 텐서 (mel + delta + delta2) ───────────────
def wav_to_tensor(wav_path):
    y, _ = librosa.load(wav_path, sr=SR)

    # 피크 정규화
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    y = librosa.util.fix_length(y, size=FIXED_SEC * SR)

    # mel-spectrogram
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX
    )
    mel = librosa.pcen(S * (2**31), sr=SR, hop_length=HOP_LENGTH).astype(np.float32)

    # 3채널: mel + delta + delta-delta
    delta = np.diff(mel, axis=1, prepend=mel[:, :1])
    delta2 = np.diff(delta, axis=1, prepend=delta[:, :1])
    spec = np.stack([mel, delta, delta2], axis=0)  # (3, 128, T)

    # 채널별 정규화
    for c in range(3):
        mean = spec[c].mean()
        std = spec[c].std() + 1e-6
        spec[c] = (spec[c] - mean) / std

    return torch.from_numpy(spec).unsqueeze(0)  # (1, 3, 128, T)


# ── 추론 ───────────────────────────────────────────────────
def predict(wav_path, model):
    tensor = wav_to_tensor(wav_path).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    # Unknown 우선 판정
    if probs[UNKNOWN_IDX] * 100 >= UNKNOWN_THRESHOLD:
        label = "Unknown"
    else:
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
        print("  호흡음으로 인식되지 않음 -- 다시 청진해주세요.")
    print()
    for cls, prob in sorted(result["probs"].items(), key=lambda x: -x[1]):
        bar = "#" * int(prob / 5)
        print(f"  {cls:<10} {prob:>6.2f}%  {bar}")
