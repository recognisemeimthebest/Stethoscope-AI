"""
5초 오디오 평가 파이프라인
- 5초 WAV → 2초 사이클로 분할 → 사이클별 분류 → 비율 집계 → 최종 판정
- SPRSound(Dataset_Split) test set으로 검증
"""

import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

# ── 모델 정의 (frame_train.py와 동일) ────────────────────────
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

class FrameCNN(nn.Module):
    def __init__(self, num_classes, n_mels=64):
        super().__init__()
        self.rfn = RFN(num_features=n_mels)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(128, num_classes))
    def forward(self, x):
        x = self.rfn(x)
        x = self.conv(x)
        x = x.flatten(1)
        return self.classifier(x)

# ── 전처리 파라미터 ──────────────────────────────────────────
SR = 8000
CYCLE_SEC = 2.0
CYCLE_SAMPLES = int(SR * CYCLE_SEC)
MIN_ENERGY = 0.001
N_MELS = 64
FMIN = 50
FMAX = 3500
HOP_LENGTH = 40
N_FFT = 256

CLASSES = ['Crackle', 'Normal', 'Unknown', 'Wheeze']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SPRSound 클래스 매핑
class_map = {
    'Normal': 'Normal',
    'Fine Crackle': 'Crackle',
    'Coarse Crackle': 'Crackle',
    'Wheeze': 'Wheeze',
    'Rhonchi': 'Wheeze',
    'Wheeze+Crackle': 'Crackle',
    'Stridor': 'Wheeze'
}


def make_mel(y_seg):
    S = librosa.feature.melspectrogram(
        y=y_seg, sr=SR, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX
    )
    return librosa.pcen(S * (2**31), sr=SR, hop_length=HOP_LENGTH, bias=10).astype(np.float32)


def mel_to_tensor(mel):
    """mel → 3ch tensor (mel + delta + delta2)"""
    delta = np.diff(mel, axis=1, prepend=mel[:, :1])
    delta2 = np.diff(delta, axis=1, prepend=delta[:, :1])
    spec = np.stack([mel, delta, delta2], axis=0)
    for c in range(3):
        m = spec[c].mean()
        s = spec[c].std() + 1e-6
        spec[c] = (spec[c] - m) / s
    return torch.from_numpy(spec).unsqueeze(0)


def predict_5sec(wav_path, model):
    """5초 WAV → 2초 사이클별 분류 → 비율 집계 → 최종 판정"""
    y, _ = librosa.load(wav_path, sr=SR)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    # 2초씩 자르기 (stride=1초로 오버랩)
    stride = int(SR * 1.0)
    cycle_preds = []

    for start in range(0, len(y) - CYCLE_SAMPLES + 1, stride):
        seg = y[start:start + CYCLE_SAMPLES]
        if np.sqrt(np.mean(seg ** 2)) < MIN_ENERGY:
            continue

        mel = make_mel(seg)
        tensor = mel_to_tensor(mel).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_cls = CLASSES[pred_idx]

        # Unknown은 집계에서 제외 (호흡음이 아닌 구간)
        if pred_cls != 'Unknown':
            cycle_preds.append(pred_cls)

    if not cycle_preds:
        return 'Unknown', {}

    # 비율 집계
    counts = Counter(cycle_preds)
    total = len(cycle_preds)
    ratios = {cls: counts.get(cls, 0) / total * 100 for cls in ['Crackle', 'Normal', 'Wheeze']}

    # 최종 판정: 가장 높은 비율
    final = max(ratios, key=ratios.get)
    return final, ratios


# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # 모델 로드
    model_path = r"G:\stetho_ai\Lung_classification\frame_cnn_lung.pth"
    model = FrameCNN(num_classes=len(CLASSES), n_mels=N_MELS)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    print(f"모델 로드: {model_path}")
    print(f"Device: {DEVICE}")

    # SPRSound test set
    src_base = r"G:\stetho_ai\_misc\lung\classification\LUNG_SOUND\Dataset_Split\test"
    print(f"\nSPRSound test 경로: {src_base}")

    all_preds = []
    all_labels = []
    cls_idx = {cls: i for i, cls in enumerate(['Crackle', 'Normal', 'Wheeze'])}

    for old_cls in os.listdir(src_base):
        if old_cls not in class_map:
            continue
        new_cls = class_map[old_cls]
        if new_cls not in cls_idx:
            continue

        cls_dir = os.path.join(src_base, old_cls)
        wavs = [f for f in os.listdir(cls_dir) if f.endswith('.wav')]

        correct = 0
        for wav_name in wavs:
            wav_path = os.path.join(cls_dir, wav_name)
            pred, ratios = predict_5sec(wav_path, model)
            all_preds.append(pred)
            all_labels.append(new_cls)
            if pred == new_cls:
                correct += 1

        acc = 100 * correct / len(wavs) if wavs else 0
        print(f"  {new_cls} ({old_cls}): {correct}/{len(wavs)} = {acc:.1f}%")

    # 전체 결과
    TARGET_CLASSES = ['Crackle', 'Normal', 'Wheeze']
    print(f"\n{'='*60}")
    print("5초 파이프라인 평가 (SPRSound test)")
    print(f"{'='*60}")

    total_correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    total = len(all_labels)
    print(f"\nAccuracy: {100*total_correct/total:.2f}% ({total_correct}/{total})")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, labels=TARGET_CLASSES, digits=3))

    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds, labels=TARGET_CLASSES)
    header = "".join(f"{c[:6]:>8}" for c in TARGET_CLASSES)
    print(f"{'':>10}{header}")
    for i, cls in enumerate(TARGET_CLASSES):
        print(f"{cls:>10}{''.join(f'{v:>8}' for v in cm[i])}")
