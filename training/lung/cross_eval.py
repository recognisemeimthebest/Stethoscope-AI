"""
크로스 데이터셋 검증
- 학습된 3클래스 모델로 SPRSound / ICBHI 각각 별도 평가
- 데이터셋 간 일반화 성능 확인
"""

import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix

# ── 모델 정의 (focal224.py와 동일) ────────────────────────────
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
        self.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(resnet.fc.in_features, num_classes))
    def forward(self, x):
        x = self.rfn(x)
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

# ── 전처리 파라미터 ──────────────────────────────────────────
SR = 8000
WINDOW_SEC = 2
WINDOW_SAMPLES = SR * WINDOW_SEC
STRIDE_SAMPLES = SR * 1
N_MELS = 128
FMIN = 50
FMAX = 3500
HOP_LENGTH = 80
N_FFT = 512
TRIM_TOP_DB = 25
MIN_AUDIO_SEC = 0.5

CLASSES = ['Abnormal', 'Normal', 'Unknown']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_wav(wav_path, model):
    """WAV → 2초 슬라이딩 윈도우 → 다수결 판정"""
    y, _ = librosa.load(wav_path, sr=SR)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)

    if len(y) < int(SR * MIN_AUDIO_SEC):
        return None

    if len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))

    preds = []
    for start in range(0, len(y) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
        seg = y[start:start + WINDOW_SAMPLES]
        S = librosa.feature.melspectrogram(y=seg, sr=SR, n_mels=N_MELS, n_fft=N_FFT,
                                           hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX)
        mel = librosa.pcen(S * (2**31), sr=SR, hop_length=HOP_LENGTH, bias=10).astype(np.float32)

        delta = np.diff(mel, axis=1, prepend=mel[:, :1])
        delta2 = np.diff(delta, axis=1, prepend=delta[:, :1])
        spec = np.stack([mel, delta, delta2], axis=0)
        for c in range(3):
            m = spec[c].mean()
            s = spec[c].std() + 1e-6
            spec[c] = (spec[c] - m) / s

        tensor = torch.from_numpy(spec).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(tensor)
            pred = torch.argmax(logits, dim=1).item()
        preds.append(pred)

    if not preds:
        return None

    # 다수결
    from collections import Counter
    counts = Counter(preds)
    return counts.most_common(1)[0][0]


def eval_dataset(name, file_dict, model):
    """데이터셋 평가"""
    print(f"\n{'='*60}")
    print(f"{name} 평가")
    print(f"{'='*60}")

    all_preds, all_labels = [], []

    for cls_name, wav_list in file_dict.items():
        label = CLASSES.index(cls_name)
        correct = 0
        for wav_path in wav_list:
            pred = predict_wav(wav_path, model)
            if pred is None:
                continue
            all_preds.append(pred)
            all_labels.append(label)
            if pred == label:
                correct += 1
        total = len(wav_list)
        print(f"  {cls_name}: {correct}/{total} = {100*correct/total:.1f}%")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = 100 * (all_preds == all_labels).sum() / len(all_labels)
    print(f"\n  Accuracy: {acc:.2f}%")
    print(classification_report(all_labels, all_preds,
                                target_names=['Abnormal', 'Normal'], digits=3))

    cm = confusion_matrix(all_labels, all_preds)
    print(f"  {'':>10}{'Abnorm':>8}{'Normal':>8}")
    for i, cls in enumerate(['Abnormal', 'Normal']):
        print(f"  {cls:>10}{''.join(f'{v:>8}' for v in cm[i])}")


# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # 모델 로드
    model = ResNet18_RFN(num_classes=len(CLASSES), n_mels=N_MELS)
    model.load_state_dict(torch.load(
        r"G:\stetho_ai\Lung_classification\stetho_resnet18_lung.pth",
        map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    print(f"Device: {DEVICE}")

    # ── SPRSound 테스트셋 ─────────────────────────────────────
    spr_base = r"G:\stetho_ai\_misc\lung\classification\LUNG_SOUND\Dataset_Split\test"
    spr_map = {
        'Normal': 'Normal', 'Fine Crackle': 'Abnormal', 'Coarse Crackle': 'Abnormal',
        'Wheeze': 'Abnormal', 'Rhonchi': 'Abnormal', 'Wheeze+Crackle': 'Abnormal', 'Stridor': 'Abnormal'
    }

    spr_files = {'Abnormal': [], 'Normal': []}
    for old_cls in os.listdir(spr_base):
        if old_cls not in spr_map:
            continue
        new_cls = spr_map[old_cls]
        cls_dir = os.path.join(spr_base, old_cls)
        for f in os.listdir(cls_dir):
            if f.endswith('.wav'):
                spr_files[new_cls].append(os.path.join(cls_dir, f))

    print(f"\nSPRSound test: Abnormal={len(spr_files['Abnormal'])}, Normal={len(spr_files['Normal'])}")

    # ── ICBHI 전체 (학습에 사용됨 — 참고용) ──────────────────
    icbhi_dir = r"G:\stetho_ai\_misc\lung\datasets\ICBHI_processed"
    icbhi_files = {'Abnormal': [], 'Normal': []}

    for sub_cls in ['Crackle', 'Wheeze']:
        d = os.path.join(icbhi_dir, sub_cls)
        if os.path.exists(d):
            icbhi_files['Abnormal'].extend([os.path.join(d, f) for f in os.listdir(d) if f.endswith('.wav')])

    d = os.path.join(icbhi_dir, 'Normal')
    if os.path.exists(d):
        icbhi_files['Normal'].extend([os.path.join(d, f) for f in os.listdir(d) if f.endswith('.wav')])

    print(f"ICBHI: Abnormal={len(icbhi_files['Abnormal'])}, Normal={len(icbhi_files['Normal'])}")

    # ── 평가 ──────────────────────────────────────────────────
    eval_dataset("SPRSound (테스트셋)", spr_files, model)
    eval_dataset("ICBHI (전체)", icbhi_files, model)
