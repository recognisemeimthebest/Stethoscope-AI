"""
6개 실험 종합 평가 — 테스트셋 + 크로스데이터셋(SPRSound/ICBHI)
"""

import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# ── 모델 정의 ────────────────────────────────────────────────
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

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        avg = x.mean(dim=(2, 3))
        mx = x.amax(dim=(2, 3))
        att = torch.sigmoid(self.fc(avg) + self.fc(mx))
        return x * att.view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x))

class ResNet18_RFN(nn.Module):
    def __init__(self, num_classes, n_mels=128):
        super().__init__()
        self.rfn = RFN(num_features=n_mels)
        resnet = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, num_classes))
    def forward(self, x):
        x = self.rfn(x)
        x = self.features(x)
        return self.classifier(x.flatten(1))

class ResNet18_CBAM_RFN(nn.Module):
    def __init__(self, num_classes, n_mels=128):
        super().__init__()
        self.rfn = RFN(num_features=n_mels)
        resnet = models.resnet18(weights=None)
        layers = list(resnet.children())
        self.stem = nn.Sequential(*layers[:4])
        self.layer1 = layers[4]
        self.cbam1 = CBAM(64)
        self.layer2 = layers[5]
        self.cbam2 = CBAM(128)
        self.layer3 = layers[6]
        self.cbam3 = CBAM(256)
        self.layer4 = layers[7]
        self.cbam4 = CBAM(512)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, num_classes))
    def forward(self, x):
        x = self.rfn(x)
        x = self.stem(x)
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        x = self.avgpool(x)
        return self.classifier(x.flatten(1))

class MobileNetV2_RFN(nn.Module):
    def __init__(self, num_classes, n_mels=128):
        super().__init__()
        self.rfn = RFN(num_features=n_mels)
        mob = models.mobilenet_v2(weights=None)
        self.features = mob.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, num_classes))
    def forward(self, x):
        x = self.rfn(x)
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x.flatten(1))


# ── 전처리 ───────────────────────────────────────────────────
SR = 8000
WINDOW_SAMPLES = SR * 2
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
BASE_DIR = r"G:\stetho_ai\Lung_classification"


def load_model(arch, path, num_classes=3):
    if arch == 'resnet':
        model = ResNet18_RFN(num_classes)
    elif arch == 'cbam':
        model = ResNet18_CBAM_RFN(num_classes)
    elif arch == 'mobilenet':
        model = MobileNetV2_RFN(num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


def wav_to_segments(wav_path):
    """WAV → mel spectrogram 세그먼트 리스트"""
    y, _ = librosa.load(wav_path, sr=SR)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)

    if len(y) < int(SR * MIN_AUDIO_SEC):
        return []

    if len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))

    tensors = []
    for start in range(0, len(y) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
        seg = y[start:start + WINDOW_SAMPLES]
        S = librosa.feature.melspectrogram(y=seg, sr=SR, n_mels=N_MELS, n_fft=N_FFT,
                                           hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX)
        mel = librosa.pcen(S * (2**31), sr=SR, hop_length=HOP_LENGTH, bias=10).astype(np.float32)
        delta = np.diff(mel, axis=1, prepend=mel[:, :1])
        delta2 = np.diff(delta, axis=1, prepend=delta[:, :1])
        spec = np.stack([mel, delta, delta2], axis=0)
        for c in range(3):
            m, s = spec[c].mean(), spec[c].std() + 1e-6
            spec[c] = (spec[c] - m) / s
        tensors.append(torch.from_numpy(spec).unsqueeze(0))

    return tensors


def predict_single(model, tensors):
    """단일 모델 다수결 예측"""
    if not tensors:
        return None
    preds = []
    with torch.no_grad():
        for t in tensors:
            logits = model(t.to(DEVICE))
            preds.append(torch.argmax(logits, 1).item())
    return Counter(preds).most_common(1)[0][0]


def predict_ensemble(models_list, tensors):
    """앙상블 (세그먼트별 소프트 보팅 → 다수결)"""
    if not tensors:
        return None
    preds = []
    with torch.no_grad():
        for t in tensors:
            t = t.to(DEVICE)
            avg_logits = sum(m(t) for m in models_list) / len(models_list)
            preds.append(torch.argmax(avg_logits, 1).item())
    return Counter(preds).most_common(1)[0][0]


def eval_on_files(file_dict, predictor_fn, label_filter=None):
    """파일 딕셔너리로 평가. predictor_fn(tensors) → pred_idx"""
    all_preds, all_labels = [], []
    per_class = {}

    for cls_name, wav_list in file_dict.items():
        if label_filter and cls_name not in label_filter:
            continue
        label = CLASSES.index(cls_name)
        correct = 0
        total = 0
        for wav_path in wav_list:
            tensors = wav_to_segments(wav_path)
            pred = predictor_fn(tensors)
            if pred is None:
                continue
            all_preds.append(pred)
            all_labels.append(label)
            total += 1
            if pred == label:
                correct += 1
        per_class[cls_name] = (correct, total)

    return np.array(all_preds), np.array(all_labels), per_class


# ── 캐시 테스트셋 평가 ───────────────────────────────────────
class CacheDataset(Dataset):
    def __init__(self, data_npy, labels_npy):
        self.mels = np.load(data_npy, mmap_mode='r')
        self.labels = np.load(labels_npy)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        mel = self.mels[idx]
        delta = np.diff(mel, axis=1, prepend=mel[:, :1])
        delta2 = np.diff(delta, axis=1, prepend=delta[:, :1])
        spec = np.stack([mel, delta, delta2], axis=0)
        for c in range(3):
            m, s = spec[c].mean(), spec[c].std() + 1e-6
            spec[c] = (spec[c] - m) / s
        return torch.from_numpy(spec), int(self.labels[idx])


def eval_cache_test(model, cache_dir):
    """캐시 테스트셋으로 단일 모델 평가"""
    ds = CacheDataset(os.path.join(cache_dir, 'test_data.npy'),
                      os.path.join(cache_dir, 'test_labels.npy'))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    preds, labels = [], []
    with torch.no_grad():
        for specs, lbl in loader:
            out = model(specs.to(DEVICE))
            preds.extend(torch.argmax(out, 1).cpu().numpy())
            labels.extend(lbl.numpy())
    return np.array(preds), np.array(labels)


def eval_cache_ensemble(models_list, cache_dir):
    """캐시 테스트셋으로 앙상블 평가"""
    ds = CacheDataset(os.path.join(cache_dir, 'test_data.npy'),
                      os.path.join(cache_dir, 'test_labels.npy'))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    preds, labels = [], []
    with torch.no_grad():
        for specs, lbl in loader:
            specs = specs.to(DEVICE)
            avg = sum(m(specs) for m in models_list) / len(models_list)
            preds.extend(torch.argmax(avg, 1).cpu().numpy())
            labels.extend(lbl.numpy())
    return np.array(preds), np.array(labels)


# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f"Device: {DEVICE}")

    # ── 데이터셋 파일 목록 ────────────────────────────────────
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

    icbhi_dir = r"G:\stetho_ai\_misc\lung\datasets\ICBHI_processed"
    icbhi_files = {'Abnormal': [], 'Normal': []}
    for sub in ['Crackle', 'Wheeze']:
        d = os.path.join(icbhi_dir, sub)
        if os.path.exists(d):
            icbhi_files['Abnormal'].extend([os.path.join(d, f) for f in os.listdir(d) if f.endswith('.wav')])
    d = os.path.join(icbhi_dir, 'Normal')
    if os.path.exists(d):
        icbhi_files['Normal'].extend([os.path.join(d, f) for f in os.listdir(d) if f.endswith('.wav')])

    print(f"SPRSound test: Abn={len(spr_files['Abnormal'])}, Nor={len(spr_files['Normal'])}")
    print(f"ICBHI: Abn={len(icbhi_files['Abnormal'])}, Nor={len(icbhi_files['Normal'])}")

    # ── 실험 정의 ─────────────────────────────────────────────
    # SPR+ICBHI 캐시 (HF 없음) — 원본 학습에 사용된 모델들
    # HF 캐시 (HF 포함) — HF 실험에 사용된 모델들
    # 캐시 테스트셋은 해당 실험의 캐시를 사용해야 하지만,
    # 크로스데이터셋은 원본 WAV로 직접 평가하므로 캐시 무관

    experiments = [
        {
            'name': 'Baseline: ResNet18+RFN',
            'type': 'single',
            'arch': 'resnet',
            'path': os.path.join(BASE_DIR, 'stetho_resnet18_lung.pth'),
        },
        {
            'name': '② HF Lung: ResNet18+RFN',
            'type': 'single',
            'arch': 'resnet',
            'path': os.path.join(BASE_DIR, 'hf_resnet_lung.pth'),
        },
        {
            'name': '③ CBAM: ResNet18+CBAM+RFN',
            'type': 'single',
            'arch': 'cbam',
            'path': os.path.join(BASE_DIR, 'cbam_lung.pth'),
        },
        {
            'name': '④ Ensemble: ResNet18 + MobileNetV2',
            'type': 'ensemble',
            'models': [
                ('resnet', os.path.join(BASE_DIR, 'stetho_resnet18_lung.pth')),
                ('mobilenet', os.path.join(BASE_DIR, 'mobilenet_lung.pth')),
            ],
        },
        {
            'name': '②+③ HF+CBAM',
            'type': 'single',
            'arch': 'cbam',
            'path': os.path.join(BASE_DIR, 'hf_cbam_lung.pth'),
        },
        {
            'name': '③+④ CBAM+Ensemble: CBAM + MobileNetV2',
            'type': 'ensemble',
            'models': [
                ('cbam', os.path.join(BASE_DIR, 'cbam_lung.pth')),
                ('mobilenet', os.path.join(BASE_DIR, 'mobilenet_lung.pth')),
            ],
        },
    ]

    results = []

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"실험: {exp['name']}")
        print(f"{'='*60}")

        # 모델 로드
        if exp['type'] == 'single':
            if not os.path.exists(exp['path']):
                print(f"  ⚠ 모델 파일 없음: {exp['path']}")
                continue
            model = load_model(exp['arch'], exp['path'])
            predictor = lambda t, m=model: predict_single(m, t)
        else:
            loaded = []
            skip = False
            for arch, path in exp['models']:
                if not os.path.exists(path):
                    print(f"  ⚠ 모델 파일 없음: {path}")
                    skip = True
                    break
                loaded.append(load_model(arch, path))
            if skip:
                continue
            predictor = lambda t, ms=loaded: predict_ensemble(ms, t)

        # 크로스데이터셋 평가
        for ds_name, ds_files in [("SPRSound", spr_files), ("ICBHI", icbhi_files)]:
            preds, labels, per_class = eval_on_files(ds_files, predictor, label_filter=['Abnormal', 'Normal'])
            if len(preds) == 0:
                continue

            acc = 100 * (preds == labels).sum() / len(labels)
            print(f"\n  [{ds_name}] Accuracy: {acc:.2f}%")
            for cls, (c, t) in per_class.items():
                print(f"    {cls}: {c}/{t} = {100*c/t:.1f}%")

            # 저장
            results.append({
                'experiment': exp['name'],
                'dataset': ds_name,
                'accuracy': acc,
                'per_class': per_class,
            })

    # ── 최종 비교 테이블 ──────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("최종 비교 테이블")
    print(f"{'='*80}")
    print(f"{'실험':<35} {'SPRSound':>10} {'ICBHI':>10}")
    print("-" * 60)

    exp_names = list(dict.fromkeys(r['experiment'] for r in results))
    for name in exp_names:
        spr = next((r for r in results if r['experiment'] == name and r['dataset'] == 'SPRSound'), None)
        icb = next((r for r in results if r['experiment'] == name and r['dataset'] == 'ICBHI'), None)
        spr_str = f"{spr['accuracy']:.2f}%" if spr else "N/A"
        icb_str = f"{icb['accuracy']:.2f}%" if icb else "N/A"
        print(f"  {name:<33} {spr_str:>10} {icb_str:>10}")

    print(f"\n완료!")
