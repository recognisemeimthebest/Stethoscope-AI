"""
폐음 3클래스 모델 학습 — 다양한 아키텍처 지원
사용법:
  python train_variants.py resnet        # ResNet18 + RFN (기본)
  python train_variants.py cbam          # ResNet18 + CBAM + RFN
  python train_variants.py mobilenet     # MobileNetV2 + RFN
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
import copy


# ── Focal Loss ────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── RFN ───────────────────────────────────────────────────────
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


# ── CBAM ──────────────────────────────────────────────────────
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


# ── 모델들 ────────────────────────────────────────────────────
class ResNet18_RFN(nn.Module):
    def __init__(self, num_classes, n_mels=128):
        super().__init__()
        self.rfn = RFN(num_features=n_mels)
        resnet = models.resnet18(weights='DEFAULT')
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
        resnet = models.resnet18(weights='DEFAULT')
        layers = list(resnet.children())
        self.stem = nn.Sequential(*layers[:4])       # conv1+bn+relu+pool
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
        mob = models.mobilenet_v2(weights='DEFAULT')
        self.features = mob.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, num_classes))
    def forward(self, x):
        x = self.rfn(x)
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x.flatten(1))


# ── SpecAugment ───────────────────────────────────────────────
class SpecAugment:
    def __init__(self, freq_mask=15, time_mask=40, n_freq=2, n_time=2):
        self.freq_mask, self.time_mask = freq_mask, time_mask
        self.n_freq, self.n_time = n_freq, n_time
    def __call__(self, spec):
        _, nf, nt = spec.shape
        for _ in range(self.n_freq):
            f = np.random.randint(0, self.freq_mask)
            f0 = np.random.randint(0, max(1, nf - f))
            spec[:, f0:f0+f, :] = 0
        for _ in range(self.n_time):
            t = np.random.randint(0, self.time_mask)
            t0 = np.random.randint(0, max(1, nt - t))
            spec[:, :, t0:t0+t] = 0
        return spec


# ── Dataset ───────────────────────────────────────────────────
class LungCacheDataset(Dataset):
    def __init__(self, data_npy, labels_npy, spec_augment=None):
        self.mels = np.load(data_npy, mmap_mode='r')
        self.labels = np.load(labels_npy)
        self.spec_augment = spec_augment
        print(f"  로드: {os.path.basename(data_npy)} -> {self.mels.shape}")
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
        spec = torch.from_numpy(spec)
        if self.spec_augment:
            spec = self.spec_augment(spec)
        return spec, int(self.labels[idx])


# ── Mixup ─────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ══════════════════════════════════════════════════════════════
def train_model(arch_name, save_name):
    cache_dir = r"G:\stetho_ai\Lung_classification\cache"
    CLASSES = ['Abnormal', 'Normal', 'Unknown']
    num_classes = len(CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"아키텍처: {arch_name}")
    print(f"저장: {save_name}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    spec_aug = SpecAugment()
    train_ds = LungCacheDataset(os.path.join(cache_dir, 'train_data.npy'),
                                os.path.join(cache_dir, 'train_labels.npy'), spec_aug)
    val_ds = LungCacheDataset(os.path.join(cache_dir, 'val_data.npy'),
                              os.path.join(cache_dir, 'val_labels.npy'))
    test_ds = LungCacheDataset(os.path.join(cache_dir, 'test_data.npy'),
                               os.path.join(cache_dir, 'test_labels.npy'))

    # 가중치
    class_counts = np.zeros(num_classes)
    for lbl in train_ds.labels:
        class_counts[int(lbl)] += 1
    print("\nTrain 분포:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {int(class_counts[i])}")
    cw = np.clip(max(class_counts) / class_counts, 1.0, 3.0)
    cw_tensor = torch.FloatTensor(cw).to(device)

    sample_weights = [cw[int(lbl)] for lbl in train_ds.labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # 모델
    if arch_name == 'resnet':
        model = ResNet18_RFN(num_classes).to(device)
    elif arch_name == 'cbam':
        model = ResNet18_CBAM_RFN(num_classes).to(device)
    elif arch_name == 'mobilenet':
        model = MobileNetV2_RFN(num_classes).to(device)
    else:
        raise ValueError(f"Unknown arch: {arch_name}")

    params = sum(p.numel() for p in model.parameters())
    print(f"모델: {arch_name} ({params:,} params)")

    criterion = FocalLoss(alpha=cw_tensor, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_wts = copy.deepcopy(model.state_dict())

    for epoch in range(50):
        model.train()
        rloss, correct, total = 0.0, 0, 0
        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            mixed, ya, yb, lam = mixup_data(specs, labels, 0.3)
            optimizer.zero_grad()
            out = model(mixed)
            loss = mixup_criterion(criterion, out, ya, yb, lam)
            loss.backward()
            optimizer.step()
            rloss += loss.item()
            _, pred = torch.max(out, 1)
            total += labels.size(0)
            correct += (lam * (pred == ya).sum().item() + (1 - lam) * (pred == yb).sum().item())

        tl = rloss / len(train_loader)
        ta = 100 * correct / total

        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                out = model(specs)
                vl += criterion(out, labels).item()
                _, pred = torch.max(out, 1)
                vt += labels.size(0)
                vc += (pred == labels).sum().item()

        vla = vl / len(val_loader)
        va = 100 * vc / vt
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1:02d}/50] LR:{lr:.6f} | Train L:{tl:.4f} A:{ta:.1f}% | Val L:{vla:.4f} A:{va:.1f}%", end="")
        scheduler.step(vla)

        if vla < best_val_loss:
            best_val_loss = vla
            epochs_no_improve = 0
            best_wts = copy.deepcopy(model.state_dict())
            print(" *BEST*")
        else:
            epochs_no_improve += 1
            print(f" ({epochs_no_improve}/10)")

        if epochs_no_improve >= 10:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_wts)
    save_path = os.path.join(r"G:\stetho_ai\Lung_classification", save_name)
    torch.save(model.state_dict(), save_path)
    print(f"\n저장: {save_path}")

    # Test
    model.eval()
    ap, al = [], []
    with torch.no_grad():
        for specs, labels in test_loader:
            out = model(specs.to(device))
            _, pred = torch.max(out, 1)
            ap.extend(pred.cpu().numpy())
            al.extend(labels.numpy())
    ap, al = np.array(ap), np.array(al)

    print(f"\nTest Accuracy: {100*(ap==al).sum()/len(al):.2f}%")
    print(classification_report(al, ap, target_names=CLASSES, digits=3))
    return save_path


if __name__ == '__main__':
    arch = sys.argv[1] if len(sys.argv) > 1 else 'resnet'
    save = sys.argv[2] if len(sys.argv) > 2 else f'{arch}_lung.pth'
    train_model(arch, save)
