"""
프레임 단위 폐음 분류 모델 학습 (100ms 프레임)
- 작은 CNN + RFN (주파수 정규화)
- Focal Loss + SpecAugment + Mixup
- 입력: (64, 41) mel + delta + delta2 → (3, 64, 41)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


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


# ── SpecAugment ───────────────────────────────────────────────
class SpecAugment:
    def __init__(self, freq_mask=8, time_mask=8, n_freq=1, n_time=1):
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq = n_freq
        self.n_time = n_time

    def __call__(self, spec):
        _, n_freq, n_time = spec.shape
        for _ in range(self.n_freq):
            f = np.random.randint(0, self.freq_mask)
            f0 = np.random.randint(0, max(1, n_freq - f))
            spec[:, f0:f0+f, :] = 0
        for _ in range(self.n_time):
            t = np.random.randint(0, self.time_mask)
            t0 = np.random.randint(0, max(1, n_time - t))
            spec[:, :, t0:t0+t] = 0
        return spec


# ── Dataset ───────────────────────────────────────────────────
class FrameDataset(Dataset):
    def __init__(self, data_npy, labels_npy, spec_augment=None):
        self.mels = np.load(data_npy, mmap_mode='r')
        self.labels = np.load(labels_npy)
        self.spec_augment = spec_augment
        print(f"  로드: {os.path.basename(data_npy)} -> {self.mels.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel = self.mels[idx]  # (64, 41)

        delta = np.diff(mel, axis=1, prepend=mel[:, :1])
        delta2 = np.diff(delta, axis=1, prepend=delta[:, :1])
        spec = np.stack([mel, delta, delta2], axis=0)  # (3, 64, 41)

        for c in range(3):
            m = spec[c].mean()
            s = spec[c].std() + 1e-6
            spec[c] = (spec[c] - m) / s

        spec = torch.from_numpy(spec)

        if self.spec_augment is not None:
            spec = self.spec_augment(spec)

        return spec, int(self.labels[idx])


# ── 작은 CNN + RFN ────────────────────────────────────────────
class FrameCNN(nn.Module):
    def __init__(self, num_classes, n_mels=64):
        super().__init__()
        self.rfn = RFN(num_features=n_mels)

        self.conv = nn.Sequential(
            # (3, 64, 41)
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # (32, 32, 20)

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # (64, 16, 10)

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # (128, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.rfn(x)
        x = self.conv(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


# ── Mixup ─────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ══════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════
cache_dir = r"G:\stetho_ai\Lung_classification\cache"
batch_size = 128      # 프레임이 작으니 배치 키움
num_epochs = 50
learning_rate = 0.001
early_stopping_patience = 10
MIXUP_ALPHA = 0.3
MAX_WEIGHT = 3.0

CLASSES = ['Crackle', 'Normal', 'Unknown', 'Wheeze']
num_classes = len(CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 데이터 로드 ───────────────────────────────────────────────
print("\n데이터 로딩:")
spec_aug = SpecAugment(freq_mask=8, time_mask=8, n_freq=1, n_time=1)

train_ds = FrameDataset(os.path.join(cache_dir, 'frame_train_data.npy'),
                        os.path.join(cache_dir, 'frame_train_labels.npy'), spec_aug)
val_ds   = FrameDataset(os.path.join(cache_dir, 'frame_val_data.npy'),
                        os.path.join(cache_dir, 'frame_val_labels.npy'))
test_ds  = FrameDataset(os.path.join(cache_dir, 'frame_test_data.npy'),
                        os.path.join(cache_dir, 'frame_test_labels.npy'))

# ── 가중치 ────────────────────────────────────────────────────
class_counts = np.zeros(num_classes)
for lbl in train_ds.labels:
    class_counts[int(lbl)] += 1

print("\nTrain 분포:")
for i, cls in enumerate(CLASSES):
    print(f"  {cls}: {int(class_counts[i])}")

class_weights = max(class_counts) / class_counts
class_weights = np.clip(class_weights, 1.0, MAX_WEIGHT)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

print("\n가중치:")
for i, cls in enumerate(CLASSES):
    print(f"  {cls}: {class_weights[i]:.2f}")

# ── Sampler + Loader ──────────────────────────────────────────
sample_weights = [class_weights[int(lbl)] for lbl in train_ds.labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# ── 모델 ─────────────────────────────────────────────────────
model = FrameCNN(num_classes=num_classes, n_mels=64).to(device)
print(f"\n모델: FrameCNN + RFN ({sum(p.numel() for p in model.parameters()):,} params)")

criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

best_val_loss = float('inf')
epochs_no_improve = 0
best_model_wts = copy.deepcopy(model.state_dict())

# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("프레임 CNN 학습 (100ms, ICBHI 어노테이션 기반)")
print(f"{'='*60}")

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for specs, labels in train_loader:
        specs, labels = specs.to(device), labels.to(device)
        mixed, y_a, y_b, lam = mixup_data(specs, labels, MIXUP_ALPHA)

        optimizer.zero_grad()
        outputs = model(mixed)
        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (lam * (pred == y_a).sum().item() + (1-lam) * (pred == y_b).sum().item())

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for specs, labels in val_loader:
            specs, labels = specs.to(device), labels.to(device)
            outputs = model(specs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (pred == labels).sum().item()

    val_loss_avg = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    lr = optimizer.param_groups[0]['lr']

    print(f"Epoch [{epoch+1:02d}/{num_epochs}] LR:{lr:.6f} | "
          f"Train L:{train_loss:.4f} A:{train_acc:.1f}% | Val L:{val_loss_avg:.4f} A:{val_acc:.1f}%", end="")

    scheduler.step(val_loss_avg)

    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        epochs_no_improve = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        print(f" *BEST*")
    else:
        epochs_no_improve += 1
        print(f" ({epochs_no_improve}/{early_stopping_patience})")

    if epochs_no_improve >= early_stopping_patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

# ── 저장 ──────────────────────────────────────────────────────
model.load_state_dict(best_model_wts)
save_path = r"G:\stetho_ai\Lung_classification\frame_cnn_lung.pth"
torch.save(model.state_dict(), save_path)
print(f"\n모델 저장: {save_path}")

# ── Test (프레임 단위) ────────────────────────────────────────
print(f"\n{'='*60}")
print("Test 평가 (프레임 단위)")
print(f"{'='*60}")

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for specs, labels in test_loader:
        specs = specs.to(device)
        outputs = model(specs)
        _, pred = torch.max(outputs, 1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print(f"\nFrame Accuracy: {100*(all_preds==all_labels).sum()/len(all_labels):.2f}%")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=3))

print("Confusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
header = "".join(f"{c[:6]:>8}" for c in CLASSES)
print(f"{'':>10}{header}")
for i, cls in enumerate(CLASSES):
    print(f"{cls:>10}{''.join(f'{v:>8}' for v in cm[i])}")
