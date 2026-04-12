"""
1D-CNN Beat Detector 학습
=========================
Shannon envelope으로 라벨링된 데이터 → tiny 1D-CNN 학습
→ ONNX 및 TFLite INT8 변환까지 한번에.

입력: 200ms @ 2kHz = 400 samples (raw waveform)
출력: P(beat) sigmoid

모델 크기 목표: INT8 양자화 후 <50KB

사용법:
  C:/Users/dwd00/anaconda3/envs/stetho_ai/python.exe 02_train_beat_detector.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ─── 설정 ───────────────────────────────────────────────
DATA_DIR = "G:/stetho_ai/BPM_ondevice/data"
OUTPUT_DIR = "G:/stetho_ai/BPM_ondevice/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SAMPLES = 400   # 200ms @ 2kHz
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ─── Dataset ────────────────────────────────────────────
class BeatDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float().unsqueeze(1)  # (N, 1, 400)
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ─── Model ──────────────────────────────────────────────
class TinyBeatCNN(nn.Module):
    """
    Tiny 1D-CNN for beat detection.
    Conv1D(1→16, k=7) → Conv1D(16→32, k=5) → Conv1D(32→32, k=3) → GAP → Dense(1)

    Target: <10K parameters → INT8 ~10-20KB
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 400 → 197
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            # Block 2: 197 → 97
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # Block 3: 97 → 47
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # (B, 32, 1)
            nn.Flatten(),              # (B, 32)
            nn.Linear(32, 1),          # (B, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ─── Training ───────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, Y_batch in loader:
        X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, Y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(Y_batch)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == Y_batch).sum().item()
        total += len(Y_batch)

    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)

            logits = model(X_batch)
            loss = criterion(logits, Y_batch)

            total_loss += loss.item() * len(Y_batch)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == Y_batch).sum().item()
            total += len(Y_batch)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(Y_batch.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def main():
    # 1) 데이터 로드
    print("Loading data...")
    X = np.load(os.path.join(DATA_DIR, "X_windows.npy"), mmap_mode='r')
    Y = np.load(os.path.join(DATA_DIR, "Y_labels.npy"), mmap_mode='r')
    print(f"X: {X.shape}, Y: {Y.shape}")
    print(f"Beat ratio: {np.mean(Y):.3f}")

    # 2) Train/Val split
    idx = np.arange(len(X))
    train_idx, val_idx = train_test_split(idx, test_size=0.15, random_state=SEED, stratify=Y)

    X_train, Y_train = X[train_idx].copy(), Y[train_idx].copy()
    X_val, Y_val = X[val_idx].copy(), Y[val_idx].copy()
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # 3) 클래스 불균형 처리 (WeightedRandomSampler)
    n_beat = np.sum(Y_train == 1)
    n_nobeat = np.sum(Y_train == 0)
    weights = np.where(Y_train == 1, 1.0 / n_beat, 1.0 / n_nobeat)
    weights = weights / weights.sum()
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_ds = BeatDataset(X_train, Y_train)
    val_ds = BeatDataset(X_val, Y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    # 4) 모델
    model = TinyBeatCNN().to(DEVICE)
    print(f"\nModel parameters: {count_params(model):,}")
    print(model)

    # 클래스 불균형 반영한 pos_weight
    pos_weight = torch.tensor([n_nobeat / n_beat]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 5) 학습
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\nTraining on {DEVICE}...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_preds, val_labels = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "beat_cnn_best.pth"))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # 6) 최종 평가
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "beat_cnn_best.pth"), weights_only=True))
    _, _, val_preds, val_labels = eval_epoch(model, val_loader, criterion)

    print(f"\n{'='*50}")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=["no-beat", "beat"]))
    print(f"Confusion Matrix:")
    print(confusion_matrix(val_labels, val_preds))

    # 7) ONNX 내보내기
    model.eval()
    dummy = torch.randn(1, 1, WINDOW_SAMPLES).to(DEVICE)
    onnx_path = os.path.join(OUTPUT_DIR, "beat_cnn.onnx")
    torch.onnx.export(model, dummy, onnx_path,
                      input_names=["audio_window"],
                      output_names=["beat_logit"],
                      opset_version=13)
    print(f"\nONNX saved: {onnx_path}")
    print(f"ONNX size: {os.path.getsize(onnx_path) / 1024:.1f} KB")

    # 8) 학습 커브 저장
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["val_loss"], label="val")
    ax1.set_title("Loss")
    ax1.legend()
    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["val_acc"], label="val")
    ax2.set_title("Accuracy")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curve.png"), dpi=120)
    plt.close()
    print(f"Training curve saved: {OUTPUT_DIR}/training_curve.png")

    # 9) 모델 크기 추정
    param_bytes = count_params(model) * 4  # float32
    print(f"\nFloat32 model: {param_bytes/1024:.1f} KB")
    print(f"INT8 예상: ~{param_bytes/1024/4:.1f} KB")


if __name__ == "__main__":
    main()
