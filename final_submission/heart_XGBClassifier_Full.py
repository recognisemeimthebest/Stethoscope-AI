import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import os
import librosa
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import copy

# 1. 경로 및 장치 설정
CACHE_DIR = r"G:\stetho_ai\Heart_binary_classification"
MODEL_PATH = os.path.join(CACHE_DIR, 'resnet_cbam_v17.pth')
XGB_PATH = os.path.join(CACHE_DIR, 'heart_xgb_v17.json')

RAW_FILE = r"G:\stetho_ai\raw_heart_sound.wav"
MATCHED_FILE = r"G:\stetho_ai\matched_heart_sound.wav"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# 전처리 파라미터
SR = 4000
WINDOW_DURATION = 5
SAMPLES_PER_WINDOW = SR * WINDOW_DURATION
HOP_LENGTH = 256
N_MELS = 64


# ----------------------------------------------------------------
# 2. SpecAugment (주파수/시간 마스킹)
# ----------------------------------------------------------------
class SpecAugment(nn.Module):
    """학습 시에만 적용되는 스펙트로그램 증강"""
    def __init__(self, freq_mask_param=8, time_mask_param=10, n_freq_masks=2, n_time_masks=2):
        super().__init__()
        self.freq_mask_param = freq_mask_param   # 최대 마스킹할 주파수 bin 수
        self.time_mask_param = time_mask_param   # 최대 마스킹할 시간 frame 수
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, x):
        if not self.training:
            return x
        # x shape: (batch, 1, n_mels, n_frames)
        cloned = x.clone()
        _, _, n_mels, n_frames = cloned.shape

        for _ in range(self.n_freq_masks):
            f = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
            f0 = torch.randint(0, max(1, n_mels - f), (1,)).item()
            cloned[:, :, f0:f0 + f, :] = 0

        for _ in range(self.n_time_masks):
            t = torch.randint(0, self.time_mask_param + 1, (1,)).item()
            t0 = torch.randint(0, max(1, n_frames - t), (1,)).item()
            cloned[:, :, :, t0:t0 + t] = 0

        return cloned


# ----------------------------------------------------------------
# 3. 모델 클래스 정의
# ----------------------------------------------------------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = self.sigmoid_channel(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)
        x = x * out
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial = self.sigmoid_spatial(self.conv_spatial(spatial))
        x = x * spatial
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.2):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cbam = CBAM(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class HeartSoundModel(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(HeartSoundModel, self).__init__()
        self.spec_augment = SpecAugment(freq_mask_param=8, time_mask_param=10,
                                        n_freq_masks=2, n_time_masks=2)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = ResNetBlock(64, 64, dropout_rate=0.2)
        self.layer2 = ResNetBlock(64, 128, stride=2, dropout_rate=0.2)
        self.layer3 = ResNetBlock(128, 256, stride=2, dropout_rate=0.3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.spec_augment(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        x = self.dropout_fc(features)
        out = self.fc(x)
        return out, features


# ----------------------------------------------------------------
# 4. 학습
# ----------------------------------------------------------------
model = HeartSoundModel(num_classes=3, dropout_rate=0.5).to(DEVICE)
xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42,
                    tree_method='hist', device='cuda',
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)

print("\n[V17] File-level split + SpecAugment + CosineAnnealing + LabelSmoothing")
print("=" * 70)

print("Loading data...")
X = np.load(os.path.join(CACHE_DIR, 'x_data.npy'))
y = np.load(os.path.join(CACHE_DIR, 'y_data.npy'))
file_ids = np.load(os.path.join(CACHE_DIR, 'file_ids.npy'))

X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)
dataset = TensorDataset(X_tensor, y_tensor)

# 파일 단위 분할
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X, y, groups=file_ids))

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

train_files = set(file_ids[train_idx])
val_files = set(file_ids[val_idx])
leak = train_files & val_files
print(f"[File Split] Train: {len(train_files)} files ({len(train_idx)} segs), "
      f"Val: {len(val_files)} files ({len(val_idx)} segs), Leak: {len(leak)}")
assert len(leak) == 0

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

# Label Smoothing CrossEntropy
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Adam + Weight Decay
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Cosine Annealing LR Scheduler
epochs = 150
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

patience = 20
best_val_loss = float('inf')
early_stop_counter = 0
best_model_wts = copy.deepcopy(model.state_dict())
best_epoch = 0

print(f"\n--- Phase 1: CBAM-ResNet Training (SpecAugment + CosineAnneal + LabelSmoothing) ---")
print(f"    Epochs: {epochs}, Patience: {patience}, LR: 0.001 -> 1e-6")

for epoch in range(epochs):
    # Train
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    scheduler.step()
    avg_train_loss = train_loss / train_total
    train_acc = train_correct / train_total

    # Val
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    lr_now = scheduler.get_last_lr()[0]

    if (epoch + 1) % 5 == 0 or epoch < 3 or avg_val_loss < best_val_loss:
        print(f"Epoch [{epoch+1:3d}/{epochs}] "
              f"TrLoss: {avg_train_loss:.4f} TrAcc: {train_acc:.4f} | "
              f"VaLoss: {avg_val_loss:.4f} VaAcc: {val_acc:.4f} | "
              f"LR: {lr_now:.6f}"
              f"{' *BEST*' if avg_val_loss < best_val_loss else ''}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        best_epoch = epoch + 1
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (best was epoch {best_epoch})")
            break

print(f"\nBest epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), MODEL_PATH)


# ----------------------------------------------------------------
# Phase 2 & 3: Feature Extraction + XGBoost
# ----------------------------------------------------------------
print("\n--- Phase 2 & 3: Feature Extraction & XGBoost Training ---")
model.eval()


def get_features(loader):
    feats, lbls = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            _, f = model(inputs.to(DEVICE))
            feats.append(f.cpu().numpy())
            lbls.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(lbls)


X_train_xgb, y_train_xgb = get_features(train_loader)
X_val_xgb, y_val_xgb = get_features(val_loader)
xgb.fit(X_train_xgb, y_train_xgb,
        eval_set=[(X_val_xgb, y_val_xgb)],
        verbose=False)
xgb.save_model(XGB_PATH)

y_pred = xgb.predict(X_val_xgb)
cm = confusion_matrix(y_val_xgb, y_pred)

# 성능 계산
sens = cm[1, 1] / cm[1].sum() if cm.shape[0] > 1 else 0
spec = cm[0, 0] / cm[0].sum() if cm.shape[0] > 0 else 0
macc = (sens + spec) / 2

print(f"\n{'='*70}")
print(f"  V17 Final Results (File-Level Split, No Data Leakage)")
print(f"{'='*70}")
print(f"  Accuracy:    {accuracy_score(y_val_xgb, y_pred)*100:.2f}%")
print(f"  Sensitivity: {sens*100:.2f}%")
print(f"  Specificity: {spec*100:.2f}%")
print(f"  MACC:        {macc*100:.2f}%")
print(f"\nConfusion Matrix:\n{cm}")
print(classification_report(y_val_xgb, y_pred,
                            target_names=["Normal", "Abnormal", "Unknown"]))


# ----------------------------------------------------------------
# Phase 4: External Validation
# ----------------------------------------------------------------
def process_new_audio(file_path):
    try:
        y, _ = librosa.load(file_path, sr=SR)
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak
        segments = []
        if len(y) < SAMPLES_PER_WINDOW:
            segments.append(np.pad(y, (0, SAMPLES_PER_WINDOW - len(y))))
        else:
            for start in range(0, len(y) - SAMPLES_PER_WINDOW + 1, SR * 2):
                segments.append(y[start: start + SAMPLES_PER_WINDOW])
        mels = [librosa.power_to_db(librosa.feature.melspectrogram(y=s, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH),
                                    ref=1.0) for s in segments]
        return np.array(mels)
    except Exception as e:
        print(f"File error: {e}")
        return None


def single_file_inference(file_path, name):
    if not os.path.exists(file_path):
        print(f"\n[{name}] File not found.")
        return
    mels = process_new_audio(file_path)
    if mels is None:
        return

    mels_t = torch.FloatTensor(mels).unsqueeze(1).to(DEVICE)
    model.eval()
    with torch.no_grad():
        _, f = model(mels_t)
        f_np = f.cpu().numpy()

    probs = xgb.predict_proba(f_np)
    avg_probs = np.mean(probs, axis=0)

    prob_normal = avg_probs[0] * 100
    prob_abnormal = avg_probs[1] * 100
    prob_unknown = avg_probs[2] * 100

    top_idx = np.argmax(avg_probs)
    labels = ["Normal", "Abnormal", "Unknown"]
    res = labels[top_idx]

    print(f"\n--- [{name}] ---")
    print(f"  Result: {res}")
    print(f"  Normal: {prob_normal:.2f}%, Abnormal: {prob_abnormal:.2f}%, Unknown: {prob_unknown:.2f}%")
    print(f"  Per-segment: {xgb.predict(f_np)}")


print("\n--- Phase 4: External Validation ---")
single_file_inference(RAW_FILE, "RAW_SOUND")
single_file_inference(MATCHED_FILE, "MATCHED_SOUND")
