import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
import librosa
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import copy

# 1. 경로 및 장치 설정
CACHE_DIR = r"G:\stetho_ai\Heart_binary_classification"
MODEL_PATH = os.path.join(CACHE_DIR, 'resnet_cbam_best.pth')
XGB_PATH = os.path.join(CACHE_DIR, 'heart_xgb_model.json')

RAW_FILE = r"G:\stetho_ai\raw_heart_sound.wav"
MATCHED_FILE = r"G:\stetho_ai\matched_heart_sound.wav"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 장치: {DEVICE}")

# 전처리 파라미터
SR = 4000
WINDOW_DURATION = 5
SAMPLES_PER_WINDOW = SR * WINDOW_DURATION
HOP_LENGTH = 256
N_MELS = 64


# ----------------------------------------------------------------
# 2. 모델 클래스 정의
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
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(HeartSoundModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = ResNetBlock(64, 64, dropout_rate=0.2)
        self.layer2 = ResNetBlock(64, 128, stride=2, dropout_rate=0.2)
        self.layer3 = ResNetBlock(128, 256, stride=2, dropout_rate=0.3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
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
# 3. 모델 로드 또는 학습 결정
# ----------------------------------------------------------------
model = HeartSoundModel(num_classes=2, dropout_rate=0.5).to(DEVICE)
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, tree_method='hist',
                    device='cuda')

if os.path.exists(MODEL_PATH) and os.path.exists(XGB_PATH):
    print("\n✅ 기존 모델 발견! 파일을 불러와 바로 검증을 시작합니다.")
    model.load_state_dict(torch.load(MODEL_PATH))
    xgb.load_model(XGB_PATH)
    needs_training = False
else:
    print("\n❌ 기존 모델 없음. 신규 학습을 시작합니다.")
    needs_training = True

# ----------------------------------------------------------------
# 4. Phase 1~3: 학습 (필요한 경우에만 실행)
# ----------------------------------------------------------------
if needs_training:
    print("데이터 로딩 중...")
    X = np.load(os.path.join(CACHE_DIR, 'x_data.npy'))
    y = np.load(os.path.join(CACHE_DIR, 'y_data.npy'))
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n--- Phase 1: CBAM-ResNet Training ---")
    epochs, patience = 100, 7
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoch [{epoch + 1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss, best_model_wts = avg_val_loss, copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), MODEL_PATH)

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
    xgb.fit(X_train_xgb, y_train_xgb)
    xgb.save_model(X_PATH)

    y_pred = xgb.predict(X_val_xgb)
    tn, fp, fn, tp = confusion_matrix(y_val_xgb, y_pred).ravel()
    print(f"\n[최종 검증 결과] Accuracy: {accuracy_score(y_val_xgb, y_pred):.4f}")
    print(f"민감도: {tp / (tp + fn):.4f} | 특이도: {tn / (tn + fp):.4f}")


# ----------------------------------------------------------------
# 5. Phase 4: 외부 파일 확률 분석 로직
# ----------------------------------------------------------------
def process_new_audio(file_path):
    try:
        y, _ = librosa.load(file_path, sr=SR)
        segments = []
        if len(y) < SAMPLES_PER_WINDOW:
            segments.append(np.pad(y, (0, SAMPLES_PER_WINDOW - len(y))))
        else:
            for start in range(0, len(y) - SAMPLES_PER_WINDOW + 1, SR * 2):
                segments.append(y[start: start + SAMPLES_PER_WINDOW])
        mels = [librosa.power_to_db(librosa.feature.melspectrogram(y=s, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH),
                                    ref=np.max) for s in segments]
        return np.array(mels)
    except Exception as e:
        print(f"파일 처리 에러: {e}");
        return None


def single_file_inference(file_path, name):
    if not os.path.exists(file_path):
        print(f"\n[{name}] 파일을 찾을 수 없습니다.")
        return
    mels = process_new_audio(file_path)
    if mels is None: return

    mels_t = torch.FloatTensor(mels).unsqueeze(1).to(DEVICE)
    model.eval()
    with torch.no_grad():
        _, f = model(mels_t)
        f_np = f.cpu().numpy()

    # 확률 예측 (predict_proba)
    probs = xgb.predict_proba(f_np)  # [[정상확률, 비정상확률], ...]
    avg_probs = np.mean(probs, axis=0)

    prob_normal = avg_probs[0] * 100
    prob_abnormal = avg_probs[1] * 100
    res = "비정상 (Abnormal)" if prob_abnormal >= 50 else "정상 (Normal)"

    print(f"\n--- [{name}] 분석 보고서 ---")
    print(f"최종 판정: {res}")
    print(f"정상 확률: {prob_normal:.2f}%")
    print(f"비정상 확률: {prob_abnormal:.2f}%")
    print(f"구간별 판단(0정상/1비정상): {xgb.predict(f_np)}")


print("\n--- Phase 4: External Validation with Probabilities ---")
single_file_inference(RAW_FILE, "RAW_SOUND")
single_file_inference(MATCHED_FILE, "MATCHED_SOUND")