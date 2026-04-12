"""
CBAM만 거친 것 vs ResNet(잔차)+CBAM 거친 것 feature map 비교
같은 입력에 대해:
- CBAM 출력만 (잔차 연결 전)
- ResNet 출력 (잔차 연결 후 = CBAM출력 + 원본)
이 둘을 비교하면 "ResNet이 왜 필요한지" 보임
"""
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

DATASET = r"G:\stetho_ai\_misc\datasets\classification-of-heart-sound-recordings\classification-of-heart-sound-recordings\training-a"
NORMAL_FILE = os.path.join(DATASET, "a0007.wav")
ABNORMAL_FILE = os.path.join(DATASET, "a0001.wav")
MODEL_PATH = r"G:\stetho_ai\server_models\Heart_binary_classification\resnet_cbam_best.pth"

SR = 4000
WINDOW_DURATION = 5
SAMPLES_PER_WIN = SR * WINDOW_DURATION
N_MELS = 64
HOP_LENGTH = 256
FIXED_REF = 1.0
DEVICE = torch.device("cpu")


# ── 모델 ──
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x).view(x.size(0), -1))
        mx = self.fc(self.max_pool(x).view(x.size(0), -1))
        ch = self.sigmoid_channel(avg + mx).view(x.size(0), x.size(1), 1, 1)
        x = x * ch
        sp = torch.cat([torch.mean(x, 1, keepdim=True),
                        torch.max(x, 1, keepdim=True)[0]], dim=1)
        x = x * self.sigmoid_spatial(self.conv_spatial(sp))
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.drop1 = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.cbam = CBAM(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop1(out)
        out = self.bn2(self.conv2(out))
        cbam_only = self.cbam(out)
        # 저장
        self._identity = identity.detach()
        self._cbam_only = cbam_only.detach()
        self._resnet_out = F.relu(cbam_only + identity).detach()
        return F.relu(cbam_only + identity)


class HeartSoundModel(nn.Module):
    def __init__(self, num_classes=3, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = ResNetBlock(64, 64, dropout=0.2)
        self.layer2 = ResNetBlock(64, 128, stride=2, dropout=0.2)
        self.layer3 = ResNetBlock(128, 256, stride=2, dropout=0.3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop_fc = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        return self.fc(self.drop_fc(feat)), feat


def load_mel(wav_path):
    y, _ = librosa.load(wav_path, sr=SR, duration=WINDOW_DURATION)
    if len(y) < SAMPLES_PER_WIN:
        y = np.pad(y, (0, SAMPLES_PER_WIN - len(y)))
    else:
        y = y[:SAMPLES_PER_WIN]
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH),
        ref=FIXED_REF
    )


def to_vis(t):
    return t[0].mean(dim=0).numpy()


def norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)


# ── 로드 ──
print("모델 로드 중...")
model = HeartSoundModel(num_classes=3, dropout=0.5).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ── 시각화 ──
print("이미지 생성 중...")

fig, axes = plt.subplots(2, 4, figsize=(22, 11))
fig.patch.set_facecolor('#0d1117')

col_titles = [
    '입력\n(원본 Mel-Spectrogram)',
    'CBAM만 적용\n(잔차 연결 없음)',
    'ResNet + CBAM\n(잔차 연결 있음)',
    '차이 (ResNet - CBAM)\n= 잔차가 보존한 정보',
]

samples = [
    (NORMAL_FILE, "정상 심음", "#58a6ff", 0),
    (ABNORMAL_FILE, "비정상 심음", "#f0883e", 1),
]

for wav_path, label, color, row in samples:
    mel = load_mel(wav_path)
    tensor = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        model(tensor)

    # layer1 기준 (해상도 가장 높음)
    identity = to_vis(model.layer1._identity)
    cbam_only = to_vis(model.layer1._cbam_only)
    resnet_out = to_vis(model.layer1._resnet_out)
    diff = resnet_out - cbam_only  # 잔차가 더해준 부분

    data = [norm(identity), norm(cbam_only), norm(resnet_out), diff]
    cmaps = ['cividis', 'plasma', 'viridis', 'PuOr']

    for col in range(4):
        ax = axes[row][col]
        ax.set_facecolor('#161b22')

        if col == 3:
            vmax = np.abs(diff).max()
            im = ax.imshow(data[col], aspect='auto', origin='lower',
                           cmap=cmaps[col], vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(data[col], aspect='auto', origin='lower',
                           cmap=cmaps[col])

        if row == 0:
            ax.set_title(col_titles[col], fontsize=13, fontweight='bold',
                         color='white', pad=10)
        if col == 0:
            ax.set_ylabel(label, fontsize=14, fontweight='bold',
                          color=color, labelpad=10)

        ax.tick_params(colors='#aaa', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#444')
        cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
        cb.ax.tick_params(colors='#aaa', labelsize=7)

# 하단 설명
bottom_texts = [
    '',
    'CBAM은 중요한 영역을 강조하지만\n원본 정보를 일부 잃을 수 있음',
    'ResNet 잔차 연결이 원본 정보를\n보존하면서 CBAM 강조도 유지',
    '밝은 부분 = ResNet이 살려준 정보\n(CBAM만으로는 사라졌을 디테일)',
]
for col in range(4):
    axes[1][col].text(0.5, -0.12, bottom_texts[col],
                       transform=axes[1][col].transAxes, fontsize=10,
                       color='#8b949e', ha='center', va='top', linespacing=1.4)

fig.suptitle('CBAM만 vs ResNet+CBAM  —  잔차 연결이 만드는 차이',
             fontsize=21, fontweight='bold', color='white', y=0.98)
fig.text(0.5, 0.94,
         'CBAM이 중요한 곳을 강조할 때, ResNet의 잔차 연결이 원본 정보를 보존해서 디테일 손실을 방지합니다',
         fontsize=12, color='#8b949e', ha='center')

plt.tight_layout(rect=[0, 0.03, 1, 0.92])
plt.savefig(r"G:\stetho_ai\cbam_vs_resnet_cbam.png", dpi=300,
            bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved: cbam_vs_resnet_cbam.png")
plt.close()
