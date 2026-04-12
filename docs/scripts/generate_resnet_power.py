"""
ResNet의 힘 — 각 레이어가 학습한 잔차(Residual)를 시각화
ResNet: output = F(x) + x → 잔차 F(x) = output - x = "이 레이어가 새로 배운 것"
"""
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import zoom as ndzoom
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

DATASET = r"G:\stetho_ai\_misc\datasets\classification-of-heart-sound-recordings\classification-of-heart-sound-recordings\training-a"
ABNORMAL_FILE = os.path.join(DATASET, "a0001.wav")
MODEL_PATH = r"G:\stetho_ai\server_models\Heart_binary_classification\resnet_cbam_best.pth"

SR = 4000
WINDOW_DURATION = 5
SAMPLES_PER_WIN = SR * WINDOW_DURATION
N_MELS = 64
HOP_LENGTH = 256
FIXED_REF = 1.0
DEVICE = torch.device("cpu")


# ── 모델 (중간값 저장용) ──
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
        out = self.cbam(out)
        # 잔차 = CBAM 출력 (identity 더하기 전)
        self._residual = out.detach()
        self._identity = identity.detach()
        out = F.relu(out + identity)
        self._output = out.detach()
        return out


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
        self._conv1_out = F.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(self._conv1_out)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x_pool = self.avgpool(x3)
        feat = torch.flatten(x_pool, 1)
        out = self.fc(self.drop_fc(feat))
        return out, feat


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


def to_vis(tensor_4d):
    """(1, C, H, W) → 채널 평균 2D numpy"""
    return tensor_4d[0].mean(dim=0).numpy()


def norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)


# ── 모델 로드 & 추론 ──
print("모델 로드 중...")
model = HeartSoundModel(num_classes=3, dropout=0.5).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

mel = load_mel(ABNORMAL_FILE)
tensor = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    model(tensor)

# ── 시각화 ──
print("ResNet 잔차 시각화 생성 중...")

fig, axes = plt.subplots(3, 4, figsize=(22, 15))
fig.patch.set_facecolor('#0d1117')

col_titles = [
    '입력\n(이전 레이어 출력)',
    '잔차 F(x)\n(이 레이어가 새로 학습한 것)',
    '입력 + 잔차\n(합쳐진 결과)',
    '잔차 강도\n(밝을수록 많이 학습)',
]

layers = [
    ('Layer 1\n64ch → 64ch', model.layer1),
    ('Layer 2\n64ch → 128ch', model.layer2),
    ('Layer 3\n128ch → 256ch', model.layer3),
]

# 입력 크기 기준 (리사이즈용)
ref_shape = mel.shape

for row, (layer_name, layer) in enumerate(layers):
    identity = to_vis(layer._identity)
    residual = to_vis(layer._residual)
    output = to_vis(layer._output)

    # 잔차 강도 (절대값)
    residual_strength = np.abs(residual)

    data = [identity, residual, output, residual_strength]
    cmaps = ['cividis', 'PuOr', 'viridis', 'inferno']

    for col in range(4):
        ax = axes[row][col]
        ax.set_facecolor('#161b22')

        d = data[col]
        if col == 1:  # 잔차: 0 중심 대칭
            vmax = np.abs(d).max()
            im = ax.imshow(d, aspect='auto', origin='lower', cmap=cmaps[col],
                           vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(norm(d), aspect='auto', origin='lower', cmap=cmaps[col])

        if row == 0:
            ax.set_title(col_titles[col], fontsize=13, fontweight='bold',
                         color='white', pad=10)
        if col == 0:
            ax.set_ylabel(layer_name, fontsize=13, fontweight='bold',
                          color='#58a6ff', labelpad=10)

        ax.tick_params(colors='#aaa', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#444')

        cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
        cb.ax.tick_params(colors='#aaa', labelsize=7)

# 하단 설명
descriptions = [
    'Layer 1: 기본 패턴\n(윤곽, 에너지 분포)',
    'Layer 2: 중간 패턴\n(박동 리듬, 주파수 구조)',
    'Layer 3: 고수준 패턴\n(정상/비정상 구분 특징)',
]
for i, desc in enumerate(descriptions):
    axes[i][0].text(-0.25, 0.5, desc, transform=axes[i][0].transAxes,
                     fontsize=10, color='#8b949e', ha='right', va='center',
                     rotation=0, linespacing=1.5)

fig.suptitle('ResNet의 힘  —  각 레이어가 새로 학습한 것 (잔차 시각화)',
             fontsize=21, fontweight='bold', color='white', y=0.98)
fig.text(0.5, 0.945,
         'ResNet은 output = F(x) + x 구조로, 각 레이어가 "변화량(잔차)"만 학습합니다. 깊어질수록 더 추상적인 패턴을 잡아냅니다.',
         fontsize=12, color='#8b949e', ha='center')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(r"G:\stetho_ai\resnet_residual_power.png", dpi=300,
            bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved: resnet_residual_power.png")
plt.close()
