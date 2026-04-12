"""
CBAM 적용 전후 효과 비교 시각화
정상/비정상 심음 각각에 대해 CBAM 전 vs 후 vs 차이를 보여줌
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

# ── 경로 ──
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


# ── 모델 정의 ──
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
        self.channel_att = None
        self.spatial_att = None

    def forward(self, x):
        avg = self.fc(self.avg_pool(x).view(x.size(0), -1))
        mx = self.fc(self.max_pool(x).view(x.size(0), -1))
        ch = self.sigmoid_channel(avg + mx).view(x.size(0), x.size(1), 1, 1)
        self.channel_att = ch.detach()
        x = x * ch
        sp = torch.cat([torch.mean(x, 1, keepdim=True),
                        torch.max(x, 1, keepdim=True)[0]], dim=1)
        spatial = self.sigmoid_spatial(self.conv_spatial(sp))
        self.spatial_att = spatial.detach()
        x = x * spatial
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
        self.before_cbam = None
        self.after_cbam = None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop1(out)
        out = self.bn2(self.conv2(out))
        self.before_cbam = out.detach()
        out = self.cbam(out)
        self.after_cbam = out.detach()
        out += self.shortcut(x)
        return F.relu(out)


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
        x0 = F.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)
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
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH),
        ref=FIXED_REF
    )
    return mel


# ── 모델 로드 ──
print("모델 로드 중...")
model = HeartSoundModel(num_classes=3, dropout=0.5).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

mel_normal = load_mel(NORMAL_FILE)
mel_abnormal = load_mel(ABNORMAL_FILE)
tensor_normal = torch.FloatTensor(mel_normal).unsqueeze(0).unsqueeze(0).to(DEVICE)
tensor_abnormal = torch.FloatTensor(mel_abnormal).unsqueeze(0).unsqueeze(0).to(DEVICE)


# ── 시각화 ──
print("CBAM 효과 비교 이미지 생성 중...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.patch.set_facecolor('#0d1117')

samples = [
    (tensor_normal, "정상 심음 (Normal)", "#58a6ff", 0),
    (tensor_abnormal, "비정상 심음 (Abnormal)", "#f0883e", 1),
]

for tensor_data, label, color, row in samples:
    with torch.no_grad():
        model(tensor_data)

    # layer1 사용 (해상도 가장 높음, 디테일 보임)
    before = model.layer1.before_cbam[0].mean(dim=0).numpy()
    after = model.layer1.after_cbam[0].mean(dim=0).numpy()

    # 각각 독립 정규화 (자체 min-max로 디테일 살리기)
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-10)
    before_vis = norm(before)
    after_vis = norm(after)

    # 차이맵
    diff = after - before
    # 정규화: 강조된 곳은 양수, 억제된 곳은 음수
    diff_abs_max = np.abs(diff).max() + 1e-10

    # ── CBAM 전 ──
    ax1 = axes[row][0]
    ax1.set_facecolor('#161b22')
    im1 = ax1.imshow(before_vis, aspect='auto', origin='lower', cmap='viridis')
    if row == 0:
        ax1.set_title('CBAM 적용 전\n(ResNet만 거친 상태)', fontsize=15,
                       fontweight='bold', color='white', pad=12)
    ax1.set_ylabel(label, fontsize=14, fontweight='bold', color=color, labelpad=10)
    ax1.tick_params(colors='#aaa')
    for spine in ax1.spines.values():
        spine.set_color('#555')
    cb1 = fig.colorbar(im1, ax=ax1, pad=0.02, shrink=0.85)
    cb1.ax.tick_params(colors='#aaa')

    # 설명
    ax1.text(0.5, -0.08, '모든 영역을 비슷한 강도로 봄',
             transform=ax1.transAxes, fontsize=11, color='#8b949e',
             ha='center', va='top')

    # ── CBAM 후 ──
    ax2 = axes[row][1]
    ax2.set_facecolor('#161b22')
    im2 = ax2.imshow(after_vis, aspect='auto', origin='lower', cmap='plasma')
    if row == 0:
        ax2.set_title('CBAM 적용 후\n(중요 영역만 강조)', fontsize=15,
                       fontweight='bold', color='white', pad=12)
    ax2.tick_params(colors='#aaa')
    for spine in ax2.spines.values():
        spine.set_color('#555')
    cb2 = fig.colorbar(im2, ax=ax2, pad=0.02, shrink=0.85)
    cb2.ax.tick_params(colors='#aaa')

    ax2.text(0.5, -0.08, '핵심 영역은 강조, 나머지는 억제',
             transform=ax2.transAxes, fontsize=11, color='#8b949e',
             ha='center', va='top')

    # ── 차이맵 ──
    ax3 = axes[row][2]
    ax3.set_facecolor('#161b22')
    im3 = ax3.imshow(diff, aspect='auto', origin='lower', cmap='PuOr',
                      vmin=-diff_abs_max, vmax=diff_abs_max)
    if row == 0:
        ax3.set_title('CBAM이 바꾼 것\n(밝음=강조 / 어두움=억제)', fontsize=15,
                       fontweight='bold', color='white', pad=12)
    ax3.tick_params(colors='#aaa')
    for spine in ax3.spines.values():
        spine.set_color('#555')
    cb3 = fig.colorbar(im3, ax=ax3, pad=0.02, shrink=0.85)
    cb3.ax.tick_params(colors='#aaa')

    # 강조/억제 비율 계산
    enhanced = (diff > 0).sum()
    suppressed = (diff <= 0).sum()
    total = diff.size
    ax3.text(0.5, -0.08,
             f'강조: {enhanced/total*100:.0f}%  |  억제: {suppressed/total*100:.0f}%',
             transform=ax3.transAxes, fontsize=11, color='#8b949e',
             ha='center', va='top')

fig.suptitle('CBAM의 효과  —  "있을 때 vs 없을 때" 비교',
             fontsize=20, fontweight='bold', color='white', y=0.98)
fig.text(0.5, 0.94,
         'CBAM은 모든 영역을 동등하게 보지 않고, 진단에 중요한 영역만 선택적으로 집중합니다',
         fontsize=13, color='#8b949e', ha='center')

plt.tight_layout(rect=[0, 0.02, 1, 0.92])
plt.savefig(r"G:\stetho_ai\cbam_effect_comparison.png", dpi=300,
            bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved: cbam_effect_comparison.png")
plt.close()
