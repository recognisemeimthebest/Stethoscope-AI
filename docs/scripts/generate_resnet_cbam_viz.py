"""
ResNet+CBAM 시각화
1) CBAM Attention Map 오버레이 (정상 vs 비정상)
2) Feature Map 단계별 시각화 (파이프라인)
"""
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
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


# ── 모델 정의 (hook용으로 중간값 저장) ──
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

        # 시각화용 저장
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


# ── 전처리 ──
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

# ── 데이터 준비 ──
mel_normal = load_mel(NORMAL_FILE)
mel_abnormal = load_mel(ABNORMAL_FILE)

tensor_normal = torch.FloatTensor(mel_normal).unsqueeze(0).unsqueeze(0).to(DEVICE)
tensor_abnormal = torch.FloatTensor(mel_abnormal).unsqueeze(0).unsqueeze(0).to(DEVICE)


# =====================================================
# 시각화 1: CBAM Attention Map (3행: 원본 / attention / 하이라이트)
# =====================================================
print("시각화 1: CBAM Attention Map 생성 중...")
from scipy.ndimage import zoom as ndzoom

fig, axes = plt.subplots(3, 2, figsize=(18, 16),
                         gridspec_kw={'height_ratios': [1, 1, 1]})
fig.patch.set_facecolor('#1a1a2e')

for col, (mel_data, tensor_data, title, color) in enumerate([
    (mel_normal, tensor_normal, "정상 심음 (Normal)", "#58a6ff"),
    (mel_abnormal, tensor_abnormal, "비정상 심음 (Abnormal)", "#f0883e"),
]):
    with torch.no_grad():
        model(tensor_data)

    # 3개 레이어 attention을 합산 (다양한 스케일 반영)
    atts = []
    for layer in [model.layer1, model.layer2, model.layer3]:
        sa = layer.cbam.spatial_att[0, 0].numpy()
        sa_resized = ndzoom(sa,
                            (mel_data.shape[0] / sa.shape[0],
                             mel_data.shape[1] / sa.shape[1]),
                            order=1)
        atts.append(sa_resized)
    combined_att = np.mean(atts, axis=0)
    # min-max 정규화로 contrast 확보
    combined_att = (combined_att - combined_att.min()) / (combined_att.max() - combined_att.min() + 1e-10)
    att_resized = combined_att

    ext = [0, WINDOW_DURATION, 0, N_MELS]

    # ── 1행: 원본 Mel-Spectrogram ──
    ax1 = axes[0][col]
    ax1.set_facecolor('#16213e')
    ax1.imshow(mel_data, aspect='auto', origin='lower', cmap='cividis', extent=ext)
    ax1.set_title(title, fontsize=16, fontweight='bold', color=color, pad=10)
    ax1.set_ylabel('Mel 주파수', fontsize=11, color='white')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_color('#444')
    row1_desc = "원본 Mel-Spectrogram\n(AI에 입력되는 데이터)"
    ax1.text(0.02, 0.95, row1_desc, transform=ax1.transAxes,
             fontsize=10, color='white', ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#555', alpha=0.5))

    # ── 2행: Spatial Attention 단독 ──
    ax2 = axes[1][col]
    ax2.set_facecolor('#16213e')
    im2 = ax2.imshow(att_resized, aspect='auto', origin='lower', cmap='inferno', extent=ext)
    ax2.set_ylabel('Mel 주파수', fontsize=11, color='white')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('#444')
    cb = fig.colorbar(im2, ax=ax2, pad=0.02, shrink=0.9)
    cb.ax.tick_params(colors='white')
    cb.set_label('집중도', color='white', fontsize=10)
    row2_desc = "CBAM Spatial Attention\n(밝을수록 AI가 집중)"
    ax2.text(0.02, 0.95, row2_desc, transform=ax2.transAxes,
             fontsize=10, color='white', ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#555', alpha=0.5))

    # ── 3행: 원본 + attention 강한 영역 윤곽선 표시 ──
    ax3 = axes[2][col]
    ax3.set_facecolor('#16213e')
    ax3.imshow(mel_data, aspect='auto', origin='lower', cmap='cividis', extent=ext)

    # attention 상위 영역 contour
    threshold = np.percentile(att_resized, 75)
    mask = (att_resized >= threshold).astype(float)
    # contour 좌표 맞추기
    y_coords = np.linspace(0, N_MELS, att_resized.shape[0])
    x_coords = np.linspace(0, WINDOW_DURATION, att_resized.shape[1])
    X, Y = np.meshgrid(x_coords, y_coords)
    ax3.contour(X, Y, att_resized, levels=[threshold], colors=['#ffffff'],
                linewidths=2.5, linestyles='solid')
    top_level = max(att_resized.max(), threshold + 1e-6)
    ax3.contourf(X, Y, att_resized, levels=[threshold, top_level],
                 colors=['#58a6ff'], alpha=0.25)

    ax3.set_xlabel('시간 (초)', fontsize=11, color='white')
    ax3.set_ylabel('Mel 주파수', fontsize=11, color='white')
    ax3.tick_params(colors='white')
    for spine in ax3.spines.values():
        spine.set_color('#444')

    if col == 0:
        row3_desc = "흰 윤곽선 = AI가 집중한 곳\n→ 심장 박동 위치에 집중"
    else:
        row3_desc = "흰 윤곽선 = AI가 집중한 곳\n→ 더 넓은 영역을 주시"
    ax3.text(0.02, 0.95, row3_desc, transform=ax3.transAxes,
             fontsize=10, color='white', ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#58a6ff', alpha=0.3))

fig.suptitle('CBAM Attention Map  —  AI가 심음의 어디를 집중해서 보는가',
             fontsize=18, fontweight='bold', color='white', y=0.98)
fig.text(0.5, 0.95,
         '1행: 원본 데이터  |  2행: AI의 집중도 히트맵  |  3행: "AI가 여기를 봤다" (흰색 윤곽선)',
         fontsize=12, color='#aaa', ha='center')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(r"G:\stetho_ai\cbam_attention_map.png", dpi=300,
            bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved: cbam_attention_map.png")
plt.close()


# =====================================================
# 시각화 2: Feature Map 단계별 파이프라인 (색약 친화)
# =====================================================
print("시각화 2: Feature Map 파이프라인 생성 중...")

# 색약 친화 컬러맵: viridis/cividis/plasma (파랑-노랑 계열, 빨강-초록 회피)
# 배경색도 대비 높게

with torch.no_grad():
    model(tensor_abnormal)

fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor('#0d1117')
gs = GridSpec(3, 5, figure=fig, hspace=0.5, wspace=0.35,
              height_ratios=[1.2, 0.5, 1])

# Conv1 출력
x0 = F.relu(model.bn1(model.conv1(tensor_abnormal)))
conv1_feat = x0.detach()[0].mean(dim=0).numpy()

# Layer1 before/after CBAM
_ = model.layer1(x0)
before_cbam = model.layer1.before_cbam[0].mean(dim=0).numpy()
after_cbam = model.layer1.after_cbam[0].mean(dim=0).numpy()
spatial_att_l1 = model.layer1.cbam.spatial_att[0, 0].numpy()

data_list = [mel_abnormal, conv1_feat, before_cbam, after_cbam, spatial_att_l1]

# 색약 친화 컬러맵 (모두 파랑-노랑 or 흑백 계열)
cmaps = ['cividis', 'viridis', 'viridis', 'plasma', 'inferno']

titles = [
    "STEP 1\n입력 Mel-Spectrogram",
    "STEP 2\nConv1 특징 추출",
    "STEP 3\nResNet Block\n(CBAM 적용 전)",
    "STEP 4\nCBAM 적용 후\n(중요 영역 강조)",
    "STEP 5\nSpatial Attention\n(AI의 시선)",
]

descriptions = [
    "청진기 소리를\n2D 이미지로 변환",
    "64개 필터가\n기본 패턴을 감지",
    "잔차(Residual) 연결로\n깊은 특징 학습",
    "CBAM이 중요한\n영역만 강조",
    "밝은 영역 =\nAI가 주목하는 곳",
]

# 설명 박스 색상 (파랑-보라-노랑 계열, 빨강-초록 없음)
box_colors = ['#6366f1', '#0ea5e9', '#8b5cf6', '#f59e0b', '#ec4899']

# ── 1행: 5단계 feature map ──
for i in range(5):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor('#161b22')
    im = ax.imshow(data_list[i], aspect='auto', origin='lower', cmap=cmaps[i])
    ax.set_title(titles[i], fontsize=13, fontweight='bold', color='white', pad=10)
    ax.tick_params(colors='#aaa', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#555')
    # 컬러바
    cb = fig.colorbar(im, ax=ax, pad=0.03, shrink=0.85)
    cb.ax.tick_params(colors='#aaa', labelsize=7)
    # 스텝 번호 크게 표시 (좌상단)
    ax.text(0.05, 0.92, str(i + 1), transform=ax.transAxes,
            fontsize=28, fontweight='bold', color='white', ha='left', va='top',
            bbox=dict(boxstyle='circle,pad=0.3', facecolor=box_colors[i], alpha=0.85))


# ── 2행: 설명 박스 ──
for i in range(5):
    ax = fig.add_subplot(gs[1, i])
    ax.set_facecolor('#0d1117')
    ax.axis('off')
    ax.text(0.5, 0.5, descriptions[i], transform=ax.transAxes,
            fontsize=13, color='white', ha='center', va='center', linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.8', facecolor=box_colors[i], alpha=0.3,
                      edgecolor=box_colors[i], linewidth=2))

# ── 3행: CBAM 전후 비교 (좌: 전, 우: 후, 가운데: 차이) ──
ax_before = fig.add_subplot(gs[2, 0:2])
ax_before.set_facecolor('#161b22')
im_b = ax_before.imshow(before_cbam, aspect='auto', origin='lower', cmap='viridis')
ax_before.set_title('CBAM 적용 전 (STEP 3)', fontsize=14, fontweight='bold',
                     color='#0ea5e9', pad=10)
ax_before.tick_params(colors='#aaa')
for spine in ax_before.spines.values():
    spine.set_color('#555')
cb = fig.colorbar(im_b, ax=ax_before, pad=0.02, shrink=0.85)
cb.ax.tick_params(colors='#aaa')

ax_after = fig.add_subplot(gs[2, 3:5])
ax_after.set_facecolor('#161b22')
im_a = ax_after.imshow(after_cbam, aspect='auto', origin='lower', cmap='plasma')
ax_after.set_title('CBAM 적용 후 (STEP 4)', fontsize=14, fontweight='bold',
                    color='#f59e0b', pad=10)
ax_after.tick_params(colors='#aaa')
for spine in ax_after.spines.values():
    spine.set_color('#555')
cb = fig.colorbar(im_a, ax=ax_after, pad=0.02, shrink=0.85)
cb.ax.tick_params(colors='#aaa')

# 가운데: 차이맵
ax_diff = fig.add_subplot(gs[2, 2])
ax_diff.set_facecolor('#161b22')
diff = after_cbam - before_cbam
# 색약 친화: coolwarm 대신 PuOr (보라-주황)
im_d = ax_diff.imshow(diff, aspect='auto', origin='lower', cmap='PuOr',
                       vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
ax_diff.set_title('변화량\n(밝음=강조 / 어두움=억제)', fontsize=12, fontweight='bold',
                   color='white', pad=10)
ax_diff.tick_params(colors='#aaa')
for spine in ax_diff.spines.values():
    spine.set_color('#555')
cb = fig.colorbar(im_d, ax=ax_diff, pad=0.03, shrink=0.85)
cb.ax.tick_params(colors='#aaa')

# 전→후 화살표

fig.suptitle('ResNet + CBAM 파이프라인  —  AI가 심음을 분석하는 5단계',
             fontsize=20, fontweight='bold', color='white', y=0.98)
fig.text(0.5, 0.95,
         '비정상 심음이 입력됐을 때, 각 단계에서 데이터가 어떻게 변환되는지 보여줍니다',
         fontsize=13, color='#8b949e', ha='center')

plt.savefig(r"G:\stetho_ai\resnet_cbam_pipeline.png", dpi=300,
            bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved: resnet_cbam_pipeline.png")
plt.close()

print("\n완료!")
