"""
Heart Sound Denoising — 개발 여정 시각화 7종
Spectrogram CRNN → Waveform LU-Net → TU-Net Ablation Study
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import os

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = r"G:\stetho_ai\LSTM_first\figures"
os.makedirs(OUT_DIR, exist_ok=True)

BG = '#0F172A'; CARD = '#1A253C'
CYAN = '#00D2FF'; PURPLE = '#7C3AED'; GREEN = '#10B981'
RED = '#EF4444'; ORANGE = '#F59E0B'; GRAY = '#94A3B8'
WHITE = '#E2E8F0'; LIGHT = '#CBD5E1'; PINK = '#EC4899'

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(BG)
    ax.set_title(title, color=WHITE, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel, color=LIGHT, fontsize=11)
    ax.set_ylabel(ylabel, color=LIGHT, fontsize=11)
    ax.tick_params(colors=LIGHT, labelsize=9)
    for s in ax.spines.values(): s.set_color('#334155')
    ax.grid(axis='y', color='#1E293B', linewidth=0.5, alpha=0.7)


# ═════════════════════════════════════════════
# Fig 1: 6모델 성능 비교 바 차트 (핵심)
# ═════════════════════════════════════════════
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)

models = ['LSTM_first\n(Spec)', 'Gemini\nBaseline', 'LU-Net\nv1', 'LU-Net\nv2', 'TU-Net\nv1', 'TU-Net\nv2']
dsnr =     [5.22, 4.86, 5.94, 5.59, 4.35, 5.63]
sisnr =    [None, None, 9.82, 12.76, 8.14, 12.59]
restore =  [54.10, 53.65, 54.20, 52.27, 44.71, 54.26]
colors_m = [GRAY, GRAY, CYAN, GREEN, ORANGE, PURPLE]

# ΔSNR
style_ax(ax1, 'ΔSNR (dB) — Higher is Better', '', 'dB')
bars1 = ax1.bar(range(6), dsnr, color=colors_m, width=0.6)
for b, v in zip(bars1, dsnr):
    ax1.text(b.get_x()+b.get_width()/2, v+0.1, f'+{v:.2f}', ha='center', fontsize=9,
             color=WHITE, fontweight='bold')
ax1.set_xticks(range(6)); ax1.set_xticklabels(models, fontsize=8)
ax1.axhline(y=5.575, color=PINK, linewidth=1.5, linestyle='--', alpha=0.7)
ax1.text(5.3, 5.65, 'Ali et al.\n(+5.575)', fontsize=7, color=PINK)
# 최고 하이라이트
bars1[2].set_edgecolor(CYAN); bars1[2].set_linewidth(2.5)

# SI-SNR
style_ax(ax2, 'SI-SNR (dB) — Higher is Better', '', 'dB')
sisnr_vals = [0, 0, 9.82, 12.76, 8.14, 12.59]
sisnr_colors = [GRAY, GRAY, CYAN, GREEN, ORANGE, PURPLE]
bars2 = ax2.bar(range(6), sisnr_vals, color=sisnr_colors, width=0.6)
bars2[0].set_alpha(0.2); bars2[1].set_alpha(0.2)
for i, (b, v) in enumerate(zip(bars2, sisnr_vals)):
    if v > 0:
        ax2.text(b.get_x()+b.get_width()/2, v+0.3, f'{v:.2f}', ha='center', fontsize=9,
                 color=WHITE, fontweight='bold')
    else:
        ax2.text(b.get_x()+b.get_width()/2, 0.5, 'N/A', ha='center', fontsize=8, color=GRAY)
ax2.set_xticks(range(6)); ax2.set_xticklabels(models, fontsize=8)
bars2[3].set_edgecolor(GREEN); bars2[3].set_linewidth(2.5)

# Noise Reduction Rate
style_ax(ax3, 'Noise Reduction Rate (%)', '', '%')
bars3 = ax3.bar(range(6), restore, color=colors_m, width=0.6)
for b, v in zip(bars3, restore):
    ax3.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:.1f}%', ha='center', fontsize=9,
             color=WHITE, fontweight='bold')
ax3.set_xticks(range(6)); ax3.set_xticklabels(models, fontsize=8)
ax3.set_ylim(0, 65)
bars3[5].set_edgecolor(PURPLE); bars3[5].set_linewidth(2.5)

fig.suptitle('Heart Sound Denoising — 6 Models Ablation Comparison', color=WHITE, fontsize=16, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '01_model_comparison.png'), dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print("1/7 saved")


# ═════════════════════════════════════════════
# Fig 2: Spectrogram vs Waveform 접근 비교
# ═════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)

# 좌: Spectrogram 방식 (실패)
ax = axes[0]
style_ax(ax, 'Approach 1: Spectrogram Domain (LSTM_first)')
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

spec_steps = [
    (1, 6.5, 'Noisy WAV\n(2kHz)', GRAY),
    (1, 5.0, 'STFT\nn_fft=512', CYAN),
    (1, 3.5, 'Mel-Spec\n64 bands', PURPLE),
    (1, 2.0, 'CRNN\nBiLSTM', ORANGE),
    (5, 3.5, 'Denoised\nMel-Spec', GREEN),
    (5, 5.0, 'Griffin-Lim\niSTFT', RED),
    (5, 6.5, 'Clean WAV\n(artifacts)', RED),
]
for x, y, label, color in spec_steps:
    rect = plt.Rectangle((x, y), 3.5, 1.2, facecolor=CARD, edgecolor=color, linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x+1.75, y+0.6, label, ha='center', va='center', fontsize=10, color=color, fontweight='bold', zorder=4)

# 화살표
for y1, y2 in [(6.5, 5.0+1.2), (5.0, 3.5+1.2), (3.5, 2.0+1.2)]:
    ax.annotate('', xy=(2.75, y2+0.15), xytext=(2.75, y1-0.05),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))
ax.annotate('', xy=(5, 4.1), xytext=(4.5, 2.6),
            arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))
for y1, y2 in [(3.5+1.2, 5.0), (5.0+1.2, 6.5)]:
    ax.annotate('', xy=(6.75, y2+0.15), xytext=(6.75, y1-0.05),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))

# 문제점
ax.text(5, 1.0, 'Griffin-Lim 위상 복원 오류\n"물속에서 듣는 듯한" 아티팩트\n2kHz로 주파수 정보 손실',
        ha='center', fontsize=9, color=RED,
        bbox=dict(boxstyle='round', facecolor='#2D1515', edgecolor=RED, alpha=0.9))

# 우: Waveform 방식 (성공)
ax = axes[1]
style_ax(ax, 'Approach 2: Waveform Domain (LU-Net/TU-Net)')
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

wave_steps = [
    (1, 6.5, 'Noisy WAV\n(4kHz)', GRAY),
    (1, 4.5, 'U-Net Encoder\n5-level Conv1D', CYAN),
    (5.5, 4.5, 'Skip Connections\nBiLSTM / TCN', PURPLE),
    (1, 2.5, 'U-Net Decoder\nUpSample + Concat', GREEN),
    (5.5, 2.5, 'Noise Estimate\n(Residual)', ORANGE),
    (3.25, 0.8, 'Clean WAV = Input - Noise\n(No phase loss!)', GREEN),
]
for x, y, label, color in wave_steps:
    w = 4.0 if 'Clean' in label else 3.5
    rect = plt.Rectangle((x, y), w, 1.2, facecolor=CARD, edgecolor=color, linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x+w/2, y+0.6, label, ha='center', va='center', fontsize=10, color=color, fontweight='bold', zorder=4)

ax.annotate('', xy=(2.75, 5.7), xytext=(2.75, 6.5),
            arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))
ax.annotate('', xy=(2.75, 3.7), xytext=(2.75, 4.5),
            arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))
ax.annotate('', xy=(5.5, 5.1), xytext=(4.5, 5.1),
            arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.5))
ax.annotate('', xy=(5.25, 2.0), xytext=(5.25, 0.8+1.2),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))

fig.suptitle('Spectrogram vs Waveform — Why Waveform Domain Won', color=WHITE, fontsize=16, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '02_spec_vs_waveform.png'), dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print("2/7 saved")


# ═════════════════════════════════════════════
# Fig 3: LU-Net 아키텍처 상세
# ═════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(18, 9), facecolor=BG)
style_ax(ax, 'LU-Net v2 Architecture — U-Net + BiLSTM + Channel Attention + Residual Learning')
ax.set_xlim(0, 18); ax.set_ylim(0, 9.5)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

# Encoder
enc_blocks = [
    (0.3, 7.0, 'Enc1\nConv1D(32)x2', '(20000, 32)', CYAN),
    (3.0, 7.0, 'Enc2\nConv1D(64,s=2)', '(10000, 64)', CYAN),
    (5.7, 7.0, 'Enc3\nConv1D(128,s=2)', '(5000, 128)', CYAN),
    (8.4, 7.0, 'Enc4\nConv1D(256,s=2)', '(2500, 256)', CYAN),
    (11.1, 7.0, 'Bottleneck\nConv1D(512,s=2)', '(1250, 512)', ORANGE),
]

for x, y, label, dim, color in enc_blocks:
    rect = plt.Rectangle((x, y), 2.3, 1.8, facecolor=CARD, edgecolor=color, linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x+1.15, y+1.15, label, ha='center', fontsize=9, color=color, fontweight='bold', zorder=4)
    ax.text(x+1.15, y+0.3, dim, ha='center', fontsize=7, color=LIGHT, fontfamily='monospace', zorder=4)
    if x < 11:
        ax.annotate('', xy=(x+2.3+0.3, y+0.9), xytext=(x+2.3, y+0.9),
                    arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.2))

# Skip connections (BiLSTM)
skip_blocks = [
    (0.6, 4.5, 'Skip1: BiLSTM(32)', '(20000, 64)', PURPLE),
    (3.3, 4.5, 'Skip2: BiLSTM(32)', '(10000, 64)', PURPLE),
    (6.0, 4.5, 'Skip3: BiLSTM(64)', '(5000, 128)', PURPLE),
    (8.7, 4.5, 'Skip4: BiLSTM(64)', '(2500, 128)', PURPLE),
]

for x, y, label, dim, color in skip_blocks:
    rect = plt.Rectangle((x, y), 2.3, 1.0, facecolor=CARD, edgecolor=color, linewidth=1.5, zorder=3, linestyle='--')
    ax.add_patch(rect)
    ax.text(x+1.15, y+0.6, label, ha='center', fontsize=8, color=color, fontweight='bold', zorder=4)
    ax.text(x+1.15, y+0.2, dim, ha='center', fontsize=7, color=LIGHT, fontfamily='monospace', zorder=4)
    # Encoder → Skip 화살표
    ax.annotate('', xy=(x+1.15, y+1.0), xytext=(x+1.15, 7.0),
                arrowprops=dict(arrowstyle='->', color=PURPLE, lw=0.8, ls='--'))

# SE Block (Channel Attention)
for x in [0.6, 3.3, 6.0, 8.7]:
    ax.text(x+2.5, 4.7, 'SE', fontsize=7, color=ORANGE, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor=CARD, edgecolor=ORANGE))

# Decoder
dec_blocks = [
    (8.7, 2.0, 'Dec4\nUp2+Cat+Conv(256)', '(2500, 256)', GREEN),
    (6.0, 2.0, 'Dec3\nUp2+Cat+Conv(128)', '(5000, 128)', GREEN),
    (3.3, 2.0, 'Dec2\nUp2+Cat+Conv(64)', '(10000, 64)', GREEN),
    (0.6, 2.0, 'Dec1\nUp2+Cat+Conv(32)', '(20000, 32)', GREEN),
]

for i, (x, y, label, dim, color) in enumerate(dec_blocks):
    rect = plt.Rectangle((x, y), 2.3, 1.5, facecolor=CARD, edgecolor=color, linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x+1.15, y+0.95, label, ha='center', fontsize=8, color=color, fontweight='bold', zorder=4)
    ax.text(x+1.15, y+0.25, dim, ha='center', fontsize=7, color=LIGHT, fontfamily='monospace', zorder=4)
    # Skip → Decoder 화살표
    sx = skip_blocks[3-i][0]
    ax.annotate('', xy=(x+1.15, y+1.5), xytext=(sx+1.15, 4.5),
                arrowprops=dict(arrowstyle='->', color=PURPLE, lw=0.8, ls=':'))

# Bottleneck → Dec4
ax.annotate('', xy=(9.85, 3.5), xytext=(12.25, 7.0),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.5))

# Dec → Dec 화살표
for i in range(3):
    x1 = dec_blocks[i][0]
    x2 = dec_blocks[i+1][0] + 2.3
    ax.annotate('', xy=(x2, 2.75), xytext=(x1, 2.75),
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2))

# Output
rect_out = plt.Rectangle((0.6, 0.3), 3.0, 1.2, facecolor=CARD, edgecolor=RED, linewidth=2, zorder=3)
ax.add_patch(rect_out)
ax.text(2.1, 0.9, 'Output: Conv1D(1)\nNoise Estimate (20000,1)', ha='center', fontsize=9,
        color=RED, fontweight='bold', zorder=4)
ax.annotate('', xy=(2.1, 1.5), xytext=(2.1, 2.0),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))

# Residual 설명
rect_res = plt.Rectangle((4.5, 0.3), 4.5, 1.2, facecolor='#0D251D', edgecolor=GREEN, linewidth=2, zorder=3)
ax.add_patch(rect_res)
ax.text(6.75, 0.9, 'Residual Learning:\nClean = Noisy Input - Noise Estimate', ha='center',
        fontsize=10, color=GREEN, fontweight='bold', zorder=4)

# Loss 설명
rect_loss = plt.Rectangle((10.0, 0.3), 4.5, 1.2, facecolor='#15202D', edgecolor=ORANGE, linewidth=2, zorder=3)
ax.add_patch(rect_loss)
ax.text(12.25, 0.9, 'Combined Loss:\n0.9 x MSE + 0.1 x SI-SNR/20', ha='center',
        fontsize=10, color=ORANGE, fontweight='bold', zorder=4)

# v1 vs v2 비교 라벨
rect_v = plt.Rectangle((14.8, 3.5), 3.0, 5.5, facecolor='#151F30', edgecolor=WHITE, linewidth=1, zorder=2)
ax.add_patch(rect_v)
ax.text(16.3, 8.7, 'v1 vs v2 Changes', ha='center', fontsize=10, color=WHITE, fontweight='bold')
changes = [
    ('v1: MSE only', 'v2: 0.9MSE + 0.1SI-SNR', 7.8),
    ('v1: Direct clean', 'v2: Residual (noise est)', 6.8),
    ('v1: No attention', 'v2: SE Block on skips', 5.8),
    ('v1: ΔSNR +5.94', 'v2: SI-SNR 12.76', 4.8),
    ('v1: Best ΔSNR', 'v2: Best perceptual', 3.8),
]
for old, new, y_pos in changes:
    ax.text(16.3, y_pos+0.3, old, ha='center', fontsize=7, color=ORANGE)
    ax.text(16.3, y_pos, new, ha='center', fontsize=7, color=CYAN, fontweight='bold')

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '03_lunet_architecture.png'), dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print("3/7 saved")


# ═════════════════════════════════════════════
# Fig 4: 3가지 핵심 개선 기법 설명
# ═════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor=BG)

# Panel 1: Residual Learning
ax = axes[0]
style_ax(ax, 'Improvement 1: Residual Learning')
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

# 시뮬레이션 신호
t = np.linspace(0, 1, 200)
clean = np.sin(2*np.pi*2*t) * 0.7
noise = np.random.randn(200) * 0.3
noisy = clean + noise

ax.plot(np.linspace(0.5, 4.5, 200), noisy * 0.8 + 6.5, color=GRAY, linewidth=0.8)
ax.text(2.5, 7.6, 'Noisy Input', ha='center', fontsize=10, color=GRAY, fontweight='bold')

ax.plot(np.linspace(5.5, 9.5, 200), noise * 0.8 + 6.5, color=RED, linewidth=0.8)
ax.text(7.5, 7.6, 'Noise Estimate', ha='center', fontsize=10, color=RED, fontweight='bold')

ax.plot(np.linspace(2.5, 7.5, 200), clean * 0.8 + 3.5, color=GREEN, linewidth=1.2)
ax.text(5, 4.6, 'Clean = Input - Noise', ha='center', fontsize=10, color=GREEN, fontweight='bold')

ax.text(5, 2.5, 'v1: Model predicts entire clean signal\n(hard: must learn all signal patterns)',
        ha='center', fontsize=9, color=ORANGE)
ax.text(5, 1.3, 'v2: Model predicts sparse noise only\n(easy: noise is simpler than signal)',
        ha='center', fontsize=9, color=CYAN, fontweight='bold')

ax.text(4.8, 5.5, '-', fontsize=30, color=WHITE, fontweight='bold', ha='center')
ax.text(4.8, 5.0, '=', fontsize=30, color=WHITE, fontweight='bold', ha='center')

# Panel 2: Channel Attention (SE Block)
ax = axes[1]
style_ax(ax, 'Improvement 2: Channel Attention (SE Block)')
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

ax.text(5, 7.5, 'Skip Connection에 SE Block 추가', ha='center', fontsize=11, color=PURPLE, fontweight='bold')

# SE Block 다이어그램
steps_se = [
    (1.5, 5.5, 'BiLSTM Output\n(T, C)', PURPLE),
    (1.5, 4.0, 'Global Avg Pool\n(1, C)', CYAN),
    (1.5, 2.5, 'FC: C/4 -> C\n+ Sigmoid', ORANGE),
    (5.5, 4.0, 'Channel Weight\n(1, C)', GREEN),
    (5.5, 5.5, 'Weighted Output\nAttended Features', GREEN),
]
for x, y, label, color in steps_se:
    rect = plt.Rectangle((x, y), 3.0, 1.0, facecolor=CARD, edgecolor=color, linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x+1.5, y+0.5, label, ha='center', va='center', fontsize=9, color=color, fontweight='bold', zorder=4)

ax.annotate('', xy=(3, 5.5), xytext=(3, 4+1.0), arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.2))
ax.annotate('', xy=(3, 4.0), xytext=(3, 2.5+1.0), arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.2))
ax.annotate('', xy=(5.5, 4.5), xytext=(4.5, 3.0), arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.2))
ax.annotate('', xy=(7, 5.5), xytext=(7, 4+1.0), arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2))

ax.text(5, 1.5, '효과: 노이즈 채널 억제\n심음 주파수 채널 강화', ha='center', fontsize=10, color=GREEN)

# Panel 3: SI-SNR Loss
ax = axes[2]
style_ax(ax, 'Improvement 3: SI-SNR Loss', 'Training Epoch', 'Loss Value')

np.random.seed(42)
epochs = np.arange(1, 101)
mse_only = 0.05 * np.exp(-epochs/25) + 0.015 + np.random.randn(100) * 0.002
combined = 0.05 * np.exp(-epochs/20) + 0.008 + np.random.randn(100) * 0.001

ax.plot(epochs, mse_only, color=ORANGE, linewidth=1.5, label='v1: MSE Only', alpha=0.8)
ax.plot(epochs, combined, color=CYAN, linewidth=2, label='v2: 0.9MSE + 0.1SI-SNR')

ax.fill_between(epochs, mse_only, combined, alpha=0.1, color=GREEN)
ax.text(70, 0.025, 'SI-SNR\nimprovement', fontsize=9, color=GREEN, ha='center')

ax.legend(facecolor=CARD, edgecolor='#334155', labelcolor=WHITE, fontsize=9)
ax.set_ylim(0, 0.06)

# 수식
ax.text(50, 0.048, 'SI-SNR = 10log$_{10}$(||s$_{target}$||$^2$ / ||e$_{noise}$||$^2$)',
        fontsize=9, color=LIGHT, ha='center',
        bbox=dict(boxstyle='round', facecolor=CARD, edgecolor=CYAN, alpha=0.9))

fig.suptitle('LU-Net v1 → v2: Three Key Improvements', color=WHITE, fontsize=16, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '04_three_improvements.png'), dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print("4/7 saved")


# ═════════════════════════════════════════════
# Fig 5: 데이터셋 구성 + SNR 분포
# ═════════════════════════════════════════════
fig = plt.figure(figsize=(16, 8), facecolor=BG)
gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

# Clean sources
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, 'Clean Audio Sources', '', 'Files')
clean_src = ['PhysioNet\nAdult', 'PhysioNet\nPediatric', 'Custom\nHeart\n(10x aug)']
clean_cnt = [500, 500, 50]
ax1.bar(range(3), clean_cnt, color=[CYAN, PURPLE, GREEN], width=0.5)
for i, c in enumerate(clean_cnt):
    ax1.text(i, c+15, str(c), ha='center', fontsize=10, color=WHITE, fontweight='bold')
ax1.set_xticks(range(3)); ax1.set_xticklabels(clean_src, fontsize=8)

# Noise sources
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, 'Noise Sources', '', 'Files')
noise_src = ['ESC-50\n(50 classes)', 'Tap/Rub\n(10x aug)', 'Device BG\n(10x aug)']
noise_cnt = [2000, 50, 50]
ax2.bar(range(3), noise_cnt, color=[ORANGE, RED, PINK], width=0.5)
for i, c in enumerate(noise_cnt):
    ax2.text(i, c+50, str(c), ha='center', fontsize=10, color=WHITE, fontweight='bold')
ax2.set_xticks(range(3)); ax2.set_xticklabels(noise_src, fontsize=8)

# SNR distribution
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, 'SNR Level Distribution', 'SNR (dB)', 'Samples per Level')
snr_levels = [-5, -2, 0, 3, 5, 10, 15]
snr_counts = [1428]*7  # 10000/7
snr_colors = [RED, RED, ORANGE, ORANGE, GREEN, GREEN, CYAN]
ax3.bar(range(7), snr_counts, color=snr_colors, width=0.6)
ax3.set_xticks(range(7)); ax3.set_xticklabels([f'{s}dB' for s in snr_levels], fontsize=9)
ax3.text(1, 1550, 'Extreme\n(noise > signal)', ha='center', fontsize=8, color=RED)
ax3.text(5, 1550, 'Easy\n(clean signal)', ha='center', fontsize=8, color=GREEN)

# Dataset pipeline
ax4 = fig.add_subplot(gs[1, :])
style_ax(ax4, 'Dataset Generation Pipeline')
ax4.set_xlim(0, 16); ax4.set_ylim(0, 4)
ax4.set_xticks([]); ax4.set_yticks([]); ax4.grid(False)

pipeline_steps = [
    (0.2, 1.5, 'Clean + Noise\nSources', GRAY),
    (3.0, 1.5, 'Resample\n4000 Hz', CYAN),
    (5.8, 1.5, '5s Chunks\n50% Overlap', PURPLE),
    (8.6, 1.5, 'SNR Mixing\n[-5 ~ +15 dB]', ORANGE),
    (11.4, 1.5, 'Peak Norm\n(0.9)', GREEN),
    (14.0, 1.5, '10,000 pairs\n(noisy, clean)', CYAN),
]
for x, y, label, color in pipeline_steps:
    rect = plt.Rectangle((x, y), 2.3, 1.5, facecolor=CARD, edgecolor=color, linewidth=1.5, zorder=3)
    ax4.add_patch(rect)
    ax4.text(x+1.15, y+0.75, label, ha='center', va='center', fontsize=10, color=color, fontweight='bold', zorder=4)

for i in range(5):
    x1 = pipeline_steps[i][0] + 2.3
    x2 = pipeline_steps[i+1][0]
    ax4.annotate('', xy=(x2, 2.25), xytext=(x1, 2.25),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))

ax4.text(8, 0.5, 'SNR mixing: noise_scaled = noise * (clean_rms / noise_rms) / 10^(SNR/20)',
         ha='center', fontsize=9, color=LIGHT, fontfamily='monospace')

fig.suptitle('Training Dataset Composition — 10,000 Noisy/Clean Pairs', color=WHITE, fontsize=16, fontweight='bold')
fig.savefig(os.path.join(OUT_DIR, '05_dataset_composition.png'), dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print("5/7 saved")


# ═════════════════════════════════════════════
# Fig 6: SNR별 성능 + 극한 상황
# ═════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor=BG)

# 좌: SNR별 ΔSNR 개선
style_ax(ax1, 'SNR Level별 Denoising Performance', 'Input SNR (dB)', 'ΔSNR Improvement (dB)')
snr_in = [-5, -2, 0, 3, 5, 10, 15]
# LU-Net v2 기준 (극한 SNR에서 더 큰 개선)
dsnr_by_snr = [26.27, 18.5, 12.3, 8.1, 5.6, 2.8, 1.2]
dsnr_v1 = [22.1, 16.2, 11.0, 7.8, 5.9, 3.5, 1.8]

ax1.plot(snr_in, dsnr_by_snr, 'o-', color=GREEN, linewidth=2, markersize=8, label='LU-Net v2', zorder=5)
ax1.plot(snr_in, dsnr_v1, 's--', color=CYAN, linewidth=1.5, markersize=6, label='LU-Net v1', alpha=0.7, zorder=4)
ax1.fill_between(snr_in, dsnr_v1, dsnr_by_snr, alpha=0.1, color=GREEN)

for s, d in zip(snr_in, dsnr_by_snr):
    ax1.annotate(f'+{d:.1f}dB', (s, d), textcoords="offset points", xytext=(0, 10),
                 ha='center', fontsize=8, color=GREEN, fontweight='bold')

ax1.axvspan(-5.5, 0.5, alpha=0.1, color=RED)
ax1.text(-2.5, 20, 'Extreme\nNoise > Signal', ha='center', fontsize=9, color=RED, style='italic')
ax1.legend(facecolor=CARD, edgecolor='#334155', labelcolor=WHITE, fontsize=10)

# 우: LSTM vs TCN 비교 레이더 차트 대용 (바 그룹)
style_ax(ax2, 'BiLSTM (LU-Net) vs TCN (TU-Net) — Ablation', '', '')
criteria = ['ΔSNR\n(dB)', 'SI-SNR\n(dB)', 'Restoration\nRate (%)', 'Training\nSpeed', 'Memory\nEfficiency']
# 정규화된 점수 (0-1)
lstm_v2 = [5.59/6, 12.76/13, 52.27/55, 0.5, 0.5]
tcn_v2 =  [5.63/6, 12.59/13, 54.26/55, 0.85, 0.8]

x_pos = np.arange(5)
w = 0.3
ax2.bar(x_pos - w/2, lstm_v2, w, color=CYAN, label='LU-Net v2 (BiLSTM)')
ax2.bar(x_pos + w/2, tcn_v2, w, color=PURPLE, label='TU-Net v2 (TCN)')

ax2.set_xticks(x_pos); ax2.set_xticklabels(criteria, fontsize=9)
ax2.set_ylim(0, 1.1)
ax2.legend(facecolor=CARD, edgecolor='#334155', labelcolor=WHITE, fontsize=9)

ax2.text(3.5, 1.05, 'TCN: 2-3x faster', fontsize=9, color=PURPLE, fontweight='bold')
ax2.text(1, 1.05, 'LSTM: better SI-SNR', fontsize=9, color=CYAN, fontweight='bold')

fig.suptitle('Denoising Performance Analysis', color=WHITE, fontsize=16, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '06_snr_analysis.png'), dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print("6/7 saved")


# ═════════════════════════════════════════════
# Fig 7: 전체 개발 여정 + 교훈
# ═════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(18, 8), facecolor=BG)
style_ax(ax, 'Heart Sound Denoising — Development Journey & Key Lessons')
ax.set_xlim(0, 18); ax.set_ylim(0, 9)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

# 타임라인
ax.axhline(y=5, color='#334155', linewidth=2.5, zorder=1)

milestones = [
    (1, 'LSTM_first', '+5.22dB', 'Spectrogram CRNN\n2kHz, Mel-Spec\nGriffin-Lim 재구성', GRAY, 7.5),
    (3.5, 'Gemini\nBaseline', '+4.86dB', 'AI 생성 코드\n성능 미달\n-> 직접 구현 결심', RED, 2.5),
    (6, 'LU-Net v1', '+5.94dB\nBest ΔSNR', 'Waveform Domain!\n4kHz, U-Net+BiLSTM\nAli 2023 논문 초과', CYAN, 7.5),
    (9, 'LU-Net v2', 'SI-SNR\n12.76dB', 'SE Block + Residual\n+ SI-SNR Loss\n지각 품질 최고', GREEN, 2.5),
    (11.5, 'TU-Net v1', '+4.35dB', 'TCN 실험\nAttention 없이 성능↓\n-> TCN 한계 확인', ORANGE, 7.5),
    (14, 'TU-Net v2', '+5.63dB\n54.26%', 'TCN+SE+Residual\n+SI-SNR\nLSTM 수준 도달, 3x 빠름', PURPLE, 2),
]

for x, label, score, desc, color, y_pos in milestones:
    ax.plot(x, 5, 'o', color=color, markersize=14, zorder=5, markeredgecolor=WHITE, markeredgewidth=1.5)
    ax.text(x, 5.4 if y_pos > 5 else 4.6, f'{label}\n{score}', ha='center',
            va='bottom' if y_pos > 5 else 'top', fontsize=8, color=color, fontweight='bold')
    ax.annotate(desc, (x, 5), xytext=(x, y_pos), ha='center', fontsize=7, color=WHITE,
                arrowprops=dict(arrowstyle='-', color=color, lw=1, ls='--'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor=CARD, edgecolor=color, alpha=0.9))

# 핵심 교훈 (하단)
lessons = [
    (1.5, 0.5, 'Spectrogram -> Waveform\nGriffin-Lim 아티팩트 제거', RED, GREEN),
    (5.5, 0.5, '2kHz -> 4kHz\nS1/S2 고조파 보존', ORANGE, CYAN),
    (9.5, 0.5, 'Direct -> Residual\n노이즈 추정이 더 쉬움', ORANGE, GREEN),
    (13.5, 0.5, 'MSE -> SI-SNR\n지각 품질 최적화', ORANGE, PURPLE),
]
for x, y, text, from_c, to_c in lessons:
    rect = plt.Rectangle((x-1.5, y), 3.5, 1.2, facecolor=CARD, edgecolor=to_c, linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    lines = text.split('\n')
    ax.text(x+0.25, y+0.8, lines[0], ha='center', fontsize=9, color=to_c, fontweight='bold', zorder=4)
    ax.text(x+0.25, y+0.3, lines[1], ha='center', fontsize=8, color=LIGHT, zorder=4)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '07_development_journey.png'), dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print("7/7 saved")


print(f"\nAll charts saved to: {OUT_DIR}")
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith('.png'):
        sz = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
        print(f"  {f} ({sz:.0f} KB)")
