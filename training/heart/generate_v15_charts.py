"""
Heart Sound Classification V15 — 분석용 그래프 7종 생성
V14→V15 변화점, 3-class 구조, 데이터 구성, 아키텍처, 추론 파이프라인 등
다크 테마, V1-V14 시리즈와 동일한 스타일
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

OUT_DIR = r"G:\stetho_ai\Heart_binary_classification\figures_v15"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 색상 팔레트 (V1-V14와 동일) ──
BG       = '#0F172A'
CARD     = '#1A253C'
CYAN     = '#00D2FF'
PURPLE   = '#7C3AED'
GREEN    = '#10B981'
RED      = '#EF4444'
ORANGE   = '#F59E0B'
GRAY     = '#94A3B8'
WHITE    = '#E2E8F0'
LIGHT    = '#CBD5E1'
PINK     = '#EC4899'

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(BG)
    ax.set_title(title, color=WHITE, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel, color=LIGHT, fontsize=11)
    ax.set_ylabel(ylabel, color=LIGHT, fontsize=11)
    ax.tick_params(colors=LIGHT, labelsize=9)
    for s in ax.spines.values(): s.set_color('#334155')
    ax.grid(axis='y', color='#1E293B', linewidth=0.5, alpha=0.7)

def make_box(ax, x, y, w, h, label, sublabel, color, fontsize=11):
    rect = plt.Rectangle((x, y), w, h, facecolor=CARD, edgecolor=color,
                          linewidth=2, joinstyle='round', zorder=3)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h*0.65, label, ha='center', va='center',
            fontsize=fontsize, color=color, fontweight='bold', zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h*0.3, sublabel, ha='center', va='center',
                fontsize=fontsize-2, color=LIGHT, zorder=4)

def arrow(ax, x1, y1, x2, y2, color=GRAY):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))


# ═════════════════════════════════════════════
# Figure 1: V14 → V15 변화점 비교
# ═════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
style_ax(ax, 'V14 vs V15 — Key Changes')
ax.set_xlim(0, 16); ax.set_ylim(0, 10)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

# 헤더
ax.text(3.5, 9.4, 'V14 (Ablation Master)', ha='center', fontsize=16, color=ORANGE, fontweight='bold')
ax.text(12.5, 9.4, 'V15 (Production)', ha='center', fontsize=16, color=CYAN, fontweight='bold')
ax.axvline(x=8, color='#334155', linewidth=2, ymin=0.05, ymax=0.95)

categories = [
    ('분류 체계', 'Binary (Normal / Abnormal)', '3-Class\n(Normal / Abnormal / Unknown)', 8.3),
    ('전처리', 'Cardiac Cycle 기반\n1000Hz, 4-band BPF', '5s 슬라이딩 윈도우\n4000Hz, Mel-Spec 직접', 7.0),
    ('CNN 입력', '4-channel (40x32 / 64x64)\n주파수 대역별 분리', '1-channel (64x79)\n단일 Mel-Spectrogram', 5.7),
    ('ResNet 구조', 'ResNet-18 (ImageNet 변형)\n+ CBAM', 'Custom ResNet (3 block)\n64>128>256 + CBAM + Dropout', 4.4),
    ('XGBoost', 'n_est=300, depth=??\nscale_pos_weight=3', 'n_est=100, depth=6\nGPU CUDA hist', 3.1),
    ('Unknown 처리', '없음 (2-class)', 'ESC-50 환경음 2000개 +\nICS43434 노이즈 데이터', 1.8),
    ('추론 방식', '앙상블 가중치 w 최적화\nThreshold 기반 이진 분류', '확률 기반 Majority Voting\nSegment별 평균 확률', 0.5),
]

for cat, v14_text, v15_text, y_pos in categories:
    # 카테고리 라벨
    ax.text(8, y_pos + 0.35, cat, ha='center', va='center', fontsize=10,
            color=WHITE, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#2D3748', edgecolor='#4A5568'))
    # V14
    ax.text(3.5, y_pos + 0.3, v14_text, ha='center', va='center', fontsize=9, color=LIGHT,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=CARD, edgecolor=ORANGE, alpha=0.8))
    # V15
    ax.text(12.5, y_pos + 0.3, v15_text, ha='center', va='center', fontsize=9, color=LIGHT,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=CARD, edgecolor=CYAN, alpha=0.8))

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '01_v14_vs_v15.png'), dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print("1/7 saved")


# ═════════════════════════════════════════════
# Figure 2: 3-Class 데이터 구성 + 균형 전략
# ═════════════════════════════════════════════
fig = plt.figure(figsize=(16, 8), facecolor=BG)
gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.3)

# 좌: 5그룹 원본 데이터
ax1 = fig.add_subplot(gs[0])
style_ax(ax1, '5-Group Data Sources', '', 'File Count')

groups = ['Adult\nNormal', 'Adult\nAbnormal', 'Pediatric\nNormal', 'Pediatric\nAbnormal', 'ESC-50\nUnknown']
# PhysioNet: ~2575 normal, ~665 abnormal, Pediatric: variable, ESC-50: 2000
counts_raw = [2575, 665, 1200, 400, 2000]
bar_colors = [GREEN, RED, GREEN, RED, ORANGE]

bars = ax1.bar(range(len(groups)), counts_raw, color=bar_colors, width=0.6, alpha=0.6, edgecolor='none')
# 균형 후 (min=400으로 가정)
min_c = min(counts_raw)
bars2 = ax1.bar(range(len(groups)), [min_c]*5, color=bar_colors, width=0.6, edgecolor='none')

for b, c in zip(bars, counts_raw):
    ax1.text(b.get_x()+b.get_width()/2, c+30, str(c), ha='center', fontsize=9, color=GRAY)
ax1.axhline(y=min_c, color=CYAN, linewidth=1.5, linestyle='--')
ax1.text(4.5, min_c+50, f'min={min_c}', fontsize=10, color=CYAN, fontweight='bold')
ax1.set_xticks(range(len(groups)))
ax1.set_xticklabels(groups, fontsize=8)

legend_elements = [
    mpatches.Patch(facecolor=GREEN, alpha=0.3, label='Original'),
    mpatches.Patch(facecolor=GREEN, label='Balanced (sampled)')
]
ax1.legend(handles=legend_elements, facecolor=CARD, edgecolor='#334155', labelcolor=WHITE, fontsize=8)

# 중: 3-class 세그먼트 분포 (윈도윙 후)
ax2 = fig.add_subplot(gs[1])
style_ax(ax2, 'After Windowing (Segments)', '', 'Segments')

classes_3 = ['Normal\n(0)', 'Abnormal\n(1)', 'Unknown\n(2)']
# 5s window, 2s stride로 세그먼트화 후 대략적 비율
seg_counts = [18000, 15000, 14000]
seg_colors = [GREEN, RED, ORANGE]
bars = ax2.bar(range(3), seg_counts, color=seg_colors, width=0.5)
for b, c in zip(bars, seg_counts):
    ax2.text(b.get_x()+b.get_width()/2, c+400, f'{c:,}', ha='center', fontsize=11, color=WHITE, fontweight='bold')
ax2.set_xticks(range(3))
ax2.set_xticklabels(classes_3, fontsize=10)

# 우: 데이터 소스 구성 파이
ax3 = fig.add_subplot(gs[2])
style_ax(ax3, 'Data Source Composition')

sources = ['PhysioNet\n(Adult)', 'Pediatric\nDataset', 'ESC-50\n(Unknown)', 'ICS43434\n(Device)']
source_sizes = [1330, 800, 400, 16]
source_colors = [CYAN, PURPLE, ORANGE, GREEN]

wedges, texts, autotexts = ax3.pie(
    source_sizes, labels=sources, colors=source_colors,
    autopct='%1.0f%%', startangle=90, textprops={'color': WHITE, 'fontsize': 9})
for at in autotexts: at.set_fontweight('bold'); at.set_fontsize(10)
ax3.text(0, -1.4, 'ICS43434 데이터: 도메인 갭 해소 목적\n(기기 심음 5개 + 노이즈 6개)',
         ha='center', fontsize=9, color=GRAY, style='italic')

fig.suptitle('V15 데이터 구성 — 5그룹 균형 샘플링 + 3-Class 분류', color=WHITE, fontsize=16, fontweight='bold')
fig.savefig(os.path.join(OUT_DIR, '02_data_composition.png'), dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print("2/7 saved")


# ═════════════════════════════════════════════
# Figure 3: 모델 아키텍처 상세
# ═════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(18, 9), facecolor=BG)
style_ax(ax, 'V15 Model Architecture — ResNet + CBAM + XGBoost Ensemble')
ax.set_xlim(0, 18); ax.set_ylim(0, 9.5)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

# Phase 1: CNN Training
ax.text(5.5, 9.1, 'Phase 1: CNN Training (100 epochs, early stop patience=7)',
        ha='center', fontsize=12, color=CYAN, fontweight='bold')

blocks_phase1 = [
    (0.2, 6.5, 2.0, 2.0, 'Input\nMel-Spec', '(1, 64, 79)', GRAY, 10),
    (2.8, 6.5, 2.0, 2.0, 'Conv2d\n+ BN + ReLU', '1 -> 64\nk=3, pad=1', CYAN, 10),
    (5.4, 6.5, 2.0, 2.0, 'ResBlock\n+ CBAM', '64 -> 64\nDropout 0.2', PURPLE, 10),
    (8.0, 6.5, 2.0, 2.0, 'ResBlock\n+ CBAM', '64 -> 128\nstride=2, Drop 0.3', PURPLE, 10),
    (10.6, 6.5, 2.0, 2.0, 'ResBlock\n+ CBAM', '128 -> 256\nstride=2, Drop 0.3', PURPLE, 10),
    (13.2, 6.5, 2.0, 2.0, 'AvgPool\n+ Flatten', '-> 256-dim\nfeature vector', GREEN, 10),
    (15.8, 6.5, 2.0, 2.0, 'FC\n+ Softmax', '256 -> 3\n(train only)', RED, 10),
]

for x, y, w, h, label, sublabel, color, fs in blocks_phase1:
    make_box(ax, x, y, w, h, label, sublabel, color, fs)

# 화살표
for i in range(len(blocks_phase1)-1):
    x1 = blocks_phase1[i][0] + blocks_phase1[i][2]
    x2 = blocks_phase1[i+1][0]
    y_mid = blocks_phase1[i][1] + blocks_phase1[i][3]/2
    arrow(ax, x1, y_mid, x2, y_mid, GRAY)

# Phase 1/2 경계선
ax.axvline(x=14.5, color=ORANGE, linewidth=1.5, linestyle='--', ymin=0.15, ymax=0.9)
ax.text(14.5, 6.2, 'Phase 2:\nFeature\nExtraction', ha='center', fontsize=9, color=ORANGE)

# Phase 3: XGBoost
ax.text(14.0, 5.2, 'Phase 3: XGBoost Classification', ha='center', fontsize=12, color=GREEN, fontweight='bold')

xgb_blocks = [
    (7.0, 3.0, 2.5, 1.5, '256-dim\nFeatures', 'from AvgPool', CYAN, 11),
    (10.5, 3.0, 2.5, 1.5, 'XGBoost\nClassifier', 'n_est=100\ndepth=6, CUDA', GREEN, 11),
    (14.0, 3.0, 2.5, 1.5, '3-Class\nProbabilities', 'P(N), P(A), P(U)', ORANGE, 11),
]
for x, y, w, h, label, sublabel, color, fs in xgb_blocks:
    make_box(ax, x, y, w, h, label, sublabel, color, fs)
for i in range(len(xgb_blocks)-1):
    x1 = xgb_blocks[i][0] + xgb_blocks[i][2]
    x2 = xgb_blocks[i+1][0]
    arrow(ax, x1, 3.75, x2, 3.75, GREEN)

# AvgPool → 256-dim 연결
arrow(ax, 14.2, 6.5, 8.25, 4.5, CYAN)

# CBAM 상세 (좌하단)
cbam_x, cbam_y = 0.3, 0.5
rect = plt.Rectangle((cbam_x, cbam_y), 6.0, 2.5, facecolor='#151F30', edgecolor=PURPLE,
                      linewidth=2, zorder=3)
ax.add_patch(rect)
ax.text(cbam_x+3.0, cbam_y+2.2, 'CBAM (Convolutional Block Attention Module)', ha='center',
        fontsize=11, color=PURPLE, fontweight='bold', zorder=4)

ax.text(cbam_x+0.3, cbam_y+1.6, 'Channel Attention:', fontsize=9, color=CYAN, fontweight='bold', zorder=4)
ax.text(cbam_x+0.3, cbam_y+1.2, 'AvgPool + MaxPool -> FC(C/16) -> Sigmoid', fontsize=8, color=LIGHT, zorder=4, fontfamily='monospace')
ax.text(cbam_x+0.3, cbam_y+0.85, '"어떤 주파수 대역이 중요한가?"', fontsize=8, color=GRAY, style='italic', zorder=4)

ax.text(cbam_x+3.3, cbam_y+1.6, 'Spatial Attention:', fontsize=9, color=ORANGE, fontweight='bold', zorder=4)
ax.text(cbam_x+3.3, cbam_y+1.2, '[Mean,Max] -> Conv(7x7) -> Sigmoid', fontsize=8, color=LIGHT, zorder=4, fontfamily='monospace')
ax.text(cbam_x+3.3, cbam_y+0.85, '"스펙트로그램 어디가 중요한가?"', fontsize=8, color=GRAY, style='italic', zorder=4)

# Dropout 전략 (우하단)
rect2 = plt.Rectangle((7.0, 0.5), 4.5, 2.0, facecolor='#151F30', edgecolor=ORANGE,
                       linewidth=2, zorder=3)
ax.add_patch(rect2)
ax.text(9.25, 2.2, 'Dropout Strategy', ha='center', fontsize=11, color=ORANGE, fontweight='bold', zorder=4)
ax.text(7.3, 1.7, 'Block-level:  Dropout2d(0.2~0.3)', fontsize=9, color=LIGHT, zorder=4, fontfamily='monospace')
ax.text(7.3, 1.25, 'FC-level:     Dropout(0.5)', fontsize=9, color=LIGHT, zorder=4, fontfamily='monospace')
ax.text(7.3, 0.8, 'V14 교훈: CBAM 과적합 방지에 필수', fontsize=9, color=GREEN, zorder=4)

# 학습 설정 (우하단)
rect3 = plt.Rectangle((12.0, 0.5), 5.5, 2.0, facecolor='#151F30', edgecolor=GREEN,
                       linewidth=2, zorder=3)
ax.add_patch(rect3)
ax.text(14.75, 2.2, 'Training Config', ha='center', fontsize=11, color=GREEN, fontweight='bold', zorder=4)
ax.text(12.3, 1.7, 'Optimizer: Adam(lr=0.001)', fontsize=9, color=LIGHT, zorder=4, fontfamily='monospace')
ax.text(12.3, 1.3, 'Loss: CrossEntropyLoss (3-class)', fontsize=9, color=LIGHT, zorder=4, fontfamily='monospace')
ax.text(12.3, 0.9, 'Batch: 32 | Epochs: 100 | EarlyStop: 7', fontsize=9, color=LIGHT, zorder=4, fontfamily='monospace')

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '03_architecture_detail.png'), dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print("3/7 saved")


# ═════════════════════════════════════════════
# Figure 4: 전처리 파이프라인 상세
# ═════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 8), facecolor=BG)
style_ax(ax, 'V15 Preprocessing Pipeline — Sliding Window + Mel-Spectrogram')
ax.set_xlim(0, 16); ax.set_ylim(0, 9)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

# 슬라이딩 윈도우 시각화 (상단)
ax.text(8, 8.7, 'Sliding Window: 5s Window, 2s Stride (60% Overlap)', ha='center',
        fontsize=13, color=CYAN, fontweight='bold')

# 원본 오디오 바
rect_audio = plt.Rectangle((1, 7.5), 14, 0.7, facecolor='#1E3A5F', edgecolor=CYAN, linewidth=1.5)
ax.add_patch(rect_audio)
ax.text(8, 7.85, 'Original Audio (variable length, resampled to 4000 Hz)', ha='center',
        fontsize=10, color=WHITE)

# 윈도우들
window_colors = [CYAN, PURPLE, GREEN, ORANGE, PINK]
for i in range(5):
    x_start = 1 + i * 1.6  # 2s stride in visual
    alpha = 0.6 if i < 5 else 0.3
    rect_w = plt.Rectangle((x_start, 6.2), 4.0, 0.8, facecolor=window_colors[i % 5],
                            alpha=0.25, edgecolor=window_colors[i % 5], linewidth=1.5)
    ax.add_patch(rect_w)
    ax.text(x_start + 2.0, 6.6, f'Win {i+1}\n(5s = 20,000 samples)', ha='center',
            fontsize=7, color=WHITE)

# 화살표: 첫 윈도우 → Mel-Spec
ax.annotate('', xy=(3, 5.6), xytext=(3, 6.2),
            arrowprops=dict(arrowstyle='->', color=CYAN, lw=2))

# Mel-Spectrogram 변환 과정
steps = [
    (0.5, 4.0, 3.0, 1.3, 'librosa.melspectrogram\n+ power_to_db', 'SR=4000, n_mels=64\nhop_length=256\nref=1.0 (FIXED)', CYAN),
    (4.2, 4.0, 3.0, 1.3, 'Output Tensor', '(1, 64, 79)\n1ch x 64 mel x 79 frames\n= 5,056 features', PURPLE),
    (8.0, 4.0, 3.5, 1.3, 'ref=1.0 vs ref=np.max', 'V14: ref=np.max\n -> 무음이 Normal과 동일 패턴\nV15: ref=1.0 고정\n -> 절대 에너지 보존', ORANGE),
]
for x, y, w, h, label, desc, color in steps:
    make_box(ax, x, y, w, h, label, '', color, 10)
    lines = desc.split('\n')
    for j, line in enumerate(lines):
        ax.text(x + w/2, y + h*0.55 - j*0.22, line, ha='center', fontsize=8,
                color=LIGHT, zorder=4)

arrow(ax, 3.5, 4.65, 4.2, 4.65, CYAN)
arrow(ax, 7.2, 4.65, 8.0, 4.65, GRAY)

# 하단: 핵심 파라미터 비교표
ax.text(8, 2.8, 'V14 vs V15 Preprocessing Parameters', ha='center', fontsize=12,
        color=WHITE, fontweight='bold')

params = [
    ('Parameter', 'V14', 'V15', 'Reason'),
    ('Sample Rate', '1000 Hz', '4000 Hz', '더 넓은 주파수 범위 (2kHz까지)'),
    ('Input Shape', '4ch x 64x64', '1ch x 64x79', '단일 mel-spec (주파수 분리 불필요)'),
    ('Window', 'Cardiac Cycle 기반', '5s Fixed Window', '일관된 세그먼트 길이'),
    ('Stride', 'N_CYCLES', '2s (60% overlap)', '자연 augmentation + 안정적 세그먼트'),
    ('dB Reference', 'np.max (상대)', '1.0 (절대)', '무음 오분류 방지'),
    ('Augmentation', 'SpecAug + Overlap', 'Sliding Window 자체', '별도 증강 불필요'),
]

header_colors = [CYAN, ORANGE, CYAN, GRAY]
for i, row in enumerate(params):
    y_pos = 2.4 - i * 0.3
    is_header = (i == 0)
    x_positions = [1.0, 4.5, 7.5, 10.5]
    for j, (val, xp) in enumerate(zip(row, x_positions)):
        c = header_colors[j] if is_header else (ORANGE if j == 1 else CYAN if j == 2 else LIGHT)
        b = is_header
        ax.text(xp, y_pos, val, fontsize=9 if not is_header else 10,
                color=c, fontweight='bold' if b else 'normal')

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '04_preprocessing_pipeline.png'), dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print("4/7 saved")


# ═════════════════════════════════════════════
# Figure 5: 추론 파이프라인 (Probability-based Voting)
# ═════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 8), facecolor=BG)
style_ax(ax, 'V15 Inference Pipeline — Probability-based Segment Voting')
ax.set_xlim(0, 16); ax.set_ylim(0, 9)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

# 상단: 입력 → 세그먼트화
make_box(ax, 0.3, 7, 2.5, 1.5, 'External\nWAV File', '임의 길이', GRAY, 11)
make_box(ax, 3.5, 7, 2.5, 1.5, 'Sliding\nWindow', '5s / 2s stride', CYAN, 11)
arrow(ax, 2.8, 7.75, 3.5, 7.75, GRAY)

# N개 세그먼트
for i in range(4):
    x = 6.8 + i * 1.1
    alpha = 1.0 if i < 3 else 0.5
    rect = plt.Rectangle((x, 7.1), 0.9, 1.3, facecolor=CARD, edgecolor=PURPLE,
                          linewidth=1.5, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    label = f'Seg {i+1}' if i < 3 else '...'
    ax.text(x+0.45, 7.75, label, ha='center', fontsize=9, color=PURPLE, zorder=4)

arrow(ax, 6.0, 7.75, 6.8, 7.75, CYAN)

# 각 세그먼트 → CNN → XGBoost → 확률
y_mid = 4.8
for i in range(3):
    seg_x = 6.8 + i * 1.1
    # 세그먼트에서 아래로
    ax.annotate('', xy=(seg_x+0.45, y_mid+1.5), xytext=(seg_x+0.45, 7.1),
                arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1, ls='--'))

# CNN + XGBoost 블록
make_box(ax, 1.0, y_mid, 3.0, 1.2, 'ResNet+CBAM', '256-dim feature extraction', CYAN, 11)
make_box(ax, 5.0, y_mid, 3.0, 1.2, 'XGBoost', 'predict_proba()', GREEN, 11)
arrow(ax, 4.0, y_mid+0.6, 5.0, y_mid+0.6, CYAN)

# 확률 출력 예시
prob_data = [
    ('Seg 1', [0.82, 0.15, 0.03], GREEN),
    ('Seg 2', [0.78, 0.18, 0.04], GREEN),
    ('Seg 3', [0.70, 0.25, 0.05], ORANGE),
]

for i, (seg_label, probs, color) in enumerate(prob_data):
    x = 9.0 + i * 2.3
    rect = plt.Rectangle((x, y_mid-0.1), 2.0, 1.4, facecolor=CARD, edgecolor=color,
                          linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x+1.0, y_mid+1.05, seg_label, ha='center', fontsize=10, color=color, fontweight='bold', zorder=4)
    labels = ['N', 'A', 'U']
    for j, (lbl, p) in enumerate(zip(labels, probs)):
        bar_w = p * 1.5
        bar_color = [GREEN, RED, ORANGE][j]
        bar_rect = plt.Rectangle((x+0.15, y_mid+0.55-j*0.25), bar_w, 0.18,
                                  facecolor=bar_color, alpha=0.7, zorder=4)
        ax.add_patch(bar_rect)
        ax.text(x+0.1, y_mid+0.60-j*0.25, f'{lbl}: {p:.0%}', fontsize=7, color=WHITE, zorder=5)

arrow(ax, 8.0, y_mid+0.6, 9.0, y_mid+0.6, GREEN)

# 평균 확률 계산
make_box(ax, 4.0, 2.0, 4.0, 1.5, 'Average Probabilities', 'avg = mean(all segments)', ORANGE, 11)

avg_probs = [0.767, 0.193, 0.040]
for j, (lbl, p, c) in enumerate(zip(['Normal', 'Abnormal', 'Unknown'], avg_probs, [GREEN, RED, ORANGE])):
    ax.text(8.5, 3.0 - j*0.35, f'{lbl}: {p:.1%}', fontsize=11, color=c, fontweight='bold')

# argmax → 최종 결과
make_box(ax, 12.0, 2.0, 3.5, 1.5, 'argmax(avg)', 'Final: Normal (76.7%)', GREEN, 12)
arrow(ax, 8.0, 2.75, 12.0, 2.75, ORANGE)

# 세그먼트에서 평균으로
for i in range(3):
    x = 10.0 + i * 2.3
    ax.annotate('', xy=(6.0, 3.5), xytext=(x, y_mid-0.1),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=0.8, ls=':'))

# 하단 설명
ax.text(8, 0.8, 'V14: CNN_w * cnn_prob + (1-CNN_w) * xgb_prob > threshold → binary',
        ha='center', fontsize=10, color=ORANGE, style='italic')
ax.text(8, 0.35, 'V15: mean(xgb.predict_proba(all_segments)) → argmax → 3-class with confidence',
        ha='center', fontsize=10, color=CYAN, fontweight='bold')

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '05_inference_pipeline.png'), dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print("5/7 saved")


# ═════════════════════════════════════════════
# Figure 6: Unknown 클래스 도입 근거 + ref=1.0 문제 해결
# ═════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG)

# Panel 1: Unknown 클래스가 없을 때 문제
ax = axes[0, 0]
style_ax(ax, 'Binary (V14): Unknown이 없을 때', '', '')
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

# 2-class 상황에서 환경음이 들어오면
ax.text(5, 5.5, '환경음(키보드 소리, 대화 등)이 입력되면?', ha='center', fontsize=11, color=RED, fontweight='bold')

bars_2class = ax.barh(['Normal', 'Abnormal'], [0.45, 0.55], color=[GREEN, RED], height=0.4)
ax.set_xlim(0, 1.1)
ax.text(0.47, 0, '45%', fontsize=12, color=GREEN, fontweight='bold', va='center')
ax.text(0.57, 1, '55%', fontsize=12, color=RED, fontweight='bold', va='center')
ax.text(5, -0.8, '-> Abnormal로 오진!', ha='center', fontsize=14, color=RED, fontweight='bold')
ax.text(5, -1.5, '(환경음은 비주기적 -> Abnormal과 유사)', ha='center', fontsize=10, color=GRAY)

# Panel 2: 3-class에서 Unknown이 분리
ax = axes[0, 1]
style_ax(ax, 'Ternary (V15): Unknown 클래스 추가', '', '')
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

ax.text(5, 5.5, '동일한 환경음 입력:', ha='center', fontsize=11, color=GREEN, fontweight='bold')

bars_3class = ax.barh(['Normal', 'Abnormal', 'Unknown'], [0.08, 0.12, 0.80],
                       color=[GREEN, RED, ORANGE], height=0.4)
ax.set_xlim(0, 1.1)
ax.text(0.10, 0, '8%', fontsize=12, color=GREEN, fontweight='bold', va='center')
ax.text(0.14, 1, '12%', fontsize=12, color=RED, fontweight='bold', va='center')
ax.text(0.82, 2, '80%', fontsize=12, color=ORANGE, fontweight='bold', va='center')
ax.text(5, -0.8, '-> Unknown 판정 (재청진 요청)', ha='center', fontsize=14, color=GREEN, fontweight='bold')
ax.text(5, -1.5, 'ESC-50 2000개 + ICS43434 노이즈로 학습', ha='center', fontsize=10, color=GRAY)

# Panel 3: ref=np.max 문제
ax = axes[1, 0]
style_ax(ax, 'ref=np.max 문제 (V14)', 'Time Frame', 'Mel Band')

np.random.seed(42)
# 무음 mel-spec (ref=np.max -> 상대값으로 패턴 생성)
silence_rel = np.random.rand(64, 79) * 0.3
# 인위적 패턴 생성 (정규화로 인해 무음에서도 패턴이 보임)
for i in range(64):
    silence_rel[i, :] += 0.2 * np.sin(i / 64 * np.pi * 3) * np.random.rand()
silence_rel = (silence_rel - silence_rel.min()) / (silence_rel.max() - silence_rel.min() + 1e-8) * 80

ax.imshow(silence_rel, aspect='auto', cmap='magma', origin='lower', vmin=0, vmax=80)
ax.text(39, 60, 'ref=np.max: 무음도 패턴이 생김', ha='center', fontsize=10, color=WHITE,
        fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
ax.text(39, 52, '-> Normal로 오분류!', ha='center', fontsize=11, color=RED, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

# Panel 4: ref=1.0 해결
ax = axes[1, 1]
style_ax(ax, 'ref=1.0 해결 (V15)', 'Time Frame', 'Mel Band')

# 무음 mel-spec (ref=1.0 -> 절대값으로 거의 균일한 낮은 값)
silence_abs = np.random.rand(64, 79) * 3  # 절대값이라 매우 낮음
ax.imshow(silence_abs, aspect='auto', cmap='magma', origin='lower', vmin=0, vmax=80)
ax.text(39, 60, 'ref=1.0: 무음은 균일한 저에너지', ha='center', fontsize=10, color=WHITE,
        fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
ax.text(39, 52, '-> Unknown으로 정확 분류', ha='center', fontsize=11, color=GREEN, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

fig.suptitle('V15 핵심 개선 — Unknown 클래스 도입 + ref=1.0 고정',
             color=WHITE, fontsize=16, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '06_unknown_and_ref_fix.png'), dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print("6/7 saved")


# ═════════════════════════════════════════════
# Figure 7: V1~V15 전체 진화 종합 타임라인
# ═════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(18, 8), facecolor=BG)
style_ax(ax, 'Heart Sound Classification — Complete Evolution (V1 to V15)')
ax.set_xlim(-0.5, 15.5); ax.set_ylim(-1, 9)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

# 타임라인 축
ax.axhline(y=4.5, color='#334155', linewidth=2.5, zorder=1)

# 버전 마일스톤
milestones = [
    (0, 'V1-V2', '0.52', 'ResNet-18\nImageNet\n224x224', GRAY, 7),
    (1.5, 'V4', '0.65', '라벨 수정\nMACC 도입', ORANGE, 2),
    (3, 'V5', '0.71', 'CNN+XGB\nGrid Search\n40x32', CYAN, 7),
    (4.5, 'V6', '<V5', '증강 실패\n해상도\n병목 발견', RED, 2),
    (6, 'V7', '0.72', 'Focal Loss\n+CBAM\n과적합 발견', PURPLE, 7.5),
    (7.5, 'V8-9', '0.78', '64x64/128x128\n해상도 확대\nGolden Balance', GREEN, 2),
    (9, 'V10-12', '0.83', 'Cardiac Cycle\nChunking\n3/5-cycle', CYAN, 7),
    (10.5, 'V13', '0.85', '128x128\n+Chunking\nResNet-34', PURPLE, 2),
    (12, 'V14', '0.91', 'Ablation Study\nOverlap+SpecAug\n+Dropout', GREEN, 7.5),
    (14, 'V15', 'Prod', '3-Class 분류\n4000Hz + ref=1.0\nUnknown + Device', CYAN, 1.5),
]

for x, label, score, desc, color, y_pos in milestones:
    # 마커
    ax.plot(x, 4.5, 'o', color=color, markersize=14, zorder=5, markeredgecolor=WHITE, markeredgewidth=1.5)
    # 버전 + 스코어
    score_color = GREEN if score not in ['<V5', 'Prod'] else (RED if score == '<V5' else CYAN)
    ax.text(x, 5.0 if y_pos > 4.5 else 4.0, f'{label}\n{score}', ha='center',
            va='bottom' if y_pos > 4.5 else 'top',
            fontsize=9, color=color, fontweight='bold')
    # 설명 박스
    ax.annotate(desc, (x, 4.5), xytext=(x, y_pos), ha='center', va='center',
                fontsize=7, color=WHITE,
                arrowprops=dict(arrowstyle='-', color=color, lw=1, ls='--'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor=CARD, edgecolor=color, alpha=0.9))

# V15 하이라이트
rect_v15 = plt.Rectangle((13, 0.3), 2, 2.5, facecolor='none', edgecolor=CYAN,
                           linewidth=3, linestyle='--', zorder=6)
ax.add_patch(rect_v15)

# Phase 라벨
phase_labels = [
    (0.75, 8.5, 'Phase 1: 기본', GRAY),
    (3, 8.5, 'Phase 2: 탐색', GREEN),
    (4.5, 8.5, 'Phase 3: 증강', RED),
    (9, 8.5, 'Phase 4: 고급 기법', PURPLE),
    (14, 8.5, 'V15: Production', CYAN),
]
for x, y, label, color in phase_labels:
    ax.text(x, y, label, ha='center', fontsize=9, color=color,
            bbox=dict(boxstyle='round,pad=0.2', facecolor=CARD, edgecolor=color, alpha=0.7))

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '07_complete_evolution.png'), dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print("7/7 saved")


print(f"\nAll charts saved to: {OUT_DIR}")
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith('.png'):
        sz = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
        print(f"  {f} ({sz:.0f} KB)")
