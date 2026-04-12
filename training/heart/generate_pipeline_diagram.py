"""
2-Stage Ensemble Pipeline 다이어그램 — 발표용 이미지 생성
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = r"G:\stetho_ai\Heart_binary_classification\figures_v15"

BG     = '#0F172A'
CARD   = '#1A253C'
CYAN   = '#00D2FF'
PURPLE = '#7C3AED'
GREEN  = '#10B981'
RED    = '#EF4444'
ORANGE = '#F59E0B'
GRAY   = '#94A3B8'
WHITE  = '#E2E8F0'
LIGHT  = '#CBD5E1'
DARK   = '#1E293B'


def rounded_box(ax, x, y, w, h, text, color, fontsize=12, text_color=WHITE,
                subtext=None, subcolor=LIGHT, subfontsize=10):
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.15", facecolor=CARD, edgecolor=color,
        linewidth=2.5, zorder=3)
    ax.add_patch(rect)
    if subtext:
        ax.text(x, y + 0.15, text, ha='center', va='center',
                fontsize=fontsize, color=text_color, fontweight='bold', zorder=4)
        ax.text(x, y - 0.25, subtext, ha='center', va='center',
                fontsize=subfontsize, color=subcolor, zorder=4)
    else:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, color=text_color, fontweight='bold', zorder=4)


def arrow_down(ax, x, y1, y2, color=GRAY, lw=2):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))


def side_label(ax, x_arrow_end, y, text, color=LIGHT, fontsize=10):
    ax.text(x_arrow_end + 0.15, y, text, ha='left', va='center',
            fontsize=fontsize, color=color, style='italic')


# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 16), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(-4, 4)
ax.set_ylim(-1, 17)
ax.axis('off')

# 타이틀
ax.text(0, 16.3, '2-Stage Ensemble Pipeline', ha='center', va='center',
        fontsize=22, color=CYAN, fontweight='bold')
ax.text(0, 15.8, 'ResNet + CBAM + XGBoost', ha='center', va='center',
        fontsize=14, color=GRAY)

# ── Input ──
rounded_box(ax, 0, 14.8, 4.5, 0.7, 'Mel-Spectrogram  (1, 64, 79)', GRAY, fontsize=14)

arrow_down(ax, 0, 14.4, 13.8, GRAY)

# ── Phase 1 배경 ──
phase1_bg = mpatches.FancyBboxPatch(
    (-3.5, 5.0), 7.0, 8.5,
    boxstyle="round,pad=0.3", facecolor='#0D1425', edgecolor=PURPLE,
    linewidth=1.5, linestyle='--', alpha=0.6, zorder=1)
ax.add_patch(phase1_bg)
ax.text(-3.0, 13.2, 'Phase 1: CNN Feature Extraction', ha='left', va='center',
        fontsize=13, color=PURPLE, fontweight='bold', zorder=2)

# ── Conv2d ──
rounded_box(ax, 0, 12.5, 5.0, 0.8,
            'Conv2d (1 → 64) + BatchNorm + ReLU', CYAN, fontsize=13)
side_label(ax, 2.6, 12.5, 'k=3, pad=1', GRAY, 10)

arrow_down(ax, 0, 12.05, 11.3, CYAN)

# ── ResBlock 1 ──
rounded_box(ax, 0, 10.7, 5.0, 0.9,
            'ResBlock + CBAM', PURPLE, fontsize=14,
            subtext='64 → 64  |  Dropout2d 0.2', subcolor=LIGHT, subfontsize=11)
side_label(ax, 2.6, 10.7, 'stride=1', GRAY, 10)

arrow_down(ax, 0, 10.2, 9.5, PURPLE)

# ── ResBlock 2 ──
rounded_box(ax, 0, 8.9, 5.0, 0.9,
            'ResBlock + CBAM', PURPLE, fontsize=14,
            subtext='64 → 128  |  Dropout2d 0.2', subcolor=LIGHT, subfontsize=11)
side_label(ax, 2.6, 8.9, 'stride=2 ↓', ORANGE, 10)

arrow_down(ax, 0, 8.4, 7.7, PURPLE)

# ── ResBlock 3 ──
rounded_box(ax, 0, 7.1, 5.0, 0.9,
            'ResBlock + CBAM', PURPLE, fontsize=14,
            subtext='128 → 256  |  Dropout2d 0.3', subcolor=LIGHT, subfontsize=11)
side_label(ax, 2.6, 7.1, 'stride=2 ↓', ORANGE, 10)

arrow_down(ax, 0, 6.6, 5.9, PURPLE)

# ── AvgPool ──
rounded_box(ax, 0, 5.5, 5.0, 0.6,
            'AdaptiveAvgPool2d → Flatten', CYAN, fontsize=12)

arrow_down(ax, 0, 5.15, 4.4, GREEN, lw=3)

# ── Feature Vector ──
rounded_box(ax, 0, 3.9, 5.5, 0.8,
            '256-dim Feature Vector', GREEN, fontsize=15, text_color=GREEN)

# CNN이 추출한 고수준 특징 라벨
ax.annotate('CNN이 추출한\n고수준 특징', xy=(-2.8, 3.9), xytext=(-3.8, 3.9),
            ha='right', va='center', fontsize=10, color=GREEN, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))

arrow_down(ax, 0, 3.45, 2.7, ORANGE, lw=3)

# ── Phase 2 배경 ──
phase2_bg = mpatches.FancyBboxPatch(
    (-3.5, 0.8), 7.0, 2.5,
    boxstyle="round,pad=0.3", facecolor='#0D1425', edgecolor=ORANGE,
    linewidth=1.5, linestyle='--', alpha=0.6, zorder=1)
ax.add_patch(phase2_bg)
ax.text(-3.0, 3.0, 'Phase 2: XGBoost Classification', ha='left', va='center',
        fontsize=13, color=ORANGE, fontweight='bold', zorder=2)

# ── XGBoost ──
rounded_box(ax, 0, 2.2, 5.0, 0.8,
            'XGBoost', ORANGE, fontsize=15,
            subtext='100 trees  |  max_depth=6  |  GPU CUDA', subcolor=LIGHT, subfontsize=11)

arrow_down(ax, 0, 1.75, 1.1, ORANGE, lw=3)

# ── Output ──
out_w = 6.0
out_rect = mpatches.FancyBboxPatch(
    (-out_w/2, 0.15), out_w, 0.8,
    boxstyle="round,pad=0.15", facecolor=CARD, edgecolor=CYAN,
    linewidth=3, zorder=3)
ax.add_patch(out_rect)

# 3-class 확률을 색상별로
ax.text(-1.5, 0.55, 'P(Normal)', ha='center', va='center',
        fontsize=13, color=GREEN, fontweight='bold', zorder=4)
ax.text(0, 0.55, 'P(Abnormal)', ha='center', va='center',
        fontsize=13, color=RED, fontweight='bold', zorder=4)
ax.text(1.6, 0.55, 'P(Unknown)', ha='center', va='center',
        fontsize=13, color=ORANGE, fontweight='bold', zorder=4)

# 구분선
ax.plot([-0.75, -0.75], [0.25, 0.85], color='#334155', linewidth=1, zorder=4)
ax.plot([0.8, 0.8], [0.25, 0.85], color='#334155', linewidth=1, zorder=4)

# 하단 라벨
ax.text(0, -0.3, '3-Class 확률 출력 → Segment 평균 → 최종 판정',
        ha='center', va='center', fontsize=11, color=GRAY, style='italic')

# 저장
path = os.path.join(OUT_DIR, '13_pipeline_diagram.png')
fig.savefig(path, dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print(f"[OK] {path}")
