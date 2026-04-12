"""
MACC 진화 타임라인 그래프 — V1~V15 전체 버전, Phase 구분, 핵심 포인트 강조
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
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

# ── 데이터 ──
versions = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15']
macc =     [52,   52,   58,   65,   71,   68,   72,   78,   78,   83,    83,    83,    85,    91,    95.7]
x = np.arange(len(versions))

# Phase 구분
phases = [
    (0, 3,  'Phase 1\n기본 파이프라인',   GRAY,   0.08),
    (3, 7,  'Phase 2\n앙상블 + 증강',     GREEN,  0.08),
    (7, 13, 'Phase 3\nAblation Study',    PURPLE, 0.08),
    (13, 15,'Phase 4\nProduction',        CYAN,   0.08),
]

# 주요 마일스톤
milestones = {
    0:  ('V1-2: ResNet-18\nImageNet 베이스라인', GRAY),
    3:  ('V4: 라벨 수정\nMACC 지표 도입', ORANGE),
    4:  ('V5: CNN+XGBoost\n앙상블 도입', GREEN),
    5:  ('V6: 증강 실패 ↓', RED),
    6:  ('V7: CBAM\nFocal Loss', PURPLE),
    7:  ('V8: 슬라이딩 윈도우\n최초 도입 ★', GREEN),
    9:  ('V10-12: Cardiac\nCycle Chunking', CYAN),
    12: ('V13: 128×128\nResNet-34', PURPLE),
    13: ('V14: Ablation\n최적화 완료', ORANGE),
    14: ('V15: 3-Class\nref=1.0 수정', CYAN),
}

# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(18, 10), facecolor=BG)
ax.set_facecolor(BG)

# Phase 배경 영역
for start, end, label, color, alpha in phases:
    ax.axvspan(start - 0.5, end - 0.5, facecolor=color, alpha=alpha, zorder=0)
    mid_x = (start + end) / 2 - 0.5
    ax.text(mid_x, 42, label, ha='center', va='bottom', fontsize=10, color=color,
            fontweight='bold', alpha=0.9)

# 메인 라인
ax.plot(x, macc, color=LIGHT, linewidth=2, alpha=0.4, zorder=2)
ax.scatter(x, macc, color=LIGHT, s=40, zorder=3, alpha=0.4)

# 주요 포인트 강조
highlight_versions = {
    0: (GRAY, 120),    # V1
    7: (GREEN, 200),   # V8
    14: (CYAN, 250),   # V15
}
for idx, (color, size) in highlight_versions.items():
    ax.scatter([idx], [macc[idx]], color=color, s=size, zorder=6,
               edgecolors=WHITE, linewidths=2.5)
    # 글로우 효과
    ax.scatter([idx], [macc[idx]], color=color, s=size * 3, zorder=5, alpha=0.15)

# V6 실패 포인트
ax.scatter([5], [macc[5]], color=RED, s=100, zorder=6, marker='x', linewidths=3)

# 마일스톤 어노테이션
annotation_offsets = {
    0:  (-50, -45),
    3:  (-60, 30),
    4:  (15, 35),
    5:  (15, -40),
    6:  (-70, 30),
    7:  (15, -45),
    9:  (15, 30),
    12: (-70, 30),
    13: (15, -40),
    14: (15, 25),
}

for idx, (text, color) in milestones.items():
    offset = annotation_offsets.get(idx, (15, 25))
    ax.annotate(text, (idx, macc[idx]),
                xytext=offset, textcoords='offset points',
                fontsize=9, color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5, connectionstyle='arc3,rad=0.1'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor=CARD, edgecolor=color, alpha=0.9),
                zorder=7)

# V1→V8 개선폭 화살표
ax.annotate('', xy=(7, 77), xytext=(0, 53),
            arrowprops=dict(arrowstyle='<->', color=GREEN, lw=2.5, ls='--'))
ax.text(3.5, 62, '+26%p', ha='center', fontsize=16, color=GREEN, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor=BG, edgecolor=GREEN, alpha=0.95, linewidth=2))

# V8→V15 개선폭 화살표
ax.annotate('', xy=(14, 94.7), xytext=(7, 79),
            arrowprops=dict(arrowstyle='<->', color=CYAN, lw=2.5, ls='--'))
ax.text(10.5, 85, '+17.7%p', ha='center', fontsize=16, color=CYAN, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor=BG, edgecolor=CYAN, alpha=0.95, linewidth=2))

# 축 스타일
ax.set_xticks(x)
ax.set_xticklabels(versions, fontsize=12, color=LIGHT, fontweight='bold')
ax.set_ylabel('MACC (%)', fontsize=14, color=LIGHT, fontweight='bold')
ax.set_ylim(40, 105)
ax.set_xlim(-0.8, 15.3)

# y축 그리드
for yval in [50, 60, 70, 80, 90, 100]:
    ax.axhline(y=yval, color='#1E293B', linewidth=0.8, alpha=0.7)

ax.tick_params(colors=LIGHT, labelsize=11)
for s in ax.spines.values():
    s.set_color('#334155')

# 타이틀
ax.set_title('MACC 진화 과정  —  V1 (52%) → V8 (78%) → V15 (95.7%)',
             color=WHITE, fontsize=18, fontweight='bold', pad=20)

# 범례
legend_elements = [
    mpatches.Patch(facecolor=GRAY, alpha=0.3, label='Phase 1: 기본 파이프라인'),
    mpatches.Patch(facecolor=GREEN, alpha=0.3, label='Phase 2: 앙상블 + 증강'),
    mpatches.Patch(facecolor=PURPLE, alpha=0.3, label='Phase 3: Ablation Study'),
    mpatches.Patch(facecolor=CYAN, alpha=0.3, label='Phase 4: Production'),
    plt.Line2D([0], [0], marker='x', color=RED, linestyle='None', markersize=10, markeredgewidth=3, label='증강 실패 (V6)'),
]
ax.legend(handles=legend_elements, facecolor=CARD, edgecolor='#334155',
          labelcolor=WHITE, fontsize=10, loc='upper left')

fig.tight_layout()
path = os.path.join(OUT_DIR, '14_macc_evolution.png')
fig.savefig(path, dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print(f"[OK] {path}")
