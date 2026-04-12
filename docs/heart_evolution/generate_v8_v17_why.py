"""
V8 vs V17 — 왜 V17이 더 좋은가 (구조적 개선 설명 다이어그램)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib
import numpy as np
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor('#0d1117')

# 전체를 하나의 axes로 사용 (다이어그램)
ax = fig.add_axes([0.02, 0.02, 0.96, 0.88])
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.set_facecolor('#0d1117')
ax.axis('off')

# ── 색상 정의 (색약 친화) ──
C_V8 = '#8b949e'       # 회색 (V8, 이전)
C_V17 = '#58a6ff'      # 파랑 (V17, 개선)
C_ACCENT = '#f0883e'   # 주황 (강조)
C_BOX_BG = '#161b22'   # 박스 배경
C_TEXT = '#e6edf3'      # 밝은 텍스트
C_DIM = '#8b949e'       # 흐린 텍스트

# ── 타이틀 ──
fig.text(0.5, 0.96, 'V8 → V17  개선 요약  —  왜 V17이 더 정확한가',
         fontsize=24, fontweight='bold', color='white', ha='center')
fig.text(0.5, 0.925, 'MACC 78% → 84.14%  |  Sensitivity 72% → 82.76%  (+10.76%p)',
         fontsize=14, color=C_ACCENT, ha='center')

# ── 헬퍼 함수 ──
def draw_box(x, y, w, h, title, items, title_color, border_color, bg_alpha=0.6):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=C_BOX_BG, edgecolor=border_color,
                          linewidth=2.5, alpha=bg_alpha)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.2, title, fontsize=14, fontweight='bold',
            color=title_color, ha='center', va='top')
    for i, (text, color) in enumerate(items):
        ax.text(x + 0.2, y + h - 0.55 - i * 0.35, text,
                fontsize=11, color=color, va='top', fontfamily=None)

def draw_arrow(x1, y1, x2, y2, text='', color=C_ACCENT):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5,
                                connectionstyle='arc3,rad=0'))
    if text:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.15
        ax.text(mx, my, text, fontsize=10, color=color, ha='center',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117',
                          edgecolor=color, alpha=0.9))

# ══════════════════════════════════════════════
# 4개 개선 포인트 (위에서 아래로)
# ══════════════════════════════════════════════

ROW_Y = [6.3, 4.5, 2.7, 0.9]
V8_X = 0.3
V17_X = 5.5
BOX_W = 4.0

# ── 1. 분류 체계 ──
draw_box(V8_X, ROW_Y[0], BOX_W, 1.4,
         'V8 — 분류 체계', [
             ('2-Class: 정상 / 비정상', C_DIM),
             ('환경음이 들어오면 → 오분류', '#f85149'),
             ('Unknown 클래스 없음', C_DIM),
         ], C_V8, C_V8)

draw_box(V17_X, ROW_Y[0], BOX_W, 1.4,
         'V17 — 분류 체계', [
             ('3-Class: 정상 / 비정상 / Unknown', C_TEXT),
             ('환경음 → Unknown으로 정확히 분류', '#58a6ff'),
             ('ESC-50 환경음 2000개 학습', C_TEXT),
         ], C_V17, C_V17)

draw_arrow(V8_X + BOX_W + 0.1, ROW_Y[0] + 0.7,
           V17_X - 0.1, ROW_Y[0] + 0.7,
           '+ Unknown 클래스')

# ── 2. 전처리 ──
draw_box(V8_X, ROW_Y[1], BOX_W, 1.4,
         'V8 — 전처리', [
             ('샘플레이트: 1000 Hz', C_DIM),
             ('스펙트로그램: 64 x 64', C_DIM),
             ('정규화 없음', '#f85149'),
         ], C_V8, C_V8)

draw_box(V17_X, ROW_Y[1], BOX_W, 1.4,
         'V17 — 전처리', [
             ('샘플레이트: 4000 Hz (4배)', C_TEXT),
             ('스펙트로그램: 64 x 79 + SpecAugment', C_TEXT),
             ('Peak Normalization (BLE 음량 보정)', '#58a6ff'),
         ], C_V17, C_V17)

draw_arrow(V8_X + BOX_W + 0.1, ROW_Y[1] + 0.7,
           V17_X - 0.1, ROW_Y[1] + 0.7,
           '해상도 4배 + 증강')

# ── 3. 모델 구조 ──
draw_box(V8_X, ROW_Y[2], BOX_W, 1.4,
         'V8 — 모델', [
             ('ResNet + CBAM + XGBoost', C_DIM),
             ('Dropout 없음', '#f85149'),
             ('기본 구조', C_DIM),
         ], C_V8, C_V8)

draw_box(V17_X, ROW_Y[2], BOX_W, 1.4,
         'V17 — 모델', [
             ('Custom ResNet 3-block + CBAM + XGBoost', C_TEXT),
             ('Dropout 0.2/0.3/0.5 (과적합 방지)', '#58a6ff'),
             ('File-level Split (데이터 누수 방지)', '#58a6ff'),
         ], C_V17, C_V17)

draw_arrow(V8_X + BOX_W + 0.1, ROW_Y[2] + 0.7,
           V17_X - 0.1, ROW_Y[2] + 0.7,
           '과적합 방지 강화')

# ── 4. 실환경 대응 ──
draw_box(V8_X, ROW_Y[3], BOX_W, 1.4,
         'V8 — 실환경', [
             ('학습 데이터만 사용', C_DIM),
             ('BLE 전송 노이즈 미고려', '#f85149'),
             ('병원 환경음 미고려', '#f85149'),
         ], C_V8, C_V8)

draw_box(V17_X, ROW_Y[3], BOX_W, 1.4,
         'V17 — 실환경', [
             ('ICS43434 마이크 노이즈 포함 학습', C_TEXT),
             ('BLE 전송 데이터로 검증', '#58a6ff'),
             ('ESC-50 환경음으로 Unknown 강화', '#58a6ff'),
         ], C_V17, C_V17)

draw_arrow(V8_X + BOX_W + 0.1, ROW_Y[3] + 0.7,
           V17_X - 0.1, ROW_Y[3] + 0.7,
           '실환경 노이즈 학습')

# ── 왼쪽/오른쪽 라벨 ──
ax.text(V8_X + BOX_W / 2, 7.9, 'V8 (이전)', fontsize=18, fontweight='bold',
        color=C_V8, ha='center')
ax.text(V17_X + BOX_W / 2, 7.9, 'V17 (현재)', fontsize=18, fontweight='bold',
        color=C_V17, ha='center')

plt.savefig(r"G:\stetho_ai\v8_v17_improvement.png", dpi=300,
            bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved: v8_v17_improvement.png")
plt.close()
