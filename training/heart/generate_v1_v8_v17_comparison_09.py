"""
V1 vs V8 vs V17 성능 비교 — 발표용 pandas 테이블 + matplotlib 시각화
V17: PeakNorm + BLE데이터 + File-level split + SpecAugment + CosineAnnealing + LabelSmoothing
"""
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = r"G:\stetho_ai\Heart_binary_classification\figures_v15"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 색상 팔레트 ──
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

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(BG)
    ax.set_title(title, color=WHITE, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel, color=LIGHT, fontsize=11)
    ax.set_ylabel(ylabel, color=LIGHT, fontsize=11)
    ax.tick_params(colors=LIGHT, labelsize=10)
    for s in ax.spines.values():
        s.set_color('#334155')


# ═══════════════════════════════════════════
# 1. 데이터 정의 (V17 실측값)
# ═══════════════════════════════════════════
versions = {
    'V1': {
        'macc': 0.52,
        'sensitivity': 0.38,
        'specificity': 0.66,
        'accuracy': 0.58,
        'classification': 'Binary (2-Class)',
        'preprocessing': 'Cardiac Cycle\n1000Hz, 224x224',
        'model': 'ResNet-18\n(ImageNet)',
        'key_feature': '초기 베이스라인',
        'unknown': '없음',
    },
    'V8': {
        'macc': 0.78,
        'sensitivity': 0.72,
        'specificity': 0.84,
        'accuracy': 0.81,
        'classification': 'Binary (2-Class)',
        'preprocessing': '슬라이딩 윈도우 도입\n1000Hz, 64x64',
        'model': 'ResNet + CBAM\n+ XGBoost',
        'key_feature': '윈도우 최초 도입',
        'unknown': '없음',
    },
    'V17': {
        'macc': 0.8414,
        'sensitivity': 0.8276,
        'specificity': 0.8552,
        'accuracy': 0.8388,
        'classification': '3-Class',
        'preprocessing': '5s Window, 2s Stride\n4000Hz, 64x79\nPeak Normalization',
        'model': 'Custom ResNet 3-block\n+ CBAM + XGBoost',
        'key_feature': '3-Class + File Split\n+ SpecAugment\n+ PeakNorm + BLE데이터',
        'unknown': 'ESC-50 2000개\n+ ICS43434 노이즈',
    },
}

# V17 클래스별 성능 (실측)
v17_per_class = {
    'Normal':   {'precision': 0.84, 'recall': 0.84, 'f1': 0.84},
    'Abnormal': {'precision': 0.83, 'recall': 0.83, 'f1': 0.84},
    'Unknown':  {'precision': 0.88, 'recall': 0.86, 'f1': 0.87},
}

ver_labels = ['V1', 'V8', 'V17']
ver_colors = [GRAY, ORANGE, CYAN]


# ═══════════════════════════════════════════
# 2. Pandas 테이블 생성
# ═══════════════════════════════════════════
metrics_data = []
for ver, d in versions.items():
    metrics_data.append({
        '버전': ver,
        'MACC (%)': round(d['macc'] * 100, 2),
        'Sensitivity (%)': round(d['sensitivity'] * 100, 2),
        'Specificity (%)': round(d['specificity'] * 100, 2),
        'Accuracy (%)': round(d['accuracy'] * 100, 2),
    })
df_metrics = pd.DataFrame(metrics_data)

df_improvement = pd.DataFrame([{
    '버전': 'V1->V8 개선',
    'MACC (%)': round((versions['V8']['macc'] - versions['V1']['macc']) * 100, 2),
    'Sensitivity (%)': round((versions['V8']['sensitivity'] - versions['V1']['sensitivity']) * 100, 2),
    'Specificity (%)': round((versions['V8']['specificity'] - versions['V1']['specificity']) * 100, 2),
    'Accuracy (%)': round((versions['V8']['accuracy'] - versions['V1']['accuracy']) * 100, 2),
}, {
    '버전': 'V8->V17 개선',
    'MACC (%)': round((versions['V17']['macc'] - versions['V8']['macc']) * 100, 2),
    'Sensitivity (%)': round((versions['V17']['sensitivity'] - versions['V8']['sensitivity']) * 100, 2),
    'Specificity (%)': round((versions['V17']['specificity'] - versions['V8']['specificity']) * 100, 2),
    'Accuracy (%)': round((versions['V17']['accuracy'] - versions['V8']['accuracy']) * 100, 2),
}, {
    '버전': 'V1->V17 총 개선',
    'MACC (%)': round((versions['V17']['macc'] - versions['V1']['macc']) * 100, 2),
    'Sensitivity (%)': round((versions['V17']['sensitivity'] - versions['V1']['sensitivity']) * 100, 2),
    'Specificity (%)': round((versions['V17']['specificity'] - versions['V1']['specificity']) * 100, 2),
    'Accuracy (%)': round((versions['V17']['accuracy'] - versions['V1']['accuracy']) * 100, 2),
}])
df_full = pd.concat([df_metrics, df_improvement], ignore_index=True)

structure_data = []
for ver, d in versions.items():
    structure_data.append({
        '버전': ver,
        '분류 체계': d['classification'],
        '전처리': d['preprocessing'].replace('\n', ', '),
        '모델': d['model'].replace('\n', ', '),
        '핵심 변경': d['key_feature'].replace('\n', ', '),
        'Unknown 처리': d['unknown'].replace('\n', ', '),
    })
df_structure = pd.DataFrame(structure_data)

class_data = []
for cls_name, cls_metrics in v17_per_class.items():
    class_data.append({
        '클래스': cls_name,
        'Precision (%)': round(cls_metrics['precision'] * 100, 2),
        'Recall (%)': round(cls_metrics['recall'] * 100, 2),
        'F1-Score (%)': round(cls_metrics['f1'] * 100, 2),
    })
df_v17_class = pd.DataFrame(class_data)

print("\n" + "=" * 70)
print("  테이블 1: V1 vs V8 vs V17 핵심 성능 지표")
print("=" * 70)
print(df_full.to_string(index=False))
print("\n" + "=" * 70)
print("  테이블 2: 구조적 변경 비교")
print("=" * 70)
print(df_structure.to_string(index=False))
print("\n" + "=" * 70)
print("  테이블 3: V17 클래스별 성능")
print("=" * 70)
print(df_v17_class.to_string(index=False))

# CSV 저장
csv_metrics_path = os.path.join(OUT_DIR, 'v1_v8_v17_metrics.csv')
csv_structure_path = os.path.join(OUT_DIR, 'v1_v8_v17_structure.csv')
csv_class_path = os.path.join(OUT_DIR, 'v1_v8_v17_class.csv')
df_full.to_csv(csv_metrics_path, index=False, encoding='utf-8-sig')
df_structure.to_csv(csv_structure_path, index=False, encoding='utf-8-sig')
df_v17_class.to_csv(csv_class_path, index=False, encoding='utf-8-sig')


# ═══════════════════════════════════════════
# 3. Figure: 6-패널 종합 비교 차트
# ═══════════════════════════════════════════
fig = plt.figure(figsize=(20, 14), facecolor=BG)
fig.suptitle('Heart Sound Classification — V1 vs V8 vs V17 종합 비교',
             color=WHITE, fontsize=20, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3,
                       left=0.06, right=0.96, top=0.92, bottom=0.06)

# ─── Panel 1: 핵심 지표 그룹 막대 ───
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, '핵심 성능 지표 비교', '', '%')

metrics_names = ['MACC', 'Sensitivity', 'Specificity', 'Accuracy']
x = np.arange(len(metrics_names))
width = 0.22

for i, (ver, d) in enumerate(versions.items()):
    vals = [d['macc']*100, d['sensitivity']*100, d['specificity']*100, d['accuracy']*100]
    bars = ax1.bar(x + (i-1)*width, vals, width, label=ver, color=ver_colors[i],
                   edgecolor='none', alpha=0.9)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=8, color=ver_colors[i],
                 fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(metrics_names, fontsize=10)
ax1.set_ylim(0, 110)
ax1.legend(facecolor=CARD, edgecolor='#334155', labelcolor=WHITE, fontsize=10, loc='upper left')
ax1.grid(axis='y', color='#1E293B', linewidth=0.5, alpha=0.7)


# ─── Panel 2: MACC 진화 라인 차트 ───
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, 'MACC 진화 과정', '버전', 'MACC (%)')

all_versions = ['V1', 'V2', 'V4', 'V5', 'V7', 'V8', 'V9', 'V10', 'V12', 'V13', 'V14', 'V16', 'V17']
all_macc = [52, 52, 65, 71, 72, 78, 78, 83, 83, 85, 91, 82.4, 84.1]
x_all = range(len(all_versions))

ax2.plot(x_all, all_macc, color=GRAY, linewidth=1.5, alpha=0.5, zorder=1)
ax2.scatter(x_all, all_macc, color=GRAY, s=30, zorder=2, alpha=0.5)

# V14 과적합 표시
ax2.scatter([10], [91], color=RED, s=100, zorder=6, marker='x', linewidths=3)
ax2.annotate('V14-15: 데이터 누수\n(세그먼트 분할 오류)', (10, 91),
             xytext=(15, 20), textcoords='offset points',
             fontsize=9, color=RED, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor=CARD, edgecolor=RED, alpha=0.9))

# V1, V8, V17 강조
highlight_idx = [0, 5, 12]
highlight_colors = [GRAY, ORANGE, CYAN]
highlight_labels = ['V1 (52%)', 'V8 (78%)', 'V17 (84.1%)']

for idx, hc, hl in zip(highlight_idx, highlight_colors, highlight_labels):
    ax2.scatter([idx], [all_macc[idx]], color=hc, s=200, zorder=5,
                edgecolors=WHITE, linewidths=2)
    ax2.annotate(hl, (idx, all_macc[idx]),
                 xytext=(10, 15), textcoords='offset points',
                 fontsize=11, color=hc, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=hc, lw=1.5))

# V1->V8 개선폭
ax2.annotate('', xy=(5, 78), xytext=(0, 52),
             arrowprops=dict(arrowstyle='<->', color=GREEN, lw=2, ls='--'))
ax2.text(2.5, 63, '+26%p\n(윈도우 도입)', ha='center', fontsize=10,
         color=GREEN, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor=CARD, edgecolor=GREEN, alpha=0.9))

# V8->V17 개선폭
ax2.annotate('', xy=(12, 84.1), xytext=(5, 78),
             arrowprops=dict(arrowstyle='<->', color=CYAN, lw=2, ls='--'))
ax2.text(8.5, 80, '+6.1%p\n(PeakNorm+BLE)', ha='center', fontsize=10,
         color=CYAN, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor=CARD, edgecolor=CYAN, alpha=0.9))

ax2.set_xticks(x_all)
ax2.set_xticklabels(all_versions, fontsize=8, rotation=45)
ax2.set_ylim(40, 105)
ax2.grid(axis='y', color='#1E293B', linewidth=0.5, alpha=0.7)


# ─── Panel 3: Sensitivity vs Specificity 산점도 ───
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, 'Sensitivity vs Specificity 균형', 'Specificity (%)', 'Sensitivity (%)')

for i, (ver, d) in enumerate(versions.items()):
    sens = d['sensitivity'] * 100
    spec = d['specificity'] * 100
    ax3.scatter([spec], [sens], color=ver_colors[i], s=300,
                edgecolors=WHITE, linewidths=2, zorder=5)
    ax3.annotate(f'{ver}\n({sens:.1f}, {spec:.1f})',
                 (spec, sens), xytext=(12, -5), textcoords='offset points',
                 fontsize=10, color=ver_colors[i], fontweight='bold')

ax3.plot([30, 100], [30, 100], '--', color='#334155', linewidth=1, alpha=0.5)
ax3.text(45, 42, 'Sens = Spec', fontsize=8, color=GRAY, rotation=38, alpha=0.5)

for i in range(len(ver_labels) - 1):
    d1 = versions[ver_labels[i]]
    d2 = versions[ver_labels[i + 1]]
    ax3.annotate('', xy=(d2['specificity']*100, d2['sensitivity']*100),
                 xytext=(d1['specificity']*100, d1['sensitivity']*100),
                 arrowprops=dict(arrowstyle='->', color=LIGHT, lw=1.5, ls='--'))

ax3.set_xlim(30, 105)
ax3.set_ylim(30, 105)
ax3.grid(True, color='#1E293B', linewidth=0.5, alpha=0.7)


# ─── Panel 4: 구조 변경 비교 테이블 ───
ax4 = fig.add_subplot(gs[1, 0])
style_ax(ax4, '구조적 변경 비교')
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.set_xticks([])
ax4.set_yticks([])
ax4.grid(False)

rows = [
    ('분류', ['Binary\n(Normal/Abnormal)', 'Binary\n(Normal/Abnormal)', '3-Class\n(+Unknown)']),
    ('전처리', ['Cardiac Cycle\n1000Hz', '슬라이딩 윈도우\n1000Hz, 64x64', '5s Window, 2s Stride\n4kHz, PeakNorm']),
    ('모델', ['ResNet-18\n(ImageNet)', 'ResNet+CBAM\n+XGBoost', 'Custom ResNet\n+CBAM+XGBoost']),
    ('핵심', ['베이스라인', '윈도우 도입', 'PeakNorm+BLE\n+ SpecAugment']),
]

for j, (ver, color) in enumerate(zip(ver_labels, ver_colors)):
    ax4.text(3.5 + j * 2.3, 9.3, ver, ha='center', fontsize=13, color=color, fontweight='bold')

for i, (cat, vals) in enumerate(rows):
    y = 7.5 - i * 2.2
    ax4.text(0.5, y, cat, ha='left', va='center', fontsize=11, color=WHITE, fontweight='bold')
    for j, val in enumerate(vals):
        ax4.text(3.5 + j * 2.3, y, val, ha='center', va='center', fontsize=9, color=LIGHT,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor=CARD,
                           edgecolor=ver_colors[j], alpha=0.8, linewidth=1.5))


# ─── Panel 5: 개선폭 워터폴 차트 ───
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5, 'MACC 개선폭 분해', '', 'MACC (%)')

waterfall_labels = ['V1\n(시작)', 'V1->V8\n윈도우 도입', 'V8->V17\nPeakNorm+BLE', 'V17\n(최종)']
waterfall_bottoms = [0, 52, 78, 0]
waterfall_heights = [52, 26, 6.1, 84.1]
waterfall_colors = [GRAY, GREEN, CYAN, CYAN]

bars = ax5.bar(range(4), waterfall_heights, bottom=waterfall_bottoms,
               color=waterfall_colors, width=0.5, edgecolor='none', alpha=0.85)

ax5.plot([0.25, 1-0.25], [52, 52], color=LIGHT, linewidth=1, linestyle='--')
ax5.plot([1.25, 2-0.25], [78, 78], color=LIGHT, linewidth=1, linestyle='--')

for i, (b, h, bottom) in enumerate(zip(bars, waterfall_heights, waterfall_bottoms)):
    y_pos = bottom + h / 2
    label = f'{h:.1f}%' if i in [0, 3] else f'+{h:.1f}%p'
    ax5.text(b.get_x() + b.get_width()/2, y_pos, label,
             ha='center', va='center', fontsize=13, color=WHITE, fontweight='bold')

ax5.set_xticks(range(4))
ax5.set_xticklabels(waterfall_labels, fontsize=9)
ax5.set_ylim(0, 110)
ax5.grid(axis='y', color='#1E293B', linewidth=0.5, alpha=0.7)


# ─── Panel 6: V17 클래스별 F1-Score ───
ax6 = fig.add_subplot(gs[1, 2])
style_ax(ax6, 'V17 클래스별 성능', '', '%')

cls_names = ['Normal', 'Abnormal', 'Unknown']
cls_colors = [GREEN, RED, ORANGE]
f1_scores = [v17_per_class[c]['f1']*100 for c in cls_names]
precisions = [v17_per_class[c]['precision']*100 for c in cls_names]
recalls = [v17_per_class[c]['recall']*100 for c in cls_names]

x_cls = np.arange(len(cls_names))
w = 0.22

bars_p = ax6.bar(x_cls - w, precisions, w, label='Precision', color=cls_colors, alpha=0.5, edgecolor='none')
bars_r = ax6.bar(x_cls, recalls, w, label='Recall', color=cls_colors, alpha=0.75, edgecolor='none')
bars_f = ax6.bar(x_cls + w, f1_scores, w, label='F1-Score', color=cls_colors, alpha=1.0, edgecolor='none')

for bars_set in [bars_p, bars_r, bars_f]:
    for bar in bars_set:
        h = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                 f'{h:.1f}', ha='center', va='bottom', fontsize=8, color=WHITE)

ax6.set_xticks(x_cls)
ax6.set_xticklabels(cls_names, fontsize=11)
ax6.set_ylim(70, 105)

legend_elements = [
    mpatches.Patch(facecolor=GRAY, alpha=0.5, label='Precision'),
    mpatches.Patch(facecolor=GRAY, alpha=0.75, label='Recall'),
    mpatches.Patch(facecolor=GRAY, alpha=1.0, label='F1-Score'),
]
ax6.legend(handles=legend_elements, facecolor=CARD, edgecolor='#334155',
           labelcolor=WHITE, fontsize=9, loc='lower right')
ax6.grid(axis='y', color='#1E293B', linewidth=0.5, alpha=0.7)

chart_path = os.path.join(OUT_DIR, '09_v1_v8_v17_comparison.png')
fig.savefig(chart_path, dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print(f"\n[OK] 차트 저장: {chart_path}")


# ═══════════════════════════════════════════
# 4. Figure: 테이블 이미지
# ═══════════════════════════════════════════
fig_table, ax_t = plt.subplots(figsize=(14, 5), facecolor=BG)
ax_t.set_facecolor(BG)
ax_t.axis('off')
ax_t.set_title('V1 vs V8 vs V17 — 핵심 성능 지표 비교 (PeakNorm + BLE데이터 + File-Level Split)',
               color=WHITE, fontsize=15, fontweight='bold', pad=20)

col_labels = ['버전', 'MACC (%)', 'Sensitivity (%)', 'Specificity (%)', 'Accuracy (%)']

table_rows = []
table_row_colors = []

row_configs = [
    ('V1 (Baseline)', versions['V1'], GRAY),
    ('V8 (Windowing)', versions['V8'], ORANGE),
    ('V17 (Production)', versions['V17'], CYAN),
]

for ver_name, d, color in row_configs:
    table_rows.append([ver_name,
                       f'{d["macc"]*100:.2f}',
                       f'{d["sensitivity"]*100:.2f}',
                       f'{d["specificity"]*100:.2f}',
                       f'{d["accuracy"]*100:.2f}'])
    table_row_colors.append(color)

# 개선폭 행
improvements = [
    ('V1->V8 개선', versions['V1'], versions['V8'], GREEN),
    ('V8->V17 개선', versions['V8'], versions['V17'], GREEN),
    ('V1->V17 총 개선', versions['V1'], versions['V17'], GREEN),
]
for label, d1, d2, color in improvements:
    table_rows.append([label,
                       f'+{(d2["macc"]-d1["macc"])*100:.2f}',
                       f'+{(d2["sensitivity"]-d1["sensitivity"])*100:.2f}',
                       f'+{(d2["specificity"]-d1["specificity"])*100:.2f}',
                       f'+{(d2["accuracy"]-d1["accuracy"])*100:.2f}'])
    table_row_colors.append(color)

table = ax_t.table(cellText=table_rows, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.0, 2.0)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#334155')
    if row == 0:
        cell.set_facecolor('#2D3748')
        cell.set_text_props(color=WHITE, fontweight='bold', fontsize=12)
    elif row <= 3:
        cell.set_facecolor(CARD)
        color = table_row_colors[row - 1]
        cell.set_text_props(color=color if col == 0 else WHITE, fontsize=11,
                            fontweight='bold' if col == 0 else 'normal')
    else:
        cell.set_facecolor('#1E293B')
        cell.set_text_props(color=GREEN, fontsize=11, fontweight='bold')

table_img_path = os.path.join(OUT_DIR, '10_v1_v8_v17_table.png')
fig_table.savefig(table_img_path, dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig_table)
print(f"[OK] 테이블 이미지 저장: {table_img_path}")

print("\n완료!")
