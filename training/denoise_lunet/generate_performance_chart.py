import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import rcParams

rcParams['font.family'] = 'Malgun Gothic'
rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 성능 비교표 (테이블 이미지)
# ============================================================
fig1, ax1 = plt.subplots(figsize=(20, 12))
ax1.axis('off')

# 테이블 데이터
col_labels = ['모델', '구성', '복원율', 'ΔSNR', 'SI-SNR']

rows = [
    # 섹션 헤더는 None으로 표시
    None,  # [외부 비교군]
    ['CRNN', 'Phase 1 (스펙트로그램)', '54.10%', '+5.22 dB', '—'],
    ['LU-Net 논문', 'Ali et al. 2023', '—', '+5.575 dB', '—'],
    None,  # [우리 실험]
    ['LU-Net v1', 'LSTM + MSE', '54.20%', '+5.94 dB', '9.82 dB'],
    ['LU-Net v2', 'LSTM + Attn + Res + SI-SNR', '52.27%', '+5.59 dB', '12.76 dB'],
    ['TU-Net v1', 'TCN + MSE', '44.71%', '+4.35 dB', '8.14 dB'],
    ['TU-Net v2', 'TCN + Attn + Res + SI-SNR', '54.26%', '+5.63 dB', '12.59 dB'],
    None,  # [윈도우 크기]
    ['LU-Net v1', '5초 윈도우', '54.20%', '+5.94 dB', '9.82 dB'],
    ['LU-Net v1', '2초 윈도우', '50.44%', '+5.57 dB', '9.25 dB'],
]

section_headers = {
    0: '외부 비교군',
    3: '우리 실험 — 파형 도메인',
    8: '윈도우 크기 비교',
}

# 레이아웃
n_cols = 5
col_widths = [0.16, 0.30, 0.14, 0.14, 0.14]  # 비율 (합=0.88, 나머지 여백)
table_x = 0.06
table_w = 0.88
row_h = 0.058
header_h = 0.065
section_h = 0.048
start_y = 0.88

# 제목
ax1.text(0.5, 0.96, '디노이즈 모델 성능 비교표', fontsize=28, fontweight='bold',
         ha='center', va='center', color='#e94560', transform=ax1.transAxes)
ax1.text(0.5, 0.92, 'Denoising Model Performance Comparison', fontsize=15,
         ha='center', va='center', color='#8888aa', style='italic', transform=ax1.transAxes)

# 헤더 행
y = start_y
header_bg = mpatches.FancyBboxPatch((table_x, y - header_h), table_w, header_h,
                                     boxstyle="round,pad=0.005",
                                     facecolor='#e94560', edgecolor='#ff6b80', linewidth=2)
ax1.add_patch(header_bg)

cx = table_x
for j, (label, w) in enumerate(zip(col_labels, col_widths)):
    ax1.text(cx + w * table_w / 2, y - header_h / 2, label,
             fontsize=16, fontweight='bold', ha='center', va='center',
             color='white', transform=ax1.transAxes)
    cx += w * table_w

y -= header_h

# 데이터 행
highlight_rows = {5: '#12124a', 7: '#12124a'}  # LU-Net v2, TU-Net v2 강조
best_cells = {(4, 3): True, (5, 4): True, (7, 2): True}  # 최고 성능 셀

data_row_idx = 0
for i, row in enumerate(rows):
    if row is None:
        # 섹션 헤더
        sec_bg = mpatches.FancyBboxPatch((table_x, y - section_h), table_w, section_h,
                                          boxstyle="round,pad=0.003",
                                          facecolor='#1a1a3e', edgecolor='#4a4a6a', linewidth=1.5)
        ax1.add_patch(sec_bg)
        ax1.text(table_x + 0.015, y - section_h / 2, section_headers[i],
                 fontsize=14, fontweight='bold', ha='left', va='center',
                 color='#00d2ff', transform=ax1.transAxes)
        y -= section_h
    else:
        # 데이터 행
        bg_color = highlight_rows.get(i, '#0f0f2a' if data_row_idx % 2 == 0 else '#151535')
        row_bg = mpatches.FancyBboxPatch((table_x, y - row_h), table_w, row_h,
                                          boxstyle="round,pad=0.003",
                                          facecolor=bg_color, edgecolor='#2a2a4a', linewidth=1)
        ax1.add_patch(row_bg)

        cx = table_x
        for j, (val, w) in enumerate(zip(row, col_widths)):
            cell_x = cx + w * table_w / 2
            cell_y = y - row_h / 2

            # 최고 성능 셀 강조
            if (i, j) in best_cells:
                badge = mpatches.FancyBboxPatch(
                    (cell_x - 0.055, cell_y - 0.018), 0.11, 0.036,
                    boxstyle="round,pad=0.005",
                    facecolor='#e94560', edgecolor='none', alpha=0.3
                )
                ax1.add_patch(badge)
                ax1.text(cell_x, cell_y, val,
                         fontsize=15, fontweight='bold', ha='center', va='center',
                         color='#ff8fa3', transform=ax1.transAxes)
            else:
                fc = '#ffffff' if j == 0 else '#d0d0f0'
                fw = 'bold' if j == 0 else 'normal'
                fs = 14 if j <= 1 else 15
                ax1.text(cell_x, cell_y, val,
                         fontsize=fs, fontweight=fw, ha='center', va='center',
                         color=fc, transform=ax1.transAxes)
            cx += w * table_w

        y -= row_h
        data_row_idx += 1

# 지표 설명
y -= 0.025
legend_items = [
    ('복원율', 'MAE 기반 노이즈 제거 비율 (높을수록 좋음)'),
    ('ΔSNR', '입력 대비 출력 SNR 향상 (높을수록 좋음)'),
    ('SI-SNR', 'Scale-Invariant SNR — 위상/구조 보존 (높을수록 좋음)'),
]
for item_name, item_desc in legend_items:
    y -= 0.032
    ax1.text(table_x + 0.02, y, f'{item_name}:', fontsize=12, fontweight='bold',
             ha='left', va='center', color='#e94560', transform=ax1.transAxes)
    ax1.text(table_x + 0.10, y, item_desc, fontsize=12,
             ha='left', va='center', color='#8888aa', transform=ax1.transAxes)

fig1.patch.set_facecolor('#0a0a1a')
fig1.savefig('g:/stetho_ai/ShittyDenoise/performance_table.png',
             dpi=180, bbox_inches='tight', facecolor='#0a0a1a', edgecolor='none')
plt.close(fig1)
print("Done: performance_table.png")


# ============================================================
# 2. 성능 추이 그래프 (ΔSNR + SI-SNR 2패널)
# ============================================================
fig2, (ax_snr, ax_si) = plt.subplots(1, 2, figsize=(22, 10))

# 공통 스타일
for ax in [ax_snr, ax_si]:
    ax.set_facecolor('#0f0f2a')
    ax.spines['bottom'].set_color('#4a4a6a')
    ax.spines['left'].set_color('#4a4a6a')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='#8888aa', labelsize=13)
    ax.grid(axis='y', color='#2a2a4a', linewidth=0.8, linestyle='--')

models = ['CRNN\n(Phase 1)', 'LU-Net v1\n(LSTM+MSE)', 'LU-Net v2\n(LSTM+v2)', 'TU-Net v1\n(TCN+MSE)', 'TU-Net v2\n(TCN+v2)']
x = np.arange(len(models))

# --- ΔSNR 그래프 ---
dsnr = [5.22, 5.94, 5.59, 4.35, 5.63]
colors_snr = ['#ff6b35', '#00d2ff', '#00d2ff', '#e94560', '#e94560']
bars1 = ax_snr.bar(x, dsnr, width=0.55, color=colors_snr, edgecolor='white', linewidth=1.2, alpha=0.9)

# 원논문 기준선
ax_snr.axhline(y=5.575, color='#ffcc00', linewidth=2.5, linestyle='--', alpha=0.8)
ax_snr.text(4.35, 5.65, '원논문 (Ali 2023)\n+5.575 dB', fontsize=11,
            ha='right', va='bottom', color='#ffcc00', fontweight='bold')

# 값 라벨
for bar, val in zip(bars1, dsnr):
    ax_snr.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f'+{val:.2f}', ha='center', va='bottom', fontsize=14,
                fontweight='bold', color='white')

ax_snr.set_xticks(x)
ax_snr.set_xticklabels(models, fontsize=12, color='#d0d0f0')
ax_snr.set_ylabel('ΔSNR (dB)', fontsize=15, fontweight='bold', color='#d0d0f0')
ax_snr.set_title('ΔSNR — 노이즈 제거량', fontsize=20, fontweight='bold',
                  color='#00d2ff', pad=15)
ax_snr.set_ylim(0, 7)

# --- SI-SNR 그래프 ---
si_snr = [9.82, 12.76, 8.14, 12.59]
si_models = ['LU-Net v1\n(LSTM+MSE)', 'LU-Net v2\n(LSTM+v2)', 'TU-Net v1\n(TCN+MSE)', 'TU-Net v2\n(TCN+v2)']
x2 = np.arange(len(si_models))
colors_si = ['#00d2ff', '#00d2ff', '#e94560', '#e94560']
fill_colors = ['#00d2ff', '#1abc9c', '#e94560', '#1abc9c']
bars2 = ax_si.bar(x2, si_snr, width=0.55, color=fill_colors, edgecolor='white', linewidth=1.2, alpha=0.9)

# 최고값 강조
bars2[1].set_edgecolor('#ffcc00')
bars2[1].set_linewidth(3)

for bar, val in zip(bars2, si_snr):
    label = f'{val:.2f}'
    if val == 12.76:
        label += ' (Best)'
    ax_si.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
               label, ha='center', va='bottom', fontsize=14,
               fontweight='bold', color='white')

ax_si.set_xticks(x2)
ax_si.set_xticklabels(si_models, fontsize=12, color='#d0d0f0')
ax_si.set_ylabel('SI-SNR (dB)', fontsize=15, fontweight='bold', color='#d0d0f0')
ax_si.set_title('SI-SNR — 청음 품질 (위상/구조 보존)', fontsize=20, fontweight='bold',
                 color='#1abc9c', pad=15)
ax_si.set_ylim(0, 15)

# 전체 제목
fig2.suptitle('디노이즈 모델 성능 추이', fontsize=26, fontweight='bold',
              color='#e94560', y=1.02)

fig2.patch.set_facecolor('#0a0a1a')
plt.tight_layout(pad=2)
fig2.savefig('g:/stetho_ai/ShittyDenoise/performance_graph.png',
             dpi=180, bbox_inches='tight', facecolor='#0a0a1a', edgecolor='none')
plt.close(fig2)
print("Done: performance_graph.png")
