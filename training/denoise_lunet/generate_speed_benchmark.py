import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import rcParams

rcParams['font.family'] = 'Malgun Gothic'
rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(24, 10),
                         gridspec_kw={'width_ratios': [1, 1, 0.7]})

for ax in axes:
    ax.set_facecolor('#0f0f2a')
    ax.spines['bottom'].set_color('#4a4a6a')
    ax.spines['left'].set_color('#4a4a6a')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='#8888aa', labelsize=13)
    ax.grid(axis='y', color='#2a2a4a', linewidth=0.8, linestyle='--')

# ============================================================
# 1. 단일 추론 속도
# ============================================================
ax1 = axes[0]
models = ['LU-Net v2\n(LSTM)', 'TU-Net v2\n(TCN)']
times = [296.0, 86.1]
errs = [5.2, 1.4]
colors = ['#e94560', '#1abc9c']

bars1 = ax1.bar(models, times, width=0.5, color=colors, edgecolor='white',
                linewidth=2, alpha=0.9, yerr=errs, capsize=8,
                error_kw={'color': 'white', 'linewidth': 2})

for bar, val, err in zip(bars1, times, errs):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 5,
             f'{val:.1f} ms', ha='center', va='bottom', fontsize=18,
             fontweight='bold', color='white')

# 3.44x 화살표
ax1.annotate('3.44x\n빠름',
             xy=(1, 86.1), xytext=(0.5, 200),
             fontsize=16, fontweight='bold', color='#ffcc00',
             ha='center', va='center',
             arrowprops=dict(arrowstyle='->', color='#ffcc00', lw=2.5))

ax1.set_ylabel('추론 시간 (ms)', fontsize=15, fontweight='bold', color='#d0d0f0')
ax1.set_title('단일 추론 (5초 오디오 1개)', fontsize=18, fontweight='bold',
              color='#00d2ff', pad=15)
ax1.set_ylim(0, 370)

# ============================================================
# 2. 배치 추론 속도
# ============================================================
ax2 = axes[1]
times_batch = [2932.0, 848.6]

bars2 = ax2.bar(models, times_batch, width=0.5, color=colors, edgecolor='white',
                linewidth=2, alpha=0.9)

for bar, val in zip(bars2, times_batch):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
             f'{val:.0f} ms', ha='center', va='bottom', fontsize=18,
             fontweight='bold', color='white')

ax2.annotate('3.46x\n빠름',
             xy=(1, 848.6), xytext=(0.5, 2000),
             fontsize=16, fontweight='bold', color='#ffcc00',
             ha='center', va='center',
             arrowprops=dict(arrowstyle='->', color='#ffcc00', lw=2.5))

ax2.set_ylabel('추론 시간 (ms)', fontsize=15, fontweight='bold', color='#d0d0f0')
ax2.set_title('배치 추론 (5초 오디오 10개)', fontsize=18, fontweight='bold',
              color='#00d2ff', pad=15)
ax2.set_ylim(0, 3600)

# ============================================================
# 3. 파라미터 수 비교
# ============================================================
ax3 = axes[2]
params = [2.84, 3.26]  # 백만 단위
bars3 = ax3.bar(models, params, width=0.5, color=colors, edgecolor='white',
                linewidth=2, alpha=0.9)

for bar, val in zip(bars3, params):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
             f'{val:.2f}M', ha='center', va='bottom', fontsize=18,
             fontweight='bold', color='white')

# +15% 표시
ax3.text(1, params[1] + 0.25, '+15%',
         ha='center', va='bottom', fontsize=14, fontweight='bold',
         color='#ff6b35',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#ff6b35', alpha=0.2, edgecolor='#ff6b35'))

ax3.set_ylabel('파라미터 수 (M)', fontsize=15, fontweight='bold', color='#d0d0f0')
ax3.set_title('모델 크기', fontsize=18, fontweight='bold',
              color='#00d2ff', pad=15)
ax3.set_ylim(0, 4.2)

# ============================================================
# 전체 제목 + 핵심 메시지
# ============================================================
fig.suptitle('LSTM vs TCN 추론 속도 벤치마크 (CPU)', fontsize=26, fontweight='bold',
             color='#e94560', y=1.02)

# 하단 핵심 메시지 박스
fig.text(0.5, -0.04,
         'TCN은 파라미터가 15% 더 많지만, 병렬 처리 덕분에 추론 속도 3.4배 빠름\n'
         'LSTM의 순차 처리(sequential) 병목이 속도 차이의 원인',
         ha='center', va='center', fontsize=15, color='#d0d0f0',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#1a1a3e',
                   edgecolor='#ffcc00', linewidth=2))

fig.patch.set_facecolor('#0a0a1a')
plt.tight_layout(pad=2)
fig.savefig('g:/stetho_ai/ShittyDenoise/speed_benchmark.png',
            dpi=180, bbox_inches='tight', facecolor='#0a0a1a', edgecolor='none')
plt.close()
print("Done: speed_benchmark.png")
