import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

rcParams['font.family'] = 'Malgun Gothic'
rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(20, 34))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

def rounded_box(ax, x, y, w, h, fc, ec, lw=2.5, alpha=0.95):
    box = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                   facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha)
    ax.add_patch(box)

def arrow(ax, y1, y2, x=50):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color='#e94560', lw=3, mutation_scale=22))

# 제목
ax.text(50, 99, '디노이즈 모델 — 전처리 파이프라인', fontsize=30, fontweight='bold',
        ha='center', va='top', color='#e94560')
ax.text(50, 97.3, 'Heart Sound Denoising Preprocessing Pipeline', fontsize=16,
        ha='center', va='top', color='#8888aa', style='italic')

# 단계 데이터
steps = [
    {
        'num': '입력', 'title': '원본 심음 WAV',
        'lines': ['가변 길이 · 가변 Sample Rate'],
        'fc': '#533483', 'ec': '#9b59b6'
    },
    {
        'num': '①', 'title': '4000Hz 리샘플링',
        'lines': [
            '심음 유효 대역: 20~600Hz',
            'Nyquist: 4000/2 = 2000Hz > 600Hz',
            '초기 2000Hz → 4000Hz 업그레이드 (S1/S2 고차 하모닉스 포착)',
        ],
        'fc': '#0f3460', 'ec': '#e94560'
    },
    {
        'num': '②', 'title': '피크 정규화 (peak = 0.9)',
        'lines': [
            'x = x / max(|x|) × 0.9',
            '음량 편차 제거 + 클리핑 방지 headroom 확보',
        ],
        'fc': '#16213e', 'ec': '#00d2ff'
    },
    {
        'num': '③', 'title': '무음 필터링 (RMS > 0.005)',
        'lines': [
            '무음 구간은 학습에 무의미 → 제거',
        ],
        'fc': '#1a1a2e', 'ec': '#ff6b35'
    },
    {
        'num': '④', 'title': '5초 고정 윈도우 (50% 오버랩)',
        'lines': [
            'chunk = 20,000 samples  /  hop = 10,000 samples',
            '오버랩 → 데이터 ~2배 증강 + 경계 심박 누락 방지',
        ],
        'fc': '#0f3460', 'ec': '#e94560'
    },
    {
        'num': '⑤', 'title': 'SNR 기반 노이즈 합성',
        'lines': [
            'SNR 7단계: [-5, -2, 0, 3, 5, 10, 15] dB',
            'noisy = clean + noise × (clean_rms / noise_rms / 10^(SNR/20))',
        ],
        'fc': '#16213e', 'ec': '#00d2ff'
    },
    {
        'num': '⑥', 'title': '데이터 증강',
        'lines': [
            'Gain Jitter: volume × random(0.7 ~ 1.3)',
            'Time Shift: np.roll(±0.2초)',
            '30% 확률 백색소음 추가 (σ = 0.002)',
        ],
        'fc': '#1a1a2e', 'ec': '#ff6b35'
    },
    {
        'num': '출력', 'title': '최종 데이터셋',
        'lines': [
            'X(10,000 × 20,000) — 노이즈 심음',
            'Y(10,000 × 20,000) — 클린 심음 (ground truth)',
            'Train 80% (8,000)  /  Val 20% (2,000)',
        ],
        'fc': '#0d7377', 'ec': '#1abc9c'
    },
]

# 레이아웃
box_w = 72
box_x = 14
arrow_gap = 1.5
line_h = 2.0  # 디테일 한 줄 높이
title_h = 3.5  # 제목 영역 높이
pad_bottom = 1.5

# 각 박스 높이 계산
heights = []
for s in steps:
    h = title_h + len(s['lines']) * line_h + pad_bottom
    heights.append(h)

total_needed = sum(heights) + arrow_gap * (len(steps) - 1)
available = 95  # y=96 ~ y=1
scale = available / total_needed

heights = [h * scale for h in heights]
arrow_gap_scaled = arrow_gap * scale
line_h_scaled = line_h * scale
title_h_scaled = title_h * scale
pad_bottom_scaled = pad_bottom * scale

y = 96

for i, (step, bh) in enumerate(zip(steps, heights)):
    box_y = y - bh

    # 박스
    rounded_box(ax, box_x, box_y, box_w, bh, step['fc'], step['ec'])

    # 번호 배지
    badge_x = box_x + 4.5
    badge_y = box_y + bh - title_h_scaled * 0.55
    circle = plt.Circle((badge_x, badge_y), 1.8,
                         facecolor=step['ec'], edgecolor='white', linewidth=2, zorder=5)
    ax.add_patch(circle)
    fs = 14 if len(step['num']) <= 1 else 11
    ax.text(badge_x, badge_y, step['num'], fontsize=fs, fontweight='bold',
            ha='center', va='center', color='white', zorder=6)

    # 제목
    ax.text(badge_x + 4, badge_y, step['title'],
            fontsize=18, fontweight='bold', ha='left', va='center', color='white')

    # 디테일 줄
    for j, line in enumerate(step['lines']):
        ly = box_y + bh - title_h_scaled - j * line_h_scaled - line_h_scaled * 0.3
        # 배경 바
        bar = mpatches.FancyBboxPatch(
            (box_x + 3, ly - line_h_scaled * 0.35), box_w - 6, line_h_scaled * 0.75,
            boxstyle="round,pad=0.15", facecolor='white', edgecolor='none', alpha=0.07
        )
        ax.add_patch(bar)
        ax.text(box_x + 5, ly, line, fontsize=14, ha='left', va='center', color='#d0d0f0')

    y = box_y

    # 화살표
    if i < len(steps) - 1:
        arrow(ax, y, y - arrow_gap_scaled)
        y -= arrow_gap_scaled

fig.patch.set_facecolor('#0a0a1a')
plt.tight_layout(pad=0.5)
plt.savefig('g:/stetho_ai/ShittyDenoise/preprocessing_pipeline.png',
            dpi=180, bbox_inches='tight', facecolor='#0a0a1a', edgecolor='none')
plt.close()
print("Done: preprocessing_pipeline.png")
