"""
기존 논문 vs 본 프로젝트 비교 — 발표용 테이블 이미지 생성
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = r"G:\stetho_ai\Heart_binary_classification\figures_v15"

BG     = '#0F172A'
CARD   = '#1A253C'
CYAN   = '#00D2FF'
GREEN  = '#10B981'
WHITE  = '#E2E8F0'
LIGHT  = '#CBD5E1'
ORANGE = '#F59E0B'


# ═══════════════════════════════════════════
# Figure 1: 동일한 점
# ═══════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(12, 4), facecolor=BG)
ax1.set_facecolor(BG)
ax1.axis('off')
ax1.set_title('기존 논문과 동일한 점', color=WHITE, fontsize=18, fontweight='bold', pad=20)

same_cols = ['항목', '설명']
same_rows = [
    ['데이터 소스',  'PhysioNet CinC 2016 training set 동일 사용'],
    ['평가 지표',    'MACC (Mean of Sensitivity & Specificity) 동일 채택'],
    ['특징 추출',    'Mel-Spectrogram 기반 (주파수 도메인 분석)'],
    ['목표',         '심음의 정상/비정상 자동 판별'],
]

table1 = ax1.table(cellText=same_rows, colLabels=same_cols, loc='center', cellLoc='center')
table1.auto_set_font_size(False)
table1.set_fontsize(13)
table1.scale(1.0, 2.2)

for (row, col), cell in table1.get_celld().items():
    cell.set_edgecolor('#334155')
    cell.set_linewidth(1.2)
    if row == 0:
        cell.set_facecolor('#2D3748')
        cell.set_text_props(color=CYAN, fontweight='bold', fontsize=14)
    else:
        cell.set_facecolor(CARD)
        if col == 0:
            cell.set_text_props(color=GREEN, fontweight='bold', fontsize=12)
        else:
            cell.set_text_props(color=WHITE, fontsize=12)

path1 = os.path.join(OUT_DIR, '11_same_points_table.png')
fig1.savefig(path1, dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig1)
print(f"[OK] {path1}")


# ═══════════════════════════════════════════
# Figure 2: 다른 점
# ═══════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(16, 6.5), facecolor=BG)
ax2.set_facecolor(BG)
ax2.axis('off')
ax2.set_title('기존 논문 vs 본 프로젝트 — 차이점', color=WHITE, fontsize=18, fontweight='bold', pad=20)

diff_cols = ['항목', '기존 논문 (CinC 2016)', '본 프로젝트 (V16)']
diff_rows = [
    ['분류 체계',    'Binary (2-Class)',                          '3-Class (Normal / Abnormal / Unknown)'],
    ['데이터 확장',  'CinC 2016만 사용',                          'CinC 2016 + 소아 + ESC-50 환경음\n+ ICS43434 기기 녹음'],
    ['세그멘테이션', 'Cardiac Cycle 검출\n(Springer 알고리즘)',    '고정 길이 슬라이딩 윈도우\n(5초 윈도우, 2초 스트라이드)'],
    ['모델',         '단일 모델 (SVM, RF 등)',                    '2-Stage 앙상블\n(ResNet+CBAM → XGBoost)'],
    ['Attention',    '없음',                                      'CBAM (Channel + Spatial)'],
    ['Unknown 처리', '없음 (비심음 → 오분류)',                    'ESC-50 환경음 2,000개로\nUnknown 클래스 학습'],
    ['실시간 추론',  '오프라인 분석',                              'BLE → MQTT → 실시간 서버 추론'],
]

table2 = ax2.table(cellText=diff_rows, colLabels=diff_cols, loc='center', cellLoc='center')
table2.auto_set_font_size(False)
table2.set_fontsize(11)
table2.scale(1.0, 2.4)

for (row, col), cell in table2.get_celld().items():
    cell.set_edgecolor('#334155')
    cell.set_linewidth(1.2)
    if row == 0:
        cell.set_facecolor('#2D3748')
        colors = [CYAN, ORANGE, CYAN]
        cell.set_text_props(color=colors[col], fontweight='bold', fontsize=13)
    else:
        cell.set_facecolor(CARD)
        if col == 0:
            cell.set_text_props(color=GREEN, fontweight='bold', fontsize=11)
        elif col == 1:
            cell.set_text_props(color='#94A3B8', fontsize=11)
        else:
            cell.set_text_props(color=WHITE, fontweight='bold', fontsize=11)

path2 = os.path.join(OUT_DIR, '12_diff_points_table.png')
fig2.savefig(path2, dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig2)
print(f"[OK] {path2}")

print("\n완료!")
