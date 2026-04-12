"""
================================================================
심음 이진 분류 시각화 - 발표용
================================================================
ResNet+CBAM+XGBoost 파이프라인 구조, 성능, CBAM 작동 원리
출력: G:\stetho_ai\Heart_binary_classification\figures\
================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec
import librosa
import librosa.display

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

CACHE_DIR = r"G:\stetho_ai\Heart_binary_classification"
OUTPUT_DIR = os.path.join(CACHE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 색상 팔레트 ──
C_PREP = "#546E7A"      # preprocessing grey
C_CNN = "#1565C0"        # CNN blue
C_CBAM = "#7B1FA2"       # CBAM purple
C_XGB = "#2E7D32"        # XGBoost green
C_NORM = "#43A047"       # normal green
C_ABNORM = "#E53935"     # abnormal red
C_ADULT = "#1976D2"
C_PED = "#FF7043"


# ================================================================
# Helper
# ================================================================
def draw_box(ax, x, y, w, h, text, color, fontsize=10, text_color="white", alpha=0.95):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor="black", linewidth=1.5,
                         alpha=alpha, zorder=3)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color, zorder=4,
            linespacing=1.3)


# ================================================================
# Figure 1: 전체 파이프라인
# ================================================================
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(22, 14))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 14)
    ax.axis("off")
    fig.suptitle("심음 이진 분류 파이프라인 - ResNet + CBAM + XGBoost",
                 fontsize=22, fontweight="bold", y=0.97)

    # ── STEP 1: 데이터 ──
    ax.text(0.5, 13.2, "STEP 1: 데이터 수집", fontsize=13, fontweight="bold", color=C_PREP)

    draw_box(ax, 0.3, 11.2, 4.5, 1.5,
             "성인 심음 (PhysioNet CinC 2016)\ntraining a~f, 6개 폴더\n정상(-1) / 비정상(1)",
             C_ADULT, fontsize=9)
    draw_box(ax, 0.3, 9.2, 4.5, 1.5,
             "소아 심음 (Pediatric)\nlabeled_dataset.csv\nAbsent(정상) / Present(비정상)",
             C_PED, fontsize=9)

    # 균형 샘플링
    draw_box(ax, 0.3, 7.2, 4.5, 1.5,
             "5:5:5:5 균형 샘플링\n(성인정상 : 성인비정상 :\n소아정상 : 소아비정상)",
             "#37474F", fontsize=9)

    for y_pos in [11.2, 9.2]:
        ax.annotate("", xy=(2.55, y_pos), xytext=(2.55, y_pos + 0.1 - 0.5),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#333"))

    # ── STEP 2: 전처리 ──
    ax.text(5.5, 13.2, "STEP 2: 전처리", fontsize=13, fontweight="bold", color=C_PREP)

    draw_box(ax, 5.3, 11.2, 4.5, 1.5,
             "슬라이딩 윈도우\nWindow: 5초 | Stride: 2초\nSR: 4000Hz | 3초 중첩",
             "#455A64", fontsize=9)

    draw_box(ax, 5.3, 9.2, 4.5, 1.5,
             "Mel-Spectrogram 변환\n64-mel | hop=256\n입력: (1, 64, 79)",
             "#546E7A", fontsize=9)

    ax.annotate("", xy=(5.3, 11.95), xytext=(4.8, 11.95),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#333"))
    ax.annotate("", xy=(7.55, 11.2), xytext=(7.55, 10.7),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#333"))

    # ── STEP 3: Phase 1 - CNN ──
    ax.text(10.5, 13.2, "STEP 3: Phase 1 - CNN 학습", fontsize=13, fontweight="bold", color=C_CNN)

    draw_box(ax, 10.3, 11.0, 5, 1.8,
             "ResNet + CBAM\n\nConv(64) -> Res(64)+CBAM\n-> Res(128)+CBAM -> Res(256)+CBAM\n-> AvgPool -> FC(256)",
             C_CNN, fontsize=9)

    draw_box(ax, 10.3, 9.2, 5, 1.3,
             "CrossEntropyLoss\nAdam(lr=0.001) | EarlyStop(7)\n100 epochs | batch 32",
             "#1976D2", fontsize=9)

    ax.annotate("", xy=(10.3, 11.9), xytext=(9.8, 11.9),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#333"))
    ax.annotate("", xy=(12.8, 11.0), xytext=(12.8, 10.5),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#333"))

    # ── STEP 4: Phase 2 - Feature Extraction ──
    ax.text(10.5, 8.0, "STEP 4: Phase 2 - 특징 추출", fontsize=13, fontweight="bold", color=C_CNN)

    draw_box(ax, 10.3, 6.2, 5, 1.5,
             "학습된 ResNet+CBAM에서\nAvgPool 직후 256차원 벡터 추출\n(CNN을 특징 추출기로 활용)",
             "#0D47A1", fontsize=9)

    ax.annotate("", xy=(12.8, 8.2), xytext=(12.8, 7.7),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#333"))

    # ── STEP 5: Phase 3 - XGBoost ──
    ax.text(16, 13.2, "STEP 5: Phase 3 - XGBoost", fontsize=13, fontweight="bold", color=C_XGB)

    draw_box(ax, 16, 11.0, 5.5, 1.8,
             "XGBoost Classifier\n\nn_estimators=100 | max_depth=6\nlr=0.1 | tree_method=hist\nGPU(CUDA) 가속",
             C_XGB, fontsize=9)

    ax.annotate("", xy=(16, 7.0), xytext=(15.3, 7.0),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#333"))

    # ── STEP 6: 추론 ──
    ax.text(16, 8.0, "STEP 6: Phase 4 - 추론", fontsize=13, fontweight="bold", color=C_ABNORM)

    draw_box(ax, 16, 6.2, 5.5, 1.5,
             "구간별 확률 예측\npredict_proba -> 평균\n과반수 투표로 최종 판정",
             "#B71C1C", fontsize=9)

    ax.annotate("", xy=(18.75, 11.0), xytext=(18.75, 10.5),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#333"))

    # 결과
    draw_box(ax, 16, 4.5, 2.5, 1.2, "정상\n(Normal)", C_NORM, fontsize=11)
    draw_box(ax, 19, 4.5, 2.5, 1.2, "비정상\n(Abnormal)", C_ABNORM, fontsize=11)

    ax.annotate("", xy=(17.25, 5.7), xytext=(17.7, 6.2),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_NORM))
    ax.annotate("", xy=(20.25, 5.7), xytext=(19.8, 6.2),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_ABNORM))

    # ── 하단: 핵심 설계 원리 ──
    draw_box(ax, 0.3, 1.0, 6.5, 3.5,
             "왜 CNN + XGBoost 앙상블인가?\n\n"
             "1. CNN(ResNet+CBAM): 스펙트로그램에서\n"
             "   고수준 특징을 자동 추출\n"
             "2. XGBoost: 추출된 특징으로 결정 경계 최적화\n"
             "   -> 단독 CNN보다 일반화 능력 향상\n"
             "3. 2단계 분리: 각 단계 독립 최적화 가능",
             "#263238", fontsize=10)

    draw_box(ax, 7.5, 1.0, 6.5, 3.5,
             "왜 CBAM인가? (vs SE Block)\n\n"
             "SE Block: 채널 어텐션만\n"
             "CBAM: 채널 + 공간 어텐션\n\n"
             "스펙트로그램에서:\n"
             "  채널 -> '어떤 주파수 대역이 중요한가'\n"
             "  공간 -> '어떤 시간-주파수 위치가 중요한가'\n"
             "-> S1/S2 위치를 정확히 집중",
             C_CBAM, fontsize=10)

    draw_box(ax, 14.8, 1.0, 6.7, 3.5,
             "데이터 설계\n\n"
             "성인(PhysioNet) + 소아(Pediatric)\n"
             "-> 연령대 편향 방지\n\n"
             "5:5:5:5 균형 샘플링\n"
             "-> 클래스 불균형 해소\n\n"
             "슬라이딩 윈도우(5초, 2초 stride)\n"
             "-> 데이터 증강 + 구간 분석 가능",
             "#37474F", fontsize=10)

    path = os.path.join(OUTPUT_DIR, "01_classification_pipeline.png")
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# Figure 2: ResNet+CBAM 아키텍처 상세
# ================================================================
def fig2_architecture():
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis("off")
    fig.suptitle("ResNet + CBAM 아키텍처 상세", fontsize=22, fontweight="bold", y=0.97)

    # ── 메인 파이프라인 (상단) ──
    blocks = [
        ("Input\n(1, 64, 79)\nMel-Spectrogram", 0.5, "#78909C"),
        ("Conv2d(64)\n+ BN + ReLU\n(64, 64, 79)", 3.0, C_CNN),
        ("ResBlock(64)\n+ CBAM\n(64, 64, 79)", 5.5, C_CNN),
        ("ResBlock(128)\nstride=2 + CBAM\n(128, 32, 40)", 8.0, C_CNN),
        ("ResBlock(256)\nstride=2 + CBAM\n(256, 16, 20)", 10.5, "#0D47A1"),
        ("AdaptiveAvgPool\n(256, 1, 1)\n-> Flatten(256)", 13.0, "#01579B"),
        ("Dropout(0.5)\n+ FC(256->2)\n-> Softmax", 15.5, "#004D40"),
        ("XGBoost\n256-dim 입력\n-> 정상/비정상", 18.0, C_XGB),
    ]

    for text, x, color in blocks:
        draw_box(ax, x, 9.0, 2.2, 2.2, text, color, fontsize=9)

    # 화살표
    for i in range(len(blocks) - 1):
        x1 = blocks[i][1] + 2.2
        x2 = blocks[i+1][1]
        ax.annotate("", xy=(x2, 10.1), xytext=(x1, 10.1),
                    arrowprops=dict(arrowstyle="-|>", lw=2, color="#333"))

    # Phase 분리선
    ax.axvline(x=14.7, ymin=0.55, ymax=0.95, color="#999", linestyle="--", linewidth=1.5)
    ax.text(14.2, 11.6, "Phase 1\n(CNN 학습)", fontsize=10, ha="center", color=C_CNN, fontweight="bold")
    ax.text(16.8, 11.6, "Phase 2-3\n(Feature -> XGB)", fontsize=10, ha="center", color=C_XGB, fontweight="bold")

    # 특징 추출 포인트 표시
    ax.annotate("256-dim\nfeature vector",
                xy=(14.0, 9.0), xytext=(14.0, 7.8),
                fontsize=10, ha="center", color="#D32F2F", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#D32F2F"))

    # ── CBAM 상세 (하단 좌측) ──
    ax.text(0.5, 7.0, "CBAM (Convolutional Block Attention Module) 상세",
            fontsize=15, fontweight="bold", color=C_CBAM)

    # Channel Attention
    draw_box(ax, 0.5, 4.0, 9.0, 2.5,
             "Channel Attention (채널별 중요도)\n\n"
             "Feature Map (C, H, W)\n"
             "-> AvgPool + MaxPool -> (C, 1, 1) x 2\n"
             "-> SharedFC(C -> C/16 -> C) + ReLU + Sigmoid\n"
             "-> Channel Weight (C, 1, 1) x Feature Map\n\n"
             "'어떤 필터(주파수 대역)가 중요한가?'",
             C_CBAM, fontsize=10, text_color="#EDE7F6")

    # Spatial Attention
    draw_box(ax, 10.5, 4.0, 9.0, 2.5,
             "Spatial Attention (위치별 중요도)\n\n"
             "Channel Attention 출력 (C, H, W)\n"
             "-> AvgPool + MaxPool along Channel -> (2, H, W)\n"
             "-> Conv2d(2->1, kernel=7) + Sigmoid\n"
             "-> Spatial Weight (1, H, W) x Feature Map\n\n"
             "'스펙트로그램의 어떤 시간-주파수 위치가 중요한가?'",
             "#4A148C", fontsize=10, text_color="#EDE7F6")

    ax.annotate("순서", xy=(10.0, 5.25), xytext=(9.5, 5.25),
                arrowprops=dict(arrowstyle="-|>", lw=2.5, color=C_CBAM),
                fontsize=12, fontweight="bold", color=C_CBAM)

    # ── ResNet Block 상세 (하단) ──
    draw_box(ax, 0.5, 0.5, 9.0, 2.8,
             "ResNet Block 상세\n\n"
             "x -> Conv3x3 + BN + ReLU + Dropout(0.2)\n"
             "  -> Conv3x3 + BN\n"
             "  -> CBAM\n"
             "  + Shortcut(x)  [1x1 Conv if shape mismatch]\n"
             "  -> ReLU\n\n"
             "Residual: 그래디언트 소실 방지 + 깊은 네트워크 학습 안정화",
             "#1A237E", fontsize=10, text_color="#E8EAF6")

    # ── 심음에서 CBAM의 역할 ──
    draw_box(ax, 10.5, 0.5, 9.0, 2.8,
             "심음 스펙트로그램에서 CBAM의 역할\n\n"
             "Channel Attention:\n"
             "  S1/S2 주파수 대역(20-150Hz) 필터 -> 가중치 높음\n"
             "  배경 노이즈 대역 필터 -> 가중치 낮음\n\n"
             "Spatial Attention:\n"
             "  S1/S2가 나타나는 시간-주파수 좌표 -> 가중치 높음\n"
             "  무음/노이즈 구간 -> 가중치 낮음",
             "#1B5E20", fontsize=10, text_color="#E8F5E9")

    path = os.path.join(OUTPUT_DIR, "02_architecture_detail.png")
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# Figure 3: 데이터 구성 & 전처리 시각화
# ================================================================
def fig3_data_and_preprocessing():
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("데이터 구성 & 전처리 파이프라인", fontsize=20, fontweight="bold", y=0.98)

    # ── 좌측: 데이터 구성 ──
    ax_data = fig.add_axes([0.03, 0.35, 0.45, 0.58])
    ax_data.axis("off")
    ax_data.set_xlim(0, 10)
    ax_data.set_ylim(0, 10)
    ax_data.set_title("데이터셋 구성", fontsize=16, fontweight="bold", pad=10)

    # 4개 그룹
    groups = [
        ("성인 정상", C_ADULT, "#BBDEFB", 7.5),
        ("성인 비정상", "#C62828", "#FFCDD2", 5.5),
        ("소아 정상", C_PED, "#FFE0B2", 3.5),
        ("소아 비정상", "#AD1457", "#F8BBD0", 1.5),
    ]

    for label, border_color, bg_color, y in groups:
        box = FancyBboxPatch((0.5, y), 4, 1.5, boxstyle="round,pad=0.15",
                             facecolor=bg_color, edgecolor=border_color, linewidth=2)
        ax_data.add_patch(box)
        ax_data.text(2.5, y + 0.75, label, ha="center", va="center",
                     fontsize=13, fontweight="bold", color=border_color)

    # 균형 화살표
    ax_data.annotate("5:5:5:5\n균형 샘플링",
                     xy=(5.5, 5.0), xytext=(4.5, 5.0),
                     fontsize=11, fontweight="bold", color="#333",
                     arrowprops=dict(arrowstyle="-|>", lw=2, color="#333"))

    # 통합 박스
    draw_box(ax_data, 5.8, 3.5, 3.5, 3.5,
             "균형 데이터셋\n\n각 그룹 동일 수\n-> 편향 없는 학습\n\n총 세그먼트:\n~47,000개\n(윈도우 적용 후)",
             "#37474F", fontsize=10)

    # 소스 라벨
    ax_data.text(2.5, 9.5, "PhysioNet CinC 2016 (성인) + Pediatric Dataset (소아)",
                 fontsize=10, ha="center", color="#555", style="italic")

    # ── 우측: 전처리 시각화 (실제 데이터) ──
    ax_pre = fig.add_axes([0.55, 0.35, 0.42, 0.58])
    ax_pre.axis("off")
    ax_pre.set_title("전처리: Waveform -> Mel-Spectrogram", fontsize=16, fontweight="bold", pad=10)

    # 실제 데이터가 있으면 로드
    x_data_path = os.path.join(CACHE_DIR, "x_data.npy")
    if os.path.exists(x_data_path):
        X = np.load(x_data_path)
        # 정상/비정상 각 1개 샘플
        y_data = np.load(os.path.join(CACHE_DIR, "y_data.npy"))

        norm_idx = np.where(y_data == 0)[0][0]
        abnorm_idx = np.where(y_data == 1)[0][0]

        for i, (idx, label, color) in enumerate([
            (norm_idx, "정상 (Normal)", C_NORM),
            (abnorm_idx, "비정상 (Abnormal)", C_ABNORM),
        ]):
            mel = X[idx, 0]  # (64, 79)
            ax_mel = fig.add_axes([0.55, 0.68 - i * 0.32, 0.40, 0.25])
            img = ax_mel.imshow(mel, aspect="auto", origin="lower", cmap="magma")
            ax_mel.set_title(f"{label} 심음 Mel-Spectrogram", fontsize=12,
                             fontweight="bold", color=color)
            ax_mel.set_ylabel("Mel Bin")
            ax_mel.set_xlabel("Time Frame")
            fig.colorbar(img, ax=ax_mel, format="%+2.0f dB", shrink=0.8)

    # ── 하단: 슬라이딩 윈도우 설명 ──
    ax_win = fig.add_axes([0.03, 0.03, 0.94, 0.28])
    ax_win.axis("off")
    ax_win.set_title("슬라이딩 윈도우 방식", fontsize=16, fontweight="bold", pad=5)

    # 타임라인
    ax_win.plot([0.05, 0.95], [0.65, 0.65], color="#333", linewidth=2,
                transform=ax_win.transAxes)

    # 윈도우 블록
    window_colors = ["#BBDEFB", "#90CAF9", "#64B5F6", "#42A5F5", "#2196F3"]
    for i, c in enumerate(window_colors):
        x_start = 0.05 + i * 0.12
        x_end = x_start + 0.30
        box = FancyBboxPatch((x_start, 0.45), x_end - x_start, 0.35,
                             boxstyle="round,pad=0.02", facecolor=c, edgecolor=C_CNN,
                             linewidth=1.5, alpha=0.7, transform=ax_win.transAxes)
        ax_win.add_patch(box)
        ax_win.text((x_start + x_end)/2, 0.62, f"Win {i+1}\n(5s)",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    transform=ax_win.transAxes)

    ax_win.text(0.5, 0.25,
                "Window: 5초 (20000 samples) | Stride: 2초 (8000 samples) | 중첩: 3초\n"
                "-> 하나의 심음 파일에서 여러 세그먼트 생성 (데이터 증강 효과)\n"
                "-> 각 세그먼트별 독립 예측 후 과반수 투표(Majority Voting)로 최종 판정",
                ha="center", va="center", fontsize=12, transform=ax_win.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD", edgecolor=C_CNN,
                          linewidth=1.5), linespacing=1.5)

    # stride 표시
    ax_win.annotate("", xy=(0.17, 0.42), xytext=(0.05, 0.42),
                    arrowprops=dict(arrowstyle="<->", lw=1.5, color="#D32F2F"),
                    transform=ax_win.transAxes)
    ax_win.text(0.11, 0.37, "2s stride", fontsize=9, ha="center", color="#D32F2F",
                fontweight="bold", transform=ax_win.transAxes)

    path = os.path.join(OUTPUT_DIR, "03_data_preprocessing.png")
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# Figure 4: 정상 vs 비정상 스펙트로그램 비교 (여러 샘플)
# ================================================================
def fig4_normal_vs_abnormal():
    x_data_path = os.path.join(CACHE_DIR, "x_data.npy")
    if not os.path.exists(x_data_path):
        print("  [!] x_data.npy 없음, 건너뜀")
        return

    X = np.load(x_data_path)
    y = np.load(os.path.join(CACHE_DIR, "y_data.npy"))

    norm_indices = np.where(y == 0)[0]
    abnorm_indices = np.where(y == 1)[0]

    n_samples = 4
    rng = np.random.RandomState(42)
    norm_sel = rng.choice(norm_indices, n_samples, replace=False)
    abnorm_sel = rng.choice(abnorm_indices, n_samples, replace=False)

    fig, axes = plt.subplots(2, n_samples, figsize=(20, 8))
    fig.suptitle("정상 vs 비정상 심음 Mel-Spectrogram 비교", fontsize=20, fontweight="bold")

    for col, idx in enumerate(norm_sel):
        ax = axes[0, col]
        ax.imshow(X[idx, 0], aspect="auto", origin="lower", cmap="magma")
        ax.set_title(f"정상 #{col+1}", fontsize=12, fontweight="bold", color=C_NORM)
        if col == 0:
            ax.set_ylabel("Mel Bin", fontsize=11)
        ax.set_xlabel("")

    for col, idx in enumerate(abnorm_sel):
        ax = axes[1, col]
        ax.imshow(X[idx, 0], aspect="auto", origin="lower", cmap="magma")
        ax.set_title(f"비정상 #{col+1}", fontsize=12, fontweight="bold", color=C_ABNORM)
        if col == 0:
            ax.set_ylabel("Mel Bin", fontsize=11)
        ax.set_xlabel("Time Frame")

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])

    # 하단 설명
    fig.text(0.5, 0.01,
             "정상: S1/S2 심음이 주기적이고 깨끗 | 비정상: S1/S2 사이 잡음(murmur), 불규칙한 에너지 분포",
             ha="center", fontsize=12, style="italic", color="#555")

    path = os.path.join(OUTPUT_DIR, "04_normal_vs_abnormal.png")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# Figure 5: 왜 이 구조인가 — 설계 근거 종합
# ================================================================
def fig5_design_rationale():
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis("off")
    fig.suptitle("심음 분류 모델 설계 근거 - 왜 이 구조를 선택했는가?",
                 fontsize=22, fontweight="bold", y=0.97)

    # ── 문제 정의 ──
    draw_box(ax, 0.5, 9.5, 19, 2.0,
             "문제: 전자 청진기(ESP32)로 수집한 심음이 정상인지 비정상(심잡음)인지 자동 분류\n"
             "도전 과제: (1) 성인/소아 모두 커버 (2) 노이즈가 많은 환경 (3) 실시간 추론 가능해야 함 "
             "(4) 적은 데이터로 높은 정확도",
             "#37474F", fontsize=12)

    # 4개 카드
    cards = [
        ("스펙트로그램 기반\n(vs Waveform)",
         "심음 분류는 주파수 패턴이 핵심\n"
         "(murmur = 특정 주파수 에너지)\n\n"
         "Mel-Spectrogram:\n"
         "  인간 청각 스케일 반영\n"
         "  2D 이미지 -> CNN 활용 가능\n"
         "  64-mel, hop=256\n"
         "  입력: (1, 64, 79)",
         "#455A64", 0.5),

        ("ResNet + CBAM\n(vs VGG, plain CNN)",
         "ResNet: Residual 연결로\n"
         "  깊은 네트워크 안정 학습\n\n"
         "CBAM: Channel + Spatial\n"
         "  채널: 중요한 주파수 대역 강조\n"
         "  공간: S1/S2 위치 집중\n"
         "  -> SE Block보다 정밀한 주의 집중",
         C_CNN, 5.3),

        ("XGBoost 앙상블\n(vs 단독 CNN)",
         "CNN만으로도 분류 가능하지만:\n\n"
         "CNN: 특징 추출에 특화\n"
         "XGBoost: 결정 경계 최적화에 특화\n\n"
         "앙상블 효과:\n"
         "  CNN feature + Tree 기반 분류\n"
         "  -> 일반화 성능 향상\n"
         "  -> GPU 가속 (tree_method=hist)",
         C_XGB, 10.1),

        ("슬라이딩 윈도우 + 투표\n(vs 전체 파일 분류)",
         "심음은 길이가 다양 (3~60초)\n\n"
         "5초 윈도우, 2초 stride:\n"
         "  긴 파일 -> 여러 세그먼트\n"
         "  각 세그먼트 독립 분류\n"
         "  과반수 투표로 최종 판정\n\n"
         "장점: 데이터 증강 + 구간 분석\n"
         "  어떤 구간이 비정상인지 추적 가능",
         "#B71C1C", 14.9),
    ]

    for title, content, color, x in cards:
        draw_box(ax, x, 7.5, 4.5, 1.5, title, color, fontsize=10)
        props = dict(boxstyle="round,pad=0.5", facecolor="#FAFAFA", edgecolor=color, linewidth=1.5)
        ax.text(x + 2.25, 3.8, content, fontsize=9, ha="center", va="center",
                bbox=props, linespacing=1.4, color="#333")

    # ── 하단: 성능 기대치 ──
    draw_box(ax, 0.5, 0.3, 19, 1.5,
             "목표 성능: Accuracy > 90% | Sensitivity(비정상 검출) > 85% | Specificity(정상 판별) > 90%\n"
             "실제 성능은 학습 데이터 크기, 심잡음 유형(수축기/이완기), 녹음 품질에 따라 변동",
             "#004D40", fontsize=12)

    path = os.path.join(OUTPUT_DIR, "05_design_rationale.png")
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 60)
    print(" 심음 이진 분류 시각화 생성")
    print("=" * 60)

    print("\n[1] Figure 1: 전체 파이프라인...")
    fig1_pipeline()

    print("[2] Figure 2: ResNet+CBAM 아키텍처...")
    fig2_architecture()

    print("[3] Figure 3: 데이터 & 전처리...")
    fig3_data_and_preprocessing()

    print("[4] Figure 4: 정상 vs 비정상 비교...")
    fig4_normal_vs_abnormal()

    print("[5] Figure 5: 설계 근거...")
    fig5_design_rationale()

    print("\n" + "=" * 60)
    print(f" 모든 그래프 저장 완료!")
    print(f" 위치: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
