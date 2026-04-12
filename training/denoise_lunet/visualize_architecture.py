"""
================================================================
디노이즈 아키텍처 & LSTM→TCN 전환 시각화 — 발표용
================================================================
Figure 7: 디노이즈 방식 소개 (U-Net 파이프라인 + 각 기법 설명)
Figure 8: LSTM vs TCN 구조 비교 & 왜 TCN을 시도했는가
Figure 9: Direct vs Residual Learning 비교
================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = r"G:\stetho_ai\ShittyDenoise\LU-NET\figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# Figure 7: 디노이즈 파이프라인 전체 소개
# ================================================================
def fig7_denoise_pipeline():
    fig, ax = plt.subplots(figsize=(22, 14))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 14)
    ax.axis("off")
    fig.suptitle("심음 디노이즈 파이프라인 — U-Net 기반 Waveform Denoising",
                 fontsize=22, fontweight="bold", y=0.97)

    # ── 색상 팔레트 ──
    C_ENC = "#1565C0"       # encoder blue
    C_DEC = "#E65100"       # decoder orange
    C_SKIP = "#2E7D32"      # skip green
    C_ATTN = "#7B1FA2"      # attention purple
    C_RES = "#C62828"       # residual red
    C_LOSS = "#F57F17"      # loss yellow
    C_BG = "#FAFAFA"

    # ── 제목 박스 ──
    def draw_box(x, y, w, h, text, color, fontsize=10, alpha=0.9, text_color="white"):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor="black", linewidth=1.5,
                             alpha=alpha, zorder=3)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)

    # ── STEP 1: 입력 ──
    ax.text(1, 13.2, "STEP 1", fontsize=12, fontweight="bold", color="#555")
    ax.text(1, 12.8, "입력: 노이즈 섞인 심음 (Raw Waveform)", fontsize=11, color="#333")

    draw_box(1, 11.5, 4.5, 1.0, "노이즈 심음\n(20000 samples, 4kHz, 5초)", "#78909C", fontsize=10)
    ax.annotate("", xy=(5.8, 12.0), xytext=(5.5, 12.0),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#333"))

    # ── STEP 2: Encoder ──
    ax.text(6.2, 13.2, "STEP 2: Encoder (특징 추출)", fontsize=12, fontweight="bold", color=C_ENC)

    enc_labels = [
        ("Enc1\nConv1D×2\n(20000, 32)", 0),
        ("Enc2\nstride=2\n(10000, 64)", 1),
        ("Enc3\nstride=2\n(5000, 128)", 2),
        ("Enc4\nstride=2\n(2500, 256)", 3),
        ("Enc5\nBottleneck\n(1250, 512)", 4),
    ]

    enc_x = 6.3
    for label, i in enc_labels:
        y = 11.5 - i * 1.8
        w = 2.5 - i * 0.15
        draw_box(enc_x, y, w, 1.2, label, C_ENC, fontsize=8)
        if i < 4:
            ax.annotate("", xy=(enc_x + w/2, y), xytext=(enc_x + w/2, y + 0.1 - 0.3),
                        arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_ENC))

    # ── STEP 3: Skip Connections ──
    ax.text(10, 13.2, "STEP 3: Skip Connection", fontsize=12, fontweight="bold", color=C_SKIP)
    ax.text(10, 12.8, "(시간 문맥 학습)", fontsize=10, color=C_SKIP)

    skip_labels = [
        "Bi-LSTM(32)\n→ (20000, 64)",
        "Bi-LSTM(32)\n→ (10000, 64)",
        "Bi-LSTM(64)\n→ (5000, 128)",
        "Bi-LSTM(64)\n→ (2500, 128)",
    ]

    for i, label in enumerate(skip_labels):
        y = 11.5 - i * 1.8
        draw_box(10.2, y, 2.3, 1.2, label, C_SKIP, fontsize=8)
        # encoder → skip 화살표
        enc_w = 2.5 - i * 0.15
        ax.annotate("", xy=(10.2, y + 0.6), xytext=(enc_x + enc_w, y + 0.6),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color=C_SKIP,
                                    connectionstyle="arc3,rad=0"))

    # ── Attention 표시 (v2에만) ──
    for i in range(4):
        y = 11.5 - i * 1.8
        draw_box(12.7, y + 0.15, 1.3, 0.9,
                 "SE Block\n(채널 가중치)", C_ATTN, fontsize=7, alpha=0.85)

    # ── STEP 4: Decoder ──
    ax.text(14.5, 13.2, "STEP 4: Decoder (복원)", fontsize=12, fontweight="bold", color=C_DEC)

    dec_labels = [
        ("Dec1\nUp+Concat\n(20000, 32)", 0),
        ("Dec2\nUp+Concat\n(10000, 64)", 1),
        ("Dec3\nUp+Concat\n(5000, 128)", 2),
        ("Dec4\nUp+Concat\n(2500, 256)", 3),
    ]

    dec_x = 14.5
    for label, i in dec_labels:
        y = 11.5 - i * 1.8
        w = 2.5 - i * 0.15
        draw_box(dec_x, y, w, 1.2, label, C_DEC, fontsize=8)
        # skip → decoder 화살표
        ax.annotate("", xy=(dec_x, y + 0.6), xytext=(14.0, y + 0.6),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color=C_SKIP))
        if i < 3:
            ax.annotate("", xy=(dec_x + w/2, y + 1.2 + 0.3), xytext=(dec_x + w/2, y + 1.2),
                        arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_DEC))

    # Bottleneck → Dec4 화살표
    ax.annotate("", xy=(dec_x + 1.0, 4.3 + 1.2), xytext=(enc_x + 1.0, 4.3 + 0.6),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#333",
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(11.5, 4.0, "Bottleneck\n(가장 압축된 표현)", fontsize=9, ha="center",
            color="#555", style="italic")

    # ── STEP 5: 출력 ──
    ax.text(17.5, 13.2, "STEP 5: 출력", fontsize=12, fontweight="bold", color=C_RES)

    draw_box(17.5, 10.0, 4, 2.5,
             "Residual Learning\n\n모델 출력 = 노이즈 추정치\n\n깨끗한 심음 =\n입력 - 노이즈 추정치",
             C_RES, fontsize=10)

    # 최종 출력
    draw_box(17.5, 8.0, 4, 1.2, "깨끗한 심음 출력\n(20000 samples)", "#00695C", fontsize=10)
    ax.annotate("", xy=(19.5, 9.2), xytext=(19.5, 10.0),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#333"))

    # ── 하단: 손실 함수 설명 ──
    loss_y = 1.0
    draw_box(1, loss_y, 5.5, 2.5,
             "Loss Function\n\n"
             "Combined = 0.9 × MSE + 0.1 × SI-SNR\n\n"
             "MSE: 빠른 수렴, 안정적 학습\n"
             "SI-SNR: 파형 형태 보존 최적화",
             C_LOSS, fontsize=10, text_color="black")

    # ── 하단: 핵심 설계 원리 ──
    draw_box(7.5, loss_y, 6.0, 2.5,
             "핵심 설계 원리\n\n"
             "① 다중 스케일 특징 추출 (5-level U-Net)\n"
             "② 시간 문맥 학습 (Bi-LSTM or TCN)\n"
             "③ 채널 선택적 강조 (SE Attention)\n"
             "④ 노이즈만 학습 (Residual Learning)",
             "#37474F", fontsize=10)

    # ── 하단: vs Gemini 방식 ──
    draw_box(14.5, loss_y, 7.0, 2.5,
             "vs 이전 방식 (Gemini CRNN)\n\n"
             "이전: Waveform → 스펙트로그램 → 마스킹 → 역변환\n"
             "     (변환 과정에서 위상 정보 손실)\n\n"
             "현재: Waveform → Waveform (End-to-End)\n"
             "     (위상 보존, 파형 직접 복원)",
             "#263238", fontsize=10)

    path = os.path.join(OUTPUT_DIR, "07_denoise_pipeline.png")
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# Figure 8: LSTM vs TCN — 왜 TCN을 시도했고 결과는 어땠는가
# ================================================================
def fig8_lstm_vs_tcn():
    fig = plt.figure(figsize=(22, 16))
    fig.suptitle("LSTM vs TCN: 왜 전환을 시도했고, 결과는 어땠는가?",
                 fontsize=22, fontweight="bold", y=0.98)

    # ── 색상 ──
    C_LSTM = "#1565C0"
    C_TCN = "#E65100"

    # ═══════════════════════════════════════
    # 상단: 구조 다이어그램
    # ═══════════════════════════════════════
    ax_lstm = fig.add_axes([0.03, 0.52, 0.45, 0.42])
    ax_tcn = fig.add_axes([0.52, 0.52, 0.45, 0.42])

    # ── LSTM 다이어그램 ──
    ax_lstm.set_xlim(0, 10)
    ax_lstm.set_ylim(0, 10)
    ax_lstm.axis("off")
    ax_lstm.set_title("Bi-LSTM Skip Connection", fontsize=16, fontweight="bold",
                      color=C_LSTM, pad=10)

    # 시간 스텝 표현
    for i in range(7):
        x = 1.2 + i * 1.1
        # Forward LSTM cell
        box = FancyBboxPatch((x, 5.5), 0.8, 0.8, boxstyle="round,pad=0.08",
                             facecolor="#BBDEFB", edgecolor=C_LSTM, linewidth=1.5)
        ax_lstm.add_patch(box)
        ax_lstm.text(x + 0.4, 5.9, "h→", ha="center", va="center", fontsize=8,
                     color=C_LSTM, fontweight="bold")

        # Backward LSTM cell
        box = FancyBboxPatch((x, 4.2), 0.8, 0.8, boxstyle="round,pad=0.08",
                             facecolor="#FFE0B2", edgecolor=C_TCN, linewidth=1.5)
        ax_lstm.add_patch(box)
        ax_lstm.text(x + 0.4, 4.6, "←h", ha="center", va="center", fontsize=8,
                     color=C_TCN, fontweight="bold")

        # Forward 화살표
        if i < 6:
            ax_lstm.annotate("", xy=(x + 1.1, 5.9), xytext=(x + 0.8, 5.9),
                            arrowprops=dict(arrowstyle="-|>", lw=1.2, color=C_LSTM))
            ax_lstm.annotate("", xy=(x, 4.6), xytext=(x + 0.3, 4.6),
                            arrowprops=dict(arrowstyle="-|>", lw=1.2, color=C_TCN))

        # 입력 화살표
        ax_lstm.annotate("", xy=(x + 0.4, 5.5), xytext=(x + 0.4, 5.0),
                        arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#666"))

        # 시간 라벨
        ax_lstm.text(x + 0.4, 3.5, f"t{i+1}", ha="center", fontsize=9, color="#555")

    # concat 표시
    ax_lstm.text(5, 7.0, "Forward →", fontsize=11, ha="center", color=C_LSTM, fontweight="bold")
    ax_lstm.text(5, 3.0, "← Backward", fontsize=11, ha="center", color=C_TCN, fontweight="bold")
    ax_lstm.text(5, 2.2, "출력 = [h→ ; ←h]  (양방향 concat)", fontsize=11, ha="center",
                 color="#333", fontweight="bold")

    # 특징 박스
    props = dict(boxstyle="round,pad=0.6", facecolor="#E3F2FD", edgecolor=C_LSTM, linewidth=1.5)
    lstm_features = (
        "특징:\n"
        "• 순차적 처리 (t1→t2→...→tn)\n"
        "• Hidden state로 장기 문맥 기억\n"
        "• 양방향 → 과거+미래 모두 참조\n"
        "• 심박 주기(~0.8초=3200 samples) 포착"
    )
    ax_lstm.text(5, 8.8, lstm_features, ha="center", va="center", fontsize=10,
                 bbox=props, linespacing=1.5)

    # ── TCN 다이어그램 ──
    ax_tcn.set_xlim(0, 10)
    ax_tcn.set_ylim(0, 10)
    ax_tcn.axis("off")
    ax_tcn.set_title("TCN (Temporal Convolutional Network)", fontsize=16,
                     fontweight="bold", color=C_TCN, pad=10)

    # Dilated Conv 시각화
    layers_info = [
        ("d=1", 1, "#FFE0B2"),
        ("d=2", 2, "#FFCC80"),
        ("d=4", 4, "#FFB74D"),
        ("d=8", 8, "#FFA726"),
    ]

    for li, (label, dilation, color) in enumerate(layers_info):
        y = 3.5 + li * 1.5
        # 8개 노드
        for ni in range(8):
            x = 1.2 + ni * 1.0
            circle = plt.Circle((x + 0.3, y + 0.3), 0.25, facecolor=color,
                               edgecolor=C_TCN, linewidth=1.0)
            ax_tcn.add_patch(circle)

        ax_tcn.text(0.3, y + 0.3, label, fontsize=10, ha="center", va="center",
                    color=C_TCN, fontweight="bold")

        # 연결선 (dilation 패턴)
        if li < 3:
            next_y = y + 1.5
            for ni in range(8):
                x_to = 1.5 + ni * 1.0
                # 현재 노드에서 dilation 간격의 이전 노드로 연결
                src_idx = ni - layers_info[li+1][1]
                if 0 <= src_idx < 8:
                    x_from = 1.5 + src_idx * 1.0
                    ax_tcn.plot([x_from, x_to], [y + 0.55, next_y + 0.05],
                               color=C_TCN, linewidth=0.6, alpha=0.5)

    # Receptive field 표시
    ax_tcn.annotate("", xy=(1.5, 3.2), xytext=(8.5, 3.2),
                    arrowprops=dict(arrowstyle="<->", lw=2, color="red"))
    ax_tcn.text(5, 2.7, f"Receptive Field = (kernel-1) × Σdilations\n= 2 × (1+2+4+8+16+32) = 126 samples",
                ha="center", fontsize=10, color="red", fontweight="bold")

    # 특징 박스
    props = dict(boxstyle="round,pad=0.6", facecolor="#FFF3E0", edgecolor=C_TCN, linewidth=1.5)
    tcn_features = (
        "특징:\n"
        "• 병렬 처리 (모든 시점 동시 계산)\n"
        "• Dilated Conv로 넓은 receptive field\n"
        "• Causal: 미래 정보 사용 안 함\n"
        "• 학습 2~3배 빠름 (GPU 병렬화)"
    )
    ax_tcn.text(5, 8.8, tcn_features, ha="center", va="center", fontsize=10,
                bbox=props, linespacing=1.5)

    # ═══════════════════════════════════════
    # 중단: 전환 동기
    # ═══════════════════════════════════════
    ax_reason = fig.add_axes([0.03, 0.28, 0.94, 0.22])
    ax_reason.axis("off")

    ax_reason.text(0.5, 0.95, "왜 LSTM에서 TCN으로 전환을 시도했는가?",
                   fontsize=18, fontweight="bold", ha="center", va="top",
                   transform=ax_reason.transAxes)

    reasons = [
        ("① 속도 문제", "LSTM은 순차 처리 → 에포크당 10~12분\nTCN은 병렬 Conv → 에포크당 3~5분 (2~3배 빠름)",
         "#E3F2FD", C_LSTM),
        ("② 학술적 동향", "Conv-TasNet(2019), DCCRN(2020) 등\n최신 음성 분리 모델들이 TCN 채택\n→ 심음에도 적용 가능성 검증",
         "#FFF3E0", C_TCN),
        ("③ 공정한 비교", "동일 Encoder/Decoder에서\nSkip만 LSTM↔TCN 교체\n→ 순수 구조 효과 측정 (Ablation)",
         "#E8F5E9", "#2E7D32"),
    ]

    for i, (title, desc, bg, color) in enumerate(reasons):
        x = 0.05 + i * 0.33
        props = dict(boxstyle="round,pad=0.8", facecolor=bg, edgecolor=color, linewidth=2)
        ax_reason.text(x + 0.15, 0.4, f"{title}\n\n{desc}",
                       fontsize=11, va="center", ha="center",
                       transform=ax_reason.transAxes, bbox=props, linespacing=1.4)

    # ═══════════════════════════════════════
    # 하단: 결과 비교표
    # ═══════════════════════════════════════
    ax_result = fig.add_axes([0.03, 0.02, 0.94, 0.25])
    ax_result.axis("off")

    ax_result.text(0.5, 0.95, "실험 결과: LSTM vs TCN 직접 비교",
                   fontsize=18, fontweight="bold", ha="center", va="top",
                   transform=ax_result.transAxes)

    # 비교 표
    table_data = [
        ["", "LSTM (LU-Net)", "TCN (TU-Net)", "차이"],
        ["v1 (MSE only)", "ΔSNR +5.94 / SI-SNR 9.82", "ΔSNR +4.35 / SI-SNR 8.14", "LSTM +1.59 / +1.68"],
        ["v2 (Attn+Res+SI-SNR)", "ΔSNR +5.59 / SI-SNR 12.76", "ΔSNR +5.63 / SI-SNR 12.59", "거의 동등 (차이 <0.2)"],
        ["학습 속도", "~10-12분/epoch", "~3-5분/epoch", "TCN 2~3배 빠름"],
        ["파라미터 수", "~3.5M", "~4.1M", "TCN이 약간 많음"],
    ]

    colors_table = [
        ["#ECEFF1", "#BBDEFB", "#FFE0B2", "#E0E0E0"],
        ["#F5F5F5", "#E3F2FD", "#FFF3E0", "#E8F5E9"],
        ["#F5F5F5", "#E3F2FD", "#FFF3E0", "#FFF9C4"],
        ["#F5F5F5", "#E3F2FD", "#FFF3E0", "#C8E6C9"],
        ["#F5F5F5", "#E3F2FD", "#FFF3E0", "#F5F5F5"],
    ]

    for row_i, row in enumerate(table_data):
        for col_i, cell in enumerate(row):
            x = 0.02 + col_i * 0.245
            y = 0.72 - row_i * 0.15
            w = 0.235
            h = 0.13
            props = dict(boxstyle="round,pad=0.3",
                        facecolor=colors_table[row_i][col_i],
                        edgecolor="#999", linewidth=0.8)
            fs = 9 if row_i == 0 else 10
            fw = "bold" if row_i == 0 else "normal"
            ax_result.text(x + w/2, y + h/2, cell,
                          fontsize=fs, fontweight=fw,
                          ha="center", va="center",
                          transform=ax_result.transAxes, bbox=props)

    # 결론 박스
    conclusion = (
        "결론: v1(MSE)에서는 LSTM이 확실히 우세. "
        "그러나 Attention+Residual+SI-SNR을 추가하면 TCN도 LSTM과 거의 동등한 성능 달성.\n"
        "→ Attention/Residual이 TCN의 약점(장기 문맥 부족)을 보완함. "
        "실시간 추론이 필요한 환경에서는 TCN(v2)이 더 실용적."
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="#FFECB3", edgecolor="#FF8F00", linewidth=2)
    ax_result.text(0.5, 0.02, conclusion, fontsize=11, ha="center", va="bottom",
                   transform=ax_result.transAxes, bbox=props, linespacing=1.4)

    path = os.path.join(OUTPUT_DIR, "08_lstm_vs_tcn.png")
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# Figure 9: Direct vs Residual Learning + v1→v2 개선 흐름
# ================================================================
def fig9_residual_learning():
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle("Direct Learning vs Residual Learning — 디노이즈 전략 비교",
                 fontsize=20, fontweight="bold", y=0.97)

    # ── 좌측: Direct Learning (v1) ──
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("v1: Direct Learning\n(모델이 깨끗한 심음 전체를 복원)", fontsize=14,
                 fontweight="bold", color="#1565C0", pad=15)

    def draw_rbox(ax, x, y, w, h, text, color, fontsize=11, text_color="white"):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor="black", linewidth=1.5, zorder=3)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=4,
                linespacing=1.3)

    # Direct flow
    draw_rbox(ax, 1, 8, 3, 1.2, "입력: 노이즈 심음\nx(t) = s(t) + n(t)", "#78909C")
    ax.annotate("", xy=(4.5, 8.6), xytext=(4.0, 8.6),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#333"))

    draw_rbox(ax, 4.5, 7.8, 3.5, 1.6, "U-Net 모델\n\n목표: s(t) 전체를\n직접 출력", "#1565C0")
    ax.annotate("", xy=(5.5, 7.8), xytext=(5.5, 7.3),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#333"))

    draw_rbox(ax, 3.5, 5.8, 4, 1.2, "출력: s'(t)\n(깨끗한 심음 추정)", "#00695C")

    # 문제점
    props = dict(boxstyle="round,pad=0.6", facecolor="#FFEBEE", edgecolor="#C62828", linewidth=2)
    problems = (
        "문제점:\n\n"
        "• 모델이 심음 전체 파형을 재구성해야 함\n"
        "  → 복잡한 문제 (심음 자체가 복잡)\n\n"
        "• S1/S2 피크가 과도하게 smoothing될 수 있음\n\n"
        "• 노이즈가 작은 구간에서도 불필요한 변형 발생\n\n"
        "• Loss: MSE = Σ|s(t) - s'(t)|²"
    )
    ax.text(5, 2.5, problems, fontsize=10, ha="center", va="center",
            bbox=props, linespacing=1.4)

    # ── 우측: Residual Learning (v2) ──
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("v2: Residual Learning\n(모델이 노이즈만 추정, 입력에서 빼기)", fontsize=14,
                 fontweight="bold", color="#C62828", pad=15)

    # Residual flow
    draw_rbox(ax, 0.5, 8, 3, 1.2, "입력: 노이즈 심음\nx(t) = s(t) + n(t)", "#78909C")

    # 분기점
    ax.annotate("", xy=(3.8, 8.6), xytext=(3.5, 8.6),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#333"))
    ax.annotate("", xy=(8.5, 8.6), xytext=(3.5, 8.6),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#C62828",
                                connectionstyle="arc3,rad=-0.2"))

    draw_rbox(ax, 3.8, 7.8, 3.5, 1.6, "U-Net 모델\n\n목표: n(t) 노이즈만\n추정", "#C62828")
    ax.annotate("", xy=(5.5, 7.8), xytext=(5.5, 7.3),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#333"))

    draw_rbox(ax, 3.5, 6.2, 4, 1.0, "노이즈 추정: n'(t)", "#FF5722")

    # Subtract 연산
    ax.annotate("", xy=(5.5, 6.2), xytext=(5.5, 5.7),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#333"))

    draw_rbox(ax, 3, 4.5, 5, 1.0,
              "s'(t) = x(t) - n'(t)\n(입력 - 노이즈 추정 = 깨끗한 심음)",
              "#00695C", fontsize=10)

    # bypass 라벨
    ax.text(8.7, 8.8, "bypass\n(원본 보존)", fontsize=9, ha="center",
            color="#C62828", fontweight="bold", style="italic")

    # 장점
    props = dict(boxstyle="round,pad=0.6", facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=2)
    advantages = (
        "장점:\n\n"
        "• 노이즈는 심음보다 단순 → 학습이 쉬움\n"
        "  (sparse한 문제로 변환)\n\n"
        "• 노이즈가 없는 구간 -> n'(t)~0 → 원본 그대로 보존\n\n"
        "• S1/S2 피크 왜곡 최소화 (입력 bypass)\n\n"
        "• Loss: Combined = 0.9×MSE + 0.1×SI-SNR\n"
        "  → 파형 형태까지 최적화"
    )
    ax.text(5, 2.2, advantages, fontsize=10, ha="center", va="center",
            bbox=props, linespacing=1.4)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    path = os.path.join(OUTPUT_DIR, "09_direct_vs_residual.png")
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# Figure 10: v1→v2 개선 요소별 효과 요약 (발표용 한 장)
# ================================================================
def fig10_improvement_summary():
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis("off")
    fig.suptitle("v1 → v2 개선: 세 가지 기법이 성능에 미친 영향",
                 fontsize=22, fontweight="bold", y=0.97)

    def draw_card(x, y, w, h, title, content, title_color, bg_color, border_color):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                             facecolor=bg_color, edgecolor=border_color,
                             linewidth=2, zorder=3)
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 0.4, title, ha="center", va="top",
                fontsize=14, fontweight="bold", color=title_color, zorder=4)
        ax.text(x + w/2, y + h/2 - 0.3, content, ha="center", va="center",
                fontsize=11, color="#333", zorder=4, linespacing=1.5)

    # ── v1 baseline ──
    draw_card(0.5, 8.5, 5.5, 3.0,
              "v1 Baseline",
              "U-Net + Bi-LSTM + MSE\n\n"
              "ΔSNR: +5.94 dB\n"
              "SI-SNR: 9.82 dB\n"
              "복원율: 54.20%",
              "#1565C0", "#E3F2FD", "#1565C0")

    # 화살표: v1 → 개선요소들
    ax.annotate("", xy=(7, 10.0), xytext=(6.0, 10.0),
                arrowprops=dict(arrowstyle="-|>", lw=3, color="#333"))

    # ── 개선 요소 1: Channel Attention ──
    draw_card(7.2, 8.5, 5.3, 3.0,
              "① Channel Attention (SE Block)",
              "Global Avg Pool → FC → Sigmoid\n"
              "→ 채널별 가중치(0~1) 학습\n\n"
              "효과: 심음 관련 채널 강조\n"
              "      노이즈 채널 억제\n"
              "기여: SI-SNR +2~3 dB 향상",
              "#7B1FA2", "#F3E5F5", "#7B1FA2")

    # ── 개선 요소 2: Residual Learning ──
    draw_card(0.5, 4.5, 5.5, 3.5,
              "② Residual Learning",
              "모델 출력 = 노이즈 추정치\n"
              "clean = input - noise_est\n\n"
              "효과: 심음 원본 bypass로 보존\n"
              "      노이즈 없는 구간 왜곡 없음\n"
              "기여: S1/S2 피크 보존 ↑\n"
              "      과도한 smoothing 방지",
              "#C62828", "#FFEBEE", "#C62828")

    # ── 개선 요소 3: SI-SNR Loss ──
    draw_card(7.2, 4.5, 5.3, 3.5,
              "③ SI-SNR Loss (90:10 배합)",
              "Combined = 0.9×MSE + 0.1×SI-SNR/20\n\n"
              "MSE: 빠른 수렴 + 안정적 학습\n"
              "SI-SNR: 파형 형태 일치도 직접 최적화\n\n"
              "50:50 → 학습 불안정 (ΔSNR 4.55)\n"
              "90:10 → 안정 + 품질 향상 (최적)",
              "#F57F17", "#FFF8E1", "#F57F17")

    # ── v2 결과 ──
    # 모든 화살표 → v2
    ax.annotate("", xy=(14, 10.0), xytext=(12.5, 10.0),
                arrowprops=dict(arrowstyle="-|>", lw=3, color="#333"))
    ax.annotate("", xy=(14, 7.0), xytext=(12.5, 6.2),
                arrowprops=dict(arrowstyle="-|>", lw=3, color="#333",
                                connectionstyle="arc3,rad=0.2"))
    ax.annotate("", xy=(14, 7.0), xytext=(6.0, 6.2),
                arrowprops=dict(arrowstyle="-|>", lw=3, color="#333",
                                connectionstyle="arc3,rad=-0.2"))

    draw_card(14, 5.5, 5.5, 5.5,
              "v2 최종 결과",
              "U-Net + Bi-LSTM\n"
              "+ Attention + Residual + SI-SNR\n\n"
              "-------------------\n"
              "ΔSNR: +5.59 dB\n"
              "SI-SNR: 12.76 dB  (+2.94↑)\n"
              "복원율: 55.30%    (+1.10↑)\n"
              "-------------------\n\n"
              "SI-SNR 30% 개선!\n"
              "(파형 형태 보존도 크게 향상)",
              "#00695C", "#E0F2F1", "#00695C")

    # ── 하단: 핵심 인사이트 ──
    props = dict(boxstyle="round,pad=0.8", facecolor="#FFECB3",
                 edgecolor="#FF8F00", linewidth=2.5)
    insight = (
        "핵심 인사이트:  ΔSNR은 비슷하지만(-0.35), SI-SNR이 +2.94 dB 크게 향상\n"
        "→ 수치적 노이즈 제거량은 비슷하나, 파형의 형태 보존도(= 음질)가 크게 개선됨\n"
        "→ 발표 포인트: \"숫자상 비슷해 보이지만, 실제로 들어보면 v2가 확연히 깨끗하다\""
    )
    ax.text(10, 1.5, insight, fontsize=12, ha="center", va="center",
            bbox=props, linespacing=1.5)

    path = os.path.join(OUTPUT_DIR, "10_improvement_summary.png")
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 60)
    print(" 아키텍처 & LSTM vs TCN 시각화 생성")
    print("=" * 60)

    print("\n[1] Figure 7: 디노이즈 파이프라인...")
    fig7_denoise_pipeline()

    print("[2] Figure 8: LSTM vs TCN 비교...")
    fig8_lstm_vs_tcn()

    print("[3] Figure 9: Direct vs Residual Learning...")
    fig9_residual_learning()

    print("[4] Figure 10: v1→v2 개선 요약...")
    fig10_improvement_summary()

    print("\n" + "=" * 60)
    print(f" 모든 그래프 저장 완료!")
    print(f" 위치: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
