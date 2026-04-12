"""
================================================================
심음 디노이즈 결과 시각화 - 발표용
================================================================
4개 모델 비교: 파형, 스펙트로그램, 잔여노이즈, SNR 분포
출력: G:\stetho_ai\ShittyDenoise\LU-NET\figures\ 에 PNG 저장
================================================================
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager
import soundfile as sf

# ── 한글 폰트 설정 ──
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

SR = 4000
DATASET_DIR = r"G:\stetho_ai\_misc\datasets\LUNet_Dataset"
OUTPUT_DIR = r"G:\stetho_ai\ShittyDenoise\LU-NET\figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 파일 매핑 ──
FILES = {
    "noisy":    os.path.join(DATASET_DIR, "원본 잡음섞인 심음.wav"),
    "LU-Net v2": os.path.join(DATASET_DIR, "LUNET_V2.wav"),
    "TU-Net v1": os.path.join(DATASET_DIR, "TUNET_V1.wav"),
    "TU-Net v2": os.path.join(DATASET_DIR, "TUNET_V2.wav"),
}

# 성능 수치 (ablation study 결과)
METRICS = {
    "LU-Net v2": {"delta_snr": 5.59, "si_snr": 12.76, "restore": 55.30},
    "TU-Net v1": {"delta_snr": 4.35, "si_snr": 8.14,  "restore": 50.31},
    "TU-Net v2": {"delta_snr": 5.63, "si_snr": 12.59, "restore": 55.38},
}

COLORS = {
    "noisy":     "#888888",
    "LU-Net v2": "#2196F3",
    "TU-Net v1": "#FF9800",
    "TU-Net v2": "#4CAF50",
}

MODEL_ORDER = ["LU-Net v2", "TU-Net v1", "TU-Net v2"]


def load_all():
    """모든 wav 로드, 길이 맞춤"""
    data = {}
    min_len = float("inf")
    for name, path in FILES.items():
        if not os.path.exists(path):
            print(f"  [!] 파일 없음: {path}")
            continue
        y, _ = librosa.load(path, sr=SR)
        data[name] = y
        min_len = min(min_len, len(y))
    # 길이 통일
    for name in data:
        data[name] = data[name][:min_len]
    return data


# ================================================================
# Figure 1: 파형 비교 (전체 + 확대)
# ================================================================
def fig1_waveform_comparison(data):
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("심음 디노이즈 파형 비교", fontsize=20, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(4, 2, hspace=0.45, wspace=0.25,
                           left=0.07, right=0.97, top=0.93, bottom=0.05)

    noisy = data["noisy"]
    t = np.arange(len(noisy)) / SR

    # S1/S2 심박 주기 확대 구간 (약 1~3초)
    zoom_start = int(1.0 * SR)
    zoom_end = int(3.0 * SR)
    t_zoom = t[zoom_start:zoom_end]

    labels = ["noisy"] + MODEL_ORDER
    titles_full = [
        "원본 (노이즈 섞인 심음)",
        "LU-Net v2  (LSTM + Attn + Res + SI-SNR)  ★ 최적",
        "TU-Net v1  (TCN + MSE)  — baseline",
        "TU-Net v2  (TCN + Attn + Res + SI-SNR)",
    ]

    for i, (name, title) in enumerate(zip(labels, titles_full)):
        if name not in data:
            continue
        y = data[name]
        color = COLORS[name]

        # 전체 파형
        ax_full = fig.add_subplot(gs[i, 0])
        ax_full.plot(t, y, color=color, linewidth=0.3, alpha=0.8)
        ax_full.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax_full.set_ylim(-1.05, 1.05)
        ax_full.set_ylabel("진폭")
        if i == 3:
            ax_full.set_xlabel("시간 (초)")
        # 확대 영역 표시
        ax_full.axvspan(t[zoom_start], t[zoom_end], alpha=0.12, color="red")

        # 확대 파형
        ax_zoom = fig.add_subplot(gs[i, 1])
        ax_zoom.plot(t_zoom, y[zoom_start:zoom_end], color=color, linewidth=0.6)
        ax_zoom.set_title(f"확대 (1.0~3.0초, S1/S2 심박 주기)", fontsize=10, color="red")
        ax_zoom.set_ylim(-1.05, 1.05)
        if i == 3:
            ax_zoom.set_xlabel("시간 (초)")

        # S1/S2 레이블 (noisy 제외, 첫 번째 모델에만)
        if name == "LU-Net v2":
            # 대략적 S1, S2 위치 자동 탐지 (피크 기반)
            chunk = np.abs(y[zoom_start:zoom_end])
            threshold = np.max(chunk) * 0.45
            peaks = []
            in_peak = False
            for j in range(len(chunk)):
                if chunk[j] > threshold and not in_peak:
                    in_peak = True
                    peak_start = j
                elif chunk[j] < threshold * 0.5 and in_peak:
                    in_peak = False
                    peak_center = (peak_start + j) // 2
                    peaks.append(peak_center)
            # S1, S2 번갈아 표시
            for pi, pk in enumerate(peaks[:6]):
                label = "S1" if pi % 2 == 0 else "S2"
                ax_zoom.annotate(
                    label,
                    xy=(t_zoom[pk], y[zoom_start + pk]),
                    xytext=(0, 15), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="red",
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
                )

    path = os.path.join(OUTPUT_DIR, "01_waveform_comparison.png")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# Figure 2: 스펙트로그램 비교
# ================================================================
def fig2_spectrogram_comparison(data):
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    fig.suptitle("멜 스펙트로그램 비교 (64-mel, 4kHz)", fontsize=18, fontweight="bold")

    labels = ["noisy"] + MODEL_ORDER
    titles = [
        "원본 (노이즈)",
        "LU-Net v2  ★ 최적 — ΔSNR +5.59 dB",
        "TU-Net v1  — ΔSNR +4.35 dB",
        "TU-Net v2  — ΔSNR +5.63 dB",
    ]

    vmin, vmax = None, None
    # 먼저 공통 스케일 계산
    for name in labels:
        if name not in data:
            continue
        S = librosa.feature.melspectrogram(y=data[name], sr=SR, n_mels=64, fmax=2000)
        S_db = librosa.power_to_db(S, ref=np.max)
        if vmin is None:
            vmin, vmax = S_db.min(), S_db.max()
        else:
            vmin = min(vmin, S_db.min())
            vmax = max(vmax, S_db.max())

    for i, (name, title) in enumerate(zip(labels, titles)):
        if name not in data:
            continue
        S = librosa.feature.melspectrogram(y=data[name], sr=SR, n_mels=64, fmax=2000)
        S_db = librosa.power_to_db(S, ref=np.max)

        ax = axes[i]
        img = librosa.display.specshow(
            S_db, sr=SR, x_axis="time", y_axis="mel",
            ax=ax, fmax=2000, vmin=vmin, vmax=vmax,
            cmap="magma"
        )
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
        ax.set_ylabel("주파수 (Hz)")
        if i < 3:
            ax.set_xlabel("")

    fig.colorbar(img, ax=axes, format="%+2.0f dB", shrink=0.6, pad=0.02)
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])

    path = os.path.join(OUTPUT_DIR, "02_spectrogram_comparison.png")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# Figure 3: 잔여 노이즈 비교 (원본 - 디노이즈 = 제거된 성분)
# ================================================================
def fig3_residual_noise(data):
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True, sharey=True)
    fig.suptitle("제거된 노이즈 성분 비교  (원본 - 디노이즈 = 제거 성분)",
                 fontsize=16, fontweight="bold")

    noisy = data["noisy"]
    t = np.arange(len(noisy)) / SR

    for i, name in enumerate(MODEL_ORDER):
        if name not in data:
            continue
        residual = noisy - data[name]
        rms = np.sqrt(np.mean(residual**2))

        ax = axes[i]
        ax.plot(t, residual, color=COLORS[name], linewidth=0.3, alpha=0.7)
        ax.set_title(f"{name}  — 제거된 성분 (RMS: {rms:.4f})",
                     fontsize=12, fontweight="bold", loc="left")
        ax.set_ylabel("진폭")
        ax.set_ylim(-1.0, 1.0)

    axes[-1].set_xlabel("시간 (초)")
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    path = os.path.join(OUTPUT_DIR, "03_residual_noise.png")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# Figure 4: 성능 지표 바 차트 (발표 핵심 슬라이드)
# ================================================================
def fig4_performance_bars():
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("모델별 디노이즈 성능 비교", fontsize=18, fontweight="bold")

    models = MODEL_ORDER
    colors = [COLORS[m] for m in models]

    # ΔSNR
    ax = axes[0]
    vals = [METRICS[m]["delta_snr"] for m in models]
    bars = ax.bar(models, vals, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_title("ΔSNR (dB)  ↑ 높을수록 좋음", fontsize=13, fontweight="bold")
    ax.set_ylabel("dB")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"+{v:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.25)

    # SI-SNR
    ax = axes[1]
    vals = [METRICS[m]["si_snr"] for m in models]
    bars = ax.bar(models, vals, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_title("SI-SNR (dB)  ↑ 높을수록 좋음", fontsize=13, fontweight="bold")
    ax.set_ylabel("dB")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{v:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.25)

    # 복원율
    ax = axes[2]
    vals = [METRICS[m]["restore"] for m in models]
    bars = ax.bar(models, vals, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_title("복원율 (%)  ↑ 높을수록 좋음", fontsize=13, fontweight="bold")
    ax.set_ylabel("%")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.2)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    path = os.path.join(OUTPUT_DIR, "04_performance_bars.png")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# Figure 5: Ablation 기여도 분석 (각 요소가 얼마나 개선했는지)
# ================================================================
def fig5_ablation_contribution():
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle("Ablation Study — 각 기법의 ΔSNR 기여도",
                 fontsize=18, fontweight="bold")

    # baseline → 각 요소 추가 시 ΔSNR 변화 (LU-Net 기준)
    # LU-Net v1 (baseline MSE) = +5.94
    # + Attention = +5.42 (단독 실험에선 약간 하락, 조합 효과)
    # + Residual = +5.52
    # + SI-SNR(50:50) = +4.55 (과도한 가중치)
    # + 전부 (90:10) = +5.59 (최종)
    # TU-Net v1 baseline = +4.35
    # TU-Net v2 (전부) = +5.63

    categories = [
        "LU-Net v1\n(Baseline\nLSTM+MSE)",
        "LU-Net v2\n+Attn+Res\n+SI-SNR(90:10)",
        "TU-Net v1\n(Baseline\nTCN+MSE)",
        "TU-Net v2\n+Attn+Res\n+SI-SNR(90:10)",
    ]
    delta_snrs = [5.94, 5.59, 4.35, 5.63]
    si_snrs = [9.82, 12.76, 8.14, 12.59]
    bar_colors = ["#90CAF9", "#2196F3", "#FFE0B2", "#4CAF50"]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, delta_snrs, width, label="ΔSNR (dB)",
                   color=bar_colors, edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width/2, si_snrs, width, label="SI-SNR (dB)",
                   color=bar_colors, edgecolor="black", linewidth=0.8, alpha=0.6,
                   hatch="//")

    for bar, v in zip(bars1, delta_snrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f"+{v:.2f}", ha="center", fontsize=11, fontweight="bold")
    for bar, v in zip(bars2, si_snrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("dB", fontsize=13)
    ax.legend(fontsize=12, loc="upper right")
    ax.set_ylim(0, max(si_snrs) * 1.3)

    # 핵심 메시지 박스
    textstr = (
        "핵심 발견:\n"
        "• Attention+Residual+SI-SNR → SI-SNR 약 +3dB 개선\n"
        "• LSTM이 TCN보다 심음의 주기적 패턴 포착에 유리\n"
        "• SI-SNR Loss 가중치: 90:10(MSE:SI-SNR)이 최적"
    )
    props = dict(boxstyle="round,pad=0.8", facecolor="#E3F2FD", alpha=0.9,
                 edgecolor="#1976D2", linewidth=1.5)
    ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", bbox=props)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    path = os.path.join(OUTPUT_DIR, "05_ablation_contribution.png")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# Figure 6: 왜 LU-Net v2가 청음에 최적인가 (종합 인포그래픽)
# ================================================================
def fig6_why_best(data):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("왜 LU-Net v2가 가장 듣기 편한가?", fontsize=20, fontweight="bold")

    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35,
                           left=0.06, right=0.97, top=0.90, bottom=0.08)

    noisy = data["noisy"]
    best = data.get("LU-Net v2", noisy)
    worst = data.get("TU-Net v1", noisy)

    # 확대 구간 (심박 1주기)
    z_s, z_e = int(1.2 * SR), int(2.4 * SR)
    t_z = np.arange(z_e - z_s) / SR + 1.2

    # ── (0,0) 원본 파형 확대 ──
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t_z, noisy[z_s:z_e], color=COLORS["noisy"], linewidth=0.8)
    ax.set_title("원본 (노이즈)", fontsize=13, fontweight="bold")
    ax.set_ylabel("진폭")
    ax.set_ylim(-1.05, 1.05)

    # ── (0,1) LU-Net v2 확대 ──
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t_z, best[z_s:z_e], color=COLORS["LU-Net v2"], linewidth=0.8)
    ax.set_title("LU-Net v2 ★ (LSTM+Attn+Res)", fontsize=13, fontweight="bold",
                 color="#1565C0")
    ax.set_ylim(-1.05, 1.05)

    # ── (0,2) TU-Net v1 확대 ──
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t_z, worst[z_s:z_e], color=COLORS["TU-Net v1"], linewidth=0.8)
    ax.set_title("TU-Net v1 (TCN+MSE)", fontsize=13, fontweight="bold",
                 color="#E65100")
    ax.set_ylim(-1.05, 1.05)

    # ── (1,0) 에너지 엔벨로프 비교 ──
    ax = fig.add_subplot(gs[1, 0])
    frame_len = int(0.025 * SR)  # 25ms
    hop = int(0.010 * SR)        # 10ms

    for name in ["noisy", "LU-Net v2", "TU-Net v1"]:
        if name not in data:
            continue
        y = data[name]
        # RMS 에너지
        frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop)
        rms = np.sqrt(np.mean(frames**2, axis=0))
        t_rms = np.arange(len(rms)) * hop / SR
        ax.plot(t_rms, rms, color=COLORS[name], linewidth=1.2, label=name, alpha=0.8)

    ax.set_title("에너지 엔벨로프 (RMS)", fontsize=13, fontweight="bold")
    ax.set_xlabel("시간 (초)")
    ax.set_ylabel("RMS")
    ax.legend(fontsize=10, loc="upper right")

    # ── (1,1) 주파수 에너지 분포 ──
    ax = fig.add_subplot(gs[1, 1])
    for name in ["noisy", "LU-Net v2", "TU-Net v1"]:
        if name not in data:
            continue
        y = data[name]
        fft = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), d=1/SR)
        # 스무딩
        window = 50
        fft_smooth = np.convolve(fft, np.ones(window)/window, mode="same")
        ax.plot(freqs, 20*np.log10(fft_smooth + 1e-10),
                color=COLORS[name], linewidth=1.2, label=name, alpha=0.8)

    ax.set_title("주파수 스펙트럼", fontsize=13, fontweight="bold")
    ax.set_xlabel("주파수 (Hz)")
    ax.set_ylabel("크기 (dB)")
    ax.set_xlim(0, 2000)
    ax.legend(fontsize=10, loc="upper right")
    # 심음 대역 표시
    ax.axvspan(20, 150, alpha=0.1, color="red", label="S1/S2 대역")
    ax.text(85, ax.get_ylim()[1]-5, "S1/S2\n(20-150Hz)", fontsize=9,
            ha="center", color="red", fontweight="bold")

    # ── (1,2) 종합 결론 텍스트 박스 ──
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")

    conclusion = (
        "━━━  LU-Net v2가 최적인 이유  ━━━\n\n"
        "1. LSTM 스킵 커넥션\n"
        "   → 심박 주기(~0.8초) 장기 패턴 보존\n"
        "   → S1/S2 심음 왜곡 최소화\n\n"
        "2. Channel Attention (SE Block)\n"
        "   → 심음 주파수 대역(20-150Hz) 선택적 강화\n"
        "   → 불필요 대역 자동 억제\n\n"
        "3. Residual Learning\n"
        "   → 노이즈만 학습 → 원본 신호 보존\n"
        "   → 과도한 스무딩 방지\n\n"
        "4. SI-SNR Loss (90:10 배합)\n"
        "   → 신호 왜곡 최소화 + 안정적 학습\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  ΔSNR +5.59 dB  |  SI-SNR 12.76 dB\n"
        "  복원율 55.30%"
    )
    props = dict(boxstyle="round,pad=1.0", facecolor="#E8F5E9",
                 edgecolor="#2E7D32", linewidth=2)
    ax.text(0.5, 0.5, conclusion, transform=ax.transAxes,
            fontsize=12, verticalalignment="center", horizontalalignment="center",
            bbox=props, family="Malgun Gothic", linespacing=1.4)

    path = os.path.join(OUTPUT_DIR, "06_why_lunet_v2_best.png")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 60)
    print(" 심음 디노이즈 시각화 생성")
    print("=" * 60)

    print("\n[1] 오디오 로드 중...")
    data = load_all()
    print(f"    로드 완료: {list(data.keys())}")
    print(f"    길이: {len(list(data.values())[0])/SR:.1f}초")

    print("\n[2] Figure 1: 파형 비교...")
    fig1_waveform_comparison(data)

    print("[3] Figure 2: 스펙트로그램 비교...")
    fig2_spectrogram_comparison(data)

    print("[4] Figure 3: 잔여 노이즈...")
    fig3_residual_noise(data)

    print("[5] Figure 4: 성능 바 차트...")
    fig4_performance_bars()

    print("[6] Figure 5: Ablation 기여도...")
    fig5_ablation_contribution()

    print("[7] Figure 6: 왜 LU-Net v2가 최적인가...")
    fig6_why_best(data)

    print("\n" + "=" * 60)
    print(f" 모든 그래프 저장 완료!")
    print(f" 위치: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
