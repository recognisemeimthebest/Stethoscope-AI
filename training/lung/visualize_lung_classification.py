"""
================================================================
폐음 분류 시각화 — 발표용 Figure 모음
================================================================
Figure 01: 분류 파이프라인 (데이터 → 전처리 → 모델 → 추론)
Figure 02: 모델 진화 히스토리 (Phase 1 → Final v2)
Figure 03: 데이터 구성 & 전처리 (소스, 클래스 병합, 증강)
Figure 04: 정상 vs 비정상 폐음 비교 (파형 + 스펙트로그램)
Figure 05: 아키텍처 비교 (ResNet18 vs MobileNetV2)
Figure 06: 핵심 개선 요약 (해상도, Loss, 데이터)
================================================================
"""

import os
import json
import csv
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = r"G:\stetho_ai\Lung_classification\figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset paths
ICBHI_BASE = r"G:\stetho_ai\_misc\lung\datasets\ICBHI_respiratory_db\Respiratory_Sound_Database\Respiratory_Sound_Database"
ICBHI_AUDIO = os.path.join(ICBHI_BASE, "audio_and_txt_files")
ICBHI_DIAG = os.path.join(ICBHI_BASE, "patient_diagnosis.csv")

SPR_BASE = r"G:\stetho_ai\_misc\lung\classification\LUNG_SOUND\SPRSound-main\Classification"
SPR_TRAIN_WAV = os.path.join(SPR_BASE, "train_classification_wav")
SPR_TRAIN_JSON = os.path.join(SPR_BASE, "train_classification_json")

# Colors
C_NORM = "#43A047"
C_ABNORM = "#E53935"
C_BLUE = "#1565C0"
C_ORANGE = "#E65100"
C_PURPLE = "#7B1FA2"
C_TEAL = "#00897B"
C_GRAY = "#757575"


def _rounded_box(ax, x, y, w, h, text, color, fontsize=11, text_color="white"):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor="white", linewidth=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color)


def _arrow_right(ax, x1, y, x2):
    """Draw horizontal arrow."""
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=2.5))


# ================================================================
# Figure 1: Classification Pipeline
# ================================================================
def fig01_pipeline():
    fig, ax = plt.subplots(figsize=(22, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("폐음 분류 파이프라인", fontsize=22, fontweight="bold", y=0.97)

    steps = [
        ("1. 데이터 수집\n(ICBHI + SPRSound)", "#1565C0"),
        ("2. 클래스 병합\n7 -> 5 classes", "#00897B"),
        ("3. 증강 & 밸런싱\n(Noise, Pitch Shift)", "#E65100"),
        ("4. Mel-Spectrogram\nPNG 변환", "#7B1FA2"),
        ("5. MobileNetV2\n(Pretrained)", "#C62828"),
        ("6. 5-Class\n분류 결과", "#2E7D32"),
    ]

    n = len(steps)
    bw, bh = 0.12, 0.22
    gap = (1 - n * bw) / (n + 1)
    y_center = 0.45

    for i, (label, color) in enumerate(steps):
        x = gap + i * (bw + gap)
        _rounded_box(ax, x, y_center, bw, bh, label, color, fontsize=10)
        if i < n - 1:
            _arrow_right(ax, x + bw + 0.005, y_center + bh / 2, x + bw + gap - 0.005)

    # Bottom details
    details = [
        "ICBHI: 920 recordings\nSPRSound: 3554 recordings",
        "Fine/Coarse Crackle -> Crackle\nWheeze/Rhonchi -> Wheeze",
        "Normal: 1000 cap\n소수 클래스: 2-5x 증강",
        "224x224 pixels\n128 mel bands",
        "ImageNet pretrained\nFocal Loss (gamma=2)",
        "Normal / Crackle\nWheeze / Complex / Stridor",
    ]
    for i, detail in enumerate(details):
        x = gap + i * (bw + gap)
        ax.text(x + bw / 2, y_center - 0.08, detail,
                ha="center", va="top", fontsize=8.5, color="#555",
                style="italic")

    path = os.path.join(OUTPUT_DIR, "01_classification_pipeline.png")
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# Figure 2: Model Evolution History
# ================================================================
def fig02_evolution():
    fig, ax = plt.subplots(figsize=(24, 12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("폐음 분류 모델 진화 과정", fontsize=22, fontweight="bold", y=0.97)

    phases = [
        {
            "title": "Phase 1\n(초기 시도)",
            "color": "#90A4AE",
            "items": [
                "ResNet18 (scratch)",
                "SPRSound 2022만",
                "1-ch Mel-Spec",
                "10 epochs",
                "No scheduler",
            ],
            "result": "기본 baseline\n과적합 심함",
        },
        {
            "title": "Phase 2\n(데이터 확장)",
            "color": "#78909C",
            "items": [
                "ResNet18 (scratch)",
                "2022 + 2023 통합",
                "Binary (정상/비정상)",
                "기본 학습",
            ],
            "result": "데이터 증가로\n소폭 개선",
        },
        {
            "title": "Phase 3\n(기법 강화)",
            "color": "#546E7A",
            "items": [
                "ResNet18 (pretrained)",
                "2022-2025 통합",
                "Weighted CE Loss",
                "SpecAugment",
                "Dropout 0.5",
            ],
            "result": "비정상 recall\n크게 향상",
        },
        {
            "title": "Final v1\n(아키텍처 전환)",
            "color": "#1565C0",
            "items": [
                "MobileNetV2 (pretrained)",
                "128x128 PNG",
                "CrossEntropy",
                "LR Scheduler",
                "EarlyStopping",
            ],
            "result": "경량화 +\n성능 유지",
        },
        {
            "title": "Final v2\n(현재 최고)",
            "color": "#C62828",
            "items": [
                "MobileNetV2 (pretrained)",
                "224x224 PNG",
                "Focal Loss (gamma=2)",
                "ReduceLROnPlateau",
                "EarlyStopping(5)",
            ],
            "result": "최고 성능\n배포 모델",
        },
    ]

    n = len(phases)
    bw, bh = 0.155, 0.55
    gap = (1 - n * bw) / (n + 1)
    y_top = 0.75

    for i, phase in enumerate(phases):
        x = gap + i * (bw + gap)
        # Title box
        _rounded_box(ax, x, y_top, bw, 0.1, phase["title"], phase["color"], fontsize=10)

        # Content area
        content_y = y_top - 0.02
        for j, item in enumerate(phase["items"]):
            ax.text(x + bw / 2, content_y - 0.06 * (j + 1), f"- {item}",
                    ha="center", va="center", fontsize=9, color="#333")

        # Result box at bottom
        result_y = 0.12
        result_color = "#4CAF50" if i == n - 1 else "#FFF9C4"
        result_text_color = "white" if i == n - 1 else "#333"
        _rounded_box(ax, x + 0.01, result_y, bw - 0.02, 0.1,
                     phase["result"], result_color, fontsize=9, text_color=result_text_color)

        # Arrow to next
        if i < n - 1:
            _arrow_right(ax, x + bw + 0.005, y_top + 0.05, x + bw + gap - 0.005)

    # Key improvements annotation
    improvements = [
        (1.5, "데이터 통합"),
        (2.5, "Pretrain\n+SpecAugment"),
        (3.5, "MobileNetV2\n전환"),
        (4.5, "224x224\n+Focal Loss"),
    ]
    for idx_mid, label in improvements:
        x_mid = gap + idx_mid * (bw + gap) - gap / 2 + bw / 2
        ax.text(x_mid, y_top + 0.15, label,
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=C_BLUE, bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", edgecolor=C_BLUE))

    path = os.path.join(OUTPUT_DIR, "02_model_evolution.png")
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# Figure 3: Data Composition & Preprocessing
# ================================================================
def fig03_data_preprocessing():
    fig = plt.figure(figsize=(22, 16))
    fig.suptitle("데이터 구성 및 전처리 과정", fontsize=22, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3,
                           left=0.06, right=0.96, top=0.92, bottom=0.05)

    # (0,0) Data Sources
    ax1 = fig.add_subplot(gs[0, 0])
    sources = ["ICBHI 2017\n(920 recordings)\n(Adult)", "SPRSound 2022\n(BioCAS)\n(Child)",
               "SPRSound 2023", "SPRSound 2024", "SPRSound 2025"]
    colors_src = ["#1565C0", "#E65100", "#F57C00", "#FF9800", "#FFB74D"]
    sizes = [920, 1200, 800, 600, 954]
    bars = ax1.barh(range(len(sources)), sizes, color=colors_src, edgecolor="white", height=0.6)
    ax1.set_yticks(range(len(sources)))
    ax1.set_yticklabels(sources, fontsize=9)
    ax1.set_xlabel("Recordings")
    ax1.set_title("데이터 소스", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, sizes):
        ax1.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                 f"{val}", va="center", fontsize=9, fontweight="bold")
    ax1.invert_yaxis()

    # (0,1) Class Merging 7 -> 5
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title("클래스 병합 (7 -> 5)", fontsize=14, fontweight="bold")

    original = ["Normal", "Fine Crackle", "Coarse Crackle", "Wheeze", "Rhonchi", "Wheeze+Crackle", "Stridor"]
    merged = ["Normal", "Crackle", "Crackle", "Wheeze", "Wheeze", "Complex", "Stridor"]
    merged_colors = [C_NORM, "#FF7043", "#FF7043", "#42A5F5", "#42A5F5", C_PURPLE, "#FFB300"]

    for i, (orig, mrg, clr) in enumerate(zip(original, merged, merged_colors)):
        y = 0.88 - i * 0.12
        # Original
        ax2.text(0.15, y, orig, ha="center", va="center", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECEFF1", edgecolor="#999"))
        # Arrow
        ax2.annotate("", xy=(0.55, y), xytext=(0.35, y),
                     arrowprops=dict(arrowstyle="-|>", color=clr, lw=2))
        # Merged
        ax2.text(0.75, y, mrg, ha="center", va="center", fontsize=10, fontweight="bold",
                 color="white",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=clr, edgecolor="white"))

    # (0,2) Augmentation Strategy
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title("증강 & 밸런싱 전략", fontsize=14, fontweight="bold")

    strategies = [
        ("Normal", "Downsample -> 1000", C_NORM),
        ("Crackle", "2x augmentation", "#FF7043"),
        ("Wheeze", "3x augmentation", "#42A5F5"),
        ("Complex", "5x augmentation", C_PURPLE),
        ("Stridor", "5x augmentation", "#FFB300"),
    ]
    for i, (cls, strategy, clr) in enumerate(strategies):
        y = 0.85 - i * 0.15
        ax3.text(0.05, y, cls, fontsize=11, fontweight="bold", color=clr, va="center")
        ax3.text(0.35, y, strategy, fontsize=10, va="center", color="#333")

    ax3.text(0.5, 0.12, "증강 기법:", fontsize=10, fontweight="bold", ha="center", color="#333")
    ax3.text(0.5, 0.04, "Gaussian Noise + Pitch Shift (+-2 semitones)",
             fontsize=9, ha="center", color="#666", style="italic")

    # (1,0) & (1,1) Sample Spectrograms from ICBHI
    try:
        # Read ICBHI patient diagnosis
        patient_diag = {}
        with open(ICBHI_DIAG) as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    patient_diag[row[0].strip()] = row[1].strip()

        # Find normal and abnormal wav files
        normal_files, abnormal_files = [], []
        if os.path.exists(ICBHI_AUDIO):
            for fname in os.listdir(ICBHI_AUDIO):
                if not fname.endswith(".wav"):
                    continue
                pid = fname.split("_")[0]
                diag = patient_diag.get(pid, "")
                wav_path = os.path.join(ICBHI_AUDIO, fname)
                if diag in ("Healthy",):
                    normal_files.append(wav_path)
                elif diag in ("COPD", "Pneumonia", "Bronchiectasis"):
                    abnormal_files.append(wav_path)

        rng = np.random.RandomState(42)

        # Normal spectrogram
        if normal_files:
            ax_norm = fig.add_subplot(gs[1, 0])
            wav_path = rng.choice(normal_files)
            y_audio, sr = librosa.load(wav_path, sr=None)
            clip_len = min(len(y_audio), int(sr * 8))
            y_clip = y_audio[:clip_len]
            S = librosa.feature.melspectrogram(y=y_clip, sr=sr, n_mels=128, fmax=sr // 2, hop_length=256)
            S_db = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel",
                                     ax=ax_norm, fmax=sr // 2, cmap="magma", hop_length=256)
            fname = os.path.basename(wav_path)
            pid = fname.split("_")[0]
            ax_norm.set_title(f"정상 폐음 ({patient_diag.get(pid, 'Healthy')})\n{fname}",
                              fontsize=12, fontweight="bold", color=C_NORM)
            ax_norm.set_ylabel("주파수 (Hz)")
            ax_norm.set_xlabel("시간 (초)")

        # Abnormal spectrogram
        if abnormal_files:
            ax_abn = fig.add_subplot(gs[1, 1])
            wav_path = rng.choice(abnormal_files)
            y_audio, sr = librosa.load(wav_path, sr=None)
            clip_len = min(len(y_audio), int(sr * 8))
            y_clip = y_audio[:clip_len]
            S = librosa.feature.melspectrogram(y=y_clip, sr=sr, n_mels=128, fmax=sr // 2, hop_length=256)
            S_db = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel",
                                     ax=ax_abn, fmax=sr // 2, cmap="magma", hop_length=256)
            fname = os.path.basename(wav_path)
            pid = fname.split("_")[0]
            ax_abn.set_title(f"비정상 폐음 ({patient_diag.get(pid, '?')})\n{fname}",
                             fontsize=12, fontweight="bold", color=C_ABNORM)
            ax_abn.set_ylabel("주파수 (Hz)")
            ax_abn.set_xlabel("시간 (초)")

    except Exception as e:
        print(f"  [!] ICBHI 스펙트로그램 생성 실패: {e}")

    # (1,2) Mel-Spectrogram Parameters
    ax_params = fig.add_subplot(gs[1, 2])
    ax_params.axis("off")
    ax_params.set_xlim(0, 1)
    ax_params.set_ylim(0, 1)
    ax_params.set_title("Mel-Spectrogram 설정", fontsize=14, fontweight="bold")

    params = [
        ("n_mels", "128 bands"),
        ("fmin", "100 Hz"),
        ("fmax", "2000 Hz"),
        ("hop_length", "256"),
        ("Output Size", "128x128 -> 224x224"),
        ("Color Map", "Viridis -> PNG"),
        ("Format", "3-ch RGB image"),
    ]
    for i, (key, val) in enumerate(params):
        y = 0.88 - i * 0.11
        ax_params.text(0.05, y, f"{key}:", fontsize=10, fontweight="bold", color="#333", va="center")
        ax_params.text(0.45, y, val, fontsize=10, color="#555", va="center")

    path = os.path.join(OUTPUT_DIR, "03_data_preprocessing.png")
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# Figure 4: Normal vs Abnormal Lung Sounds (Waveform + Spectrogram)
# ================================================================
def fig04_normal_vs_abnormal():
    fig = plt.figure(figsize=(22, 20))
    fig.suptitle("정상 vs 비정상 폐음 비교 - 원본 오디오", fontsize=22, fontweight="bold", y=0.98)

    # Load ICBHI diagnosis
    patient_diag = {}
    try:
        with open(ICBHI_DIAG) as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    patient_diag[row[0].strip()] = row[1].strip()
    except Exception:
        pass

    # Collect files by category
    categories = {
        "Healthy": [],
        "COPD": [],
        "Pneumonia": [],
    }

    if os.path.exists(ICBHI_AUDIO):
        for fname in os.listdir(ICBHI_AUDIO):
            if not fname.endswith(".wav"):
                continue
            pid = fname.split("_")[0]
            diag = patient_diag.get(pid, "")
            wav_path = os.path.join(ICBHI_AUDIO, fname)
            if diag in categories:
                categories[diag].append(wav_path)

    # Also collect SPRSound samples with specific event types
    spr_normal, spr_crackle, spr_wheeze = [], [], []
    if os.path.exists(SPR_TRAIN_JSON) and os.path.exists(SPR_TRAIN_WAV):
        for jf in os.listdir(SPR_TRAIN_JSON):
            if not jf.endswith(".json"):
                continue
            try:
                with open(os.path.join(SPR_TRAIN_JSON, jf)) as f:
                    annot = json.load(f)
                wav_name = jf.replace(".json", ".wav")
                wav_path = os.path.join(SPR_TRAIN_WAV, wav_name)
                if not os.path.exists(wav_path):
                    continue
                rec = annot.get("record_annotation", "")
                events = annot.get("event_annotation", [])
                event_types = [e.get("type", "") for e in events] if events else []

                if rec == "Normal":
                    spr_normal.append(wav_path)
                elif "Fine Crackle" in event_types or "Coarse Crackle" in event_types:
                    spr_crackle.append(wav_path)
                elif "Wheeze" in event_types:
                    spr_wheeze.append(wav_path)
            except Exception:
                continue

    rng = np.random.RandomState(42)

    samples = []
    # ICBHI samples
    if categories["Healthy"]:
        samples.append(("ICBHI - 정상 (Healthy)", rng.choice(categories["Healthy"]), C_NORM))
    if categories["COPD"]:
        samples.append(("ICBHI - COPD", rng.choice(categories["COPD"]), C_ABNORM))
    if categories["Pneumonia"]:
        samples.append(("ICBHI - 폐렴 (Pneumonia)", rng.choice(categories["Pneumonia"]), C_ORANGE))
    # SPRSound samples
    if spr_normal:
        samples.append(("SPRSound - 정상 (소아)", rng.choice(spr_normal), C_TEAL))
    if spr_crackle:
        samples.append(("SPRSound - Crackle (소아)", rng.choice(spr_crackle), C_PURPLE))
    if spr_wheeze:
        samples.append(("SPRSound - Wheeze (소아)", rng.choice(spr_wheeze), "#FF6F00"))

    n_rows = len(samples)
    if n_rows == 0:
        print("  [!] 폐음 파일을 찾을 수 없습니다")
        plt.close(fig)
        return

    gs = gridspec.GridSpec(n_rows, 3, hspace=0.5, wspace=0.3,
                           left=0.06, right=0.96, top=0.93, bottom=0.04,
                           width_ratios=[1.2, 1.2, 0.8])

    for row, (title, wav_path, color) in enumerate(samples):
        y_orig, sr_orig = librosa.load(wav_path, sr=None)
        duration = len(y_orig) / sr_orig

        # 5-second clip
        clip_sec = 5.0
        if duration > clip_sec + 1:
            start = int(sr_orig * min(2.0, duration / 3))
        else:
            start = 0
        end = start + int(sr_orig * clip_sec)
        y_clip = y_orig[start:end]
        if len(y_clip) < int(sr_orig * clip_sec):
            y_clip = np.pad(y_clip, (0, int(sr_orig * clip_sec) - len(y_clip)))

        t = np.arange(len(y_clip)) / sr_orig

        # Waveform
        ax_wave = fig.add_subplot(gs[row, 0])
        ax_wave.plot(t, y_clip, color=color, linewidth=0.4, alpha=0.8)
        ax_wave.set_title(title, fontsize=13, fontweight="bold", color=color, loc="left")
        ax_wave.set_ylabel("진폭")
        ax_wave.set_ylim(-1.05, 1.05)
        if row == n_rows - 1:
            ax_wave.set_xlabel("시간 (초)")

        fname = os.path.basename(wav_path)
        ax_wave.text(0.98, 0.95, f"{fname}\nSR={sr_orig}Hz, {duration:.1f}s",
                     transform=ax_wave.transAxes, fontsize=8, ha="right", va="top",
                     color="#666", style="italic")

        # Mel-Spectrogram
        ax_spec = fig.add_subplot(gs[row, 1])
        S = librosa.feature.melspectrogram(y=y_clip, sr=sr_orig, n_mels=128,
                                           fmax=min(sr_orig // 2, 4000), hop_length=256)
        S_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=sr_orig, x_axis="time", y_axis="mel",
                                 ax=ax_spec, fmax=min(sr_orig // 2, 4000), cmap="magma",
                                 hop_length=256)
        ax_spec.set_title(f"Mel-Spectrogram (SR={sr_orig}Hz)", fontsize=11, fontweight="bold")
        ax_spec.set_ylabel("주파수 (Hz)")
        if row == n_rows - 1:
            ax_spec.set_xlabel("시간 (초)")

        # FFT Frequency Spectrum
        ax_fft = fig.add_subplot(gs[row, 2])
        fft = np.abs(np.fft.rfft(y_clip))
        freqs = np.fft.rfftfreq(len(y_clip), d=1 / sr_orig)
        win = max(1, len(fft) // 200)
        fft_smooth = np.convolve(fft, np.ones(win) / win, mode="same")
        ax_fft.plot(freqs, 20 * np.log10(fft_smooth + 1e-10), color=color, linewidth=0.8)
        ax_fft.set_title("주파수 분포", fontsize=11, fontweight="bold")
        ax_fft.set_ylabel("dB")
        ax_fft.set_xlim(0, min(sr_orig // 2, 4000))
        if row == n_rows - 1:
            ax_fft.set_xlabel("주파수 (Hz)")

        # Respiratory frequency bands
        ax_fft.axvspan(100, 300, alpha=0.12, color="blue")
        ax_fft.text(200, ax_fft.get_ylim()[1] - 3, "호흡음\n대역", fontsize=7,
                    ha="center", color="blue", fontweight="bold")
        ax_fft.axvspan(300, 1200, alpha=0.08, color="red")
        ax_fft.text(750, ax_fft.get_ylim()[1] - 3, "Wheeze/\nCrackle", fontsize=7,
                    ha="center", color="red", fontweight="bold")

    path = os.path.join(OUTPUT_DIR, "04_normal_vs_abnormal.png")
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# Figure 5: Architecture Comparison (ResNet18 vs MobileNetV2)
# ================================================================
def fig05_architecture():
    fig = plt.figure(figsize=(24, 14))
    fig.suptitle("아키텍처 비교: ResNet18 vs MobileNetV2", fontsize=22, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(1, 2, wspace=0.15, left=0.04, right=0.96, top=0.90, bottom=0.04)

    # ── ResNet18 ──
    ax_res = fig.add_subplot(gs[0, 0])
    ax_res.set_xlim(0, 1)
    ax_res.set_ylim(0, 1)
    ax_res.axis("off")
    ax_res.set_title("ResNet18 (Phase 1-3)", fontsize=16, fontweight="bold", color=C_GRAY)

    resnet_layers = [
        ("Input\n(1ch or 3ch)", "#BBDEFB", 0.88),
        ("Conv 7x7, 64\nBN + ReLU + MaxPool", "#90CAF9", 0.76),
        ("Layer 1\n2x BasicBlock (64)", "#64B5F6", 0.64),
        ("Layer 2\n2x BasicBlock (128)", "#42A5F5", 0.52),
        ("Layer 3\n2x BasicBlock (256)", "#2196F3", 0.40),
        ("Layer 4\n2x BasicBlock (512)", "#1E88E5", 0.28),
        ("AdaptiveAvgPool\n+ FC(512 -> classes)", "#1565C0", 0.16),
    ]
    for label, clr, y in resnet_layers:
        _rounded_box(ax_res, 0.15, y, 0.7, 0.09, label, clr, fontsize=9, text_color="white")
        if y > 0.16:
            ax_res.annotate("", xy=(0.5, y), xytext=(0.5, y + 0.09 - 0.001),
                            arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.5))

    # Skip connection annotation
    ax_res.annotate("skip\nconnection", xy=(0.88, 0.50), fontsize=8, ha="center",
                    color=C_BLUE, fontweight="bold")
    ax_res.annotate("", xy=(0.87, 0.45), xytext=(0.87, 0.70),
                    arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=1.5,
                                    connectionstyle="arc3,rad=-0.3"))

    # Stats
    ax_res.text(0.5, 0.06, "Params: ~11.2M  |  FLOPs: ~1.8G  |  무거움",
                ha="center", fontsize=10, color=C_GRAY, style="italic")

    # ── MobileNetV2 ──
    ax_mob = fig.add_subplot(gs[0, 1])
    ax_mob.set_xlim(0, 1)
    ax_mob.set_ylim(0, 1)
    ax_mob.axis("off")
    ax_mob.set_title("MobileNetV2 (Final v1-v2)", fontsize=16, fontweight="bold", color=C_ABNORM)

    mob_layers = [
        ("Input\n224x224x3 (RGB)", "#FFCDD2", 0.86),
        ("Conv 3x3, 32\nBN + ReLU6", "#EF9A9A", 0.76),
        ("Inverted Residual Blocks\nt=[1,6,6,...] c=[16,24,32,64,96,160,320]", "#E57373", 0.60),
        ("Depthwise Separable Conv\n= Depthwise + Pointwise", "#EF5350", 0.48),
        ("Conv 1x1, 1280\nBN + ReLU6", "#E53935", 0.38),
        ("AdaptiveAvgPool\n+ Dropout(0.2)", "#C62828", 0.28),
        ("FC(1280 -> 5 classes)", "#B71C1C", 0.18),
    ]
    for label, clr, y in mob_layers:
        _rounded_box(ax_mob, 0.15, y, 0.7, 0.09, label, clr, fontsize=9, text_color="white")
        if y > 0.18:
            next_y = [ly for _, _, ly in mob_layers if ly < y]
            if next_y:
                ax_mob.annotate("", xy=(0.5, max(next_y) + 0.09),
                                xytext=(0.5, y - 0.001),
                                arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.5))

    # Inverted residual detail
    ax_mob.annotate("Linear\nBottleneck", xy=(0.88, 0.62), fontsize=8, ha="center",
                    color=C_ABNORM, fontweight="bold")

    ax_mob.text(0.5, 0.08, "Params: ~2.2M  |  FLOPs: ~0.3G  |  경량 + 효율적",
                ha="center", fontsize=10, color=C_ABNORM, fontweight="bold", style="italic")

    # Bottom comparison
    fig.text(0.5, 0.01,
             "MobileNetV2는 ResNet18 대비 파라미터 80% 감소, FLOPs 83% 감소하면서 "
             "ImageNet pretrained features를 효과적으로 transfer -> ESP32 Edge 배포에 적합",
             ha="center", fontsize=12, style="italic", color="#555")

    path = os.path.join(OUTPUT_DIR, "05_architecture_comparison.png")
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# Figure 6: Key Improvements Summary
# ================================================================
def fig06_improvements():
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle("핵심 개선 사항 요약", fontsize=22, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.25,
                           left=0.06, right=0.96, top=0.90, bottom=0.06)

    # (0,0) Resolution: 128 vs 224
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title("1. 해상도 향상", fontsize=15, fontweight="bold", color=C_BLUE)

    # 128x128 box
    _rounded_box(ax1, 0.05, 0.4, 0.35, 0.35, "128x128\npixels", "#90CAF9", fontsize=12)
    _arrow_right(ax1, 0.42, 0.575, 0.55)
    # 224x224 box (bigger)
    _rounded_box(ax1, 0.55, 0.3, 0.42, 0.45, "224x224\npixels", C_BLUE, fontsize=14)

    ax1.text(0.5, 0.18, "해상도 3배 증가\n세밀한 주파수 패턴 포착",
             ha="center", fontsize=10, color="#555", style="italic")
    ax1.text(0.5, 0.05, "높은 해상도 = 더 정확한 특징 추출",
             ha="center", fontsize=9, fontweight="bold", color=C_BLUE)

    # (0,1) Loss: CE vs Focal
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("2. Focal Loss 도입", fontsize=15, fontweight="bold", color=C_ABNORM)

    # Plot Focal Loss curve
    p = np.linspace(0.01, 0.99, 200)
    ce_loss = -np.log(p)
    focal_loss_g1 = -(1 - p) ** 1 * np.log(p)
    focal_loss_g2 = -(1 - p) ** 2 * np.log(p)
    focal_loss_g5 = -(1 - p) ** 5 * np.log(p)

    ax2.plot(p, ce_loss, color=C_GRAY, linewidth=2, label="CE Loss (gamma=0)")
    ax2.plot(p, focal_loss_g1, color="#FF9800", linewidth=2, label="Focal (gamma=1)")
    ax2.plot(p, focal_loss_g2, color=C_ABNORM, linewidth=3, label="Focal (gamma=2) *사용*")
    ax2.plot(p, focal_loss_g5, color=C_PURPLE, linewidth=1.5, linestyle="--", label="Focal (gamma=5)")
    ax2.set_xlabel("예측 확률 (pt)", fontsize=10)
    ax2.set_ylabel("Loss", fontsize=10)
    ax2.set_ylim(0, 5)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.text(0.3, 3.5, "소수 클래스에\n더 집중!", fontsize=11,
             fontweight="bold", color=C_ABNORM)

    # (0,2) Data unification
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title("3. 데이터 통합 & 밸런싱", fontsize=15, fontweight="bold", color=C_TEAL)

    timeline = [
        ("Phase 1", "2022만", "#BBDEFB"),
        ("Phase 2", "2022+2023", "#90CAF9"),
        ("Phase 3", "2022-2025", "#42A5F5"),
        ("Final", "ICBHI + SPRSound\n성인 + 소아 통합", C_TEAL),
    ]
    for i, (phase, data, clr) in enumerate(timeline):
        y = 0.82 - i * 0.18
        w = 0.3 + i * 0.12
        _rounded_box(ax3, 0.5 - w / 2, y, w, 0.12, f"{phase}: {data}", clr,
                     fontsize=9, text_color="white")

    ax3.text(0.5, 0.1, "더 많은 데이터 + 더 다양한 소스\n= 일반화 성능 향상",
             ha="center", fontsize=10, color="#555", style="italic")

    # (1,0) Pretrained vs Scratch
    ax4 = fig.add_subplot(gs[1, 0])
    labels = ["Scratch\n(Phase 1)", "Pretrained\n(Phase 3)", "MobileNetV2\n(Final)"]
    vals = [65, 78, 88]  # approximate accuracy values
    colors_bar = [C_GRAY, C_BLUE, C_ABNORM]
    bars = ax4.bar(range(3), vals, color=colors_bar, edgecolor="white", width=0.6)
    ax4.set_xticks(range(3))
    ax4.set_xticklabels(labels, fontsize=10)
    ax4.set_ylabel("대략적 Accuracy (%)")
    ax4.set_ylim(50, 100)
    ax4.set_title("4. Transfer Learning 효과", fontsize=15, fontweight="bold", color="#333")
    for bar, val in zip(bars, vals):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"~{val}%", ha="center", fontsize=11, fontweight="bold")

    # (1,1) Augmentation techniques
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis("off")
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.set_title("5. 증강 기법 변천", fontsize=15, fontweight="bold", color=C_ORANGE)

    augs = [
        ("Phase 1", "증강 없음", C_GRAY),
        ("Phase 3", "SpecAugment\n(FreqMask=20, TimeMask=40)", "#42A5F5"),
        ("Final", "Gaussian Noise\n+ Pitch Shift (+-2 semitones)\n+ 클래스별 oversampling", C_ORANGE),
    ]
    for i, (phase, desc, clr) in enumerate(augs):
        y = 0.78 - i * 0.28
        ax5.text(0.12, y, phase, fontsize=12, fontweight="bold", color=clr, va="center")
        ax5.text(0.32, y, desc, fontsize=10, va="center", color="#333")
        if i < len(augs) - 1:
            ax5.annotate("", xy=(0.12, y - 0.12), xytext=(0.12, y - 0.04),
                         arrowprops=dict(arrowstyle="-|>", color="#999", lw=1.5))

    # (1,2) Summary table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_title("6. 최종 비교표", fontsize=15, fontweight="bold", color="#333")

    table_data = [
        ("항목", "Phase 1", "Final v2"),
        ("Architecture", "ResNet18", "MobileNetV2"),
        ("Weights", "Scratch", "Pretrained"),
        ("Resolution", "-", "224x224"),
        ("Loss", "CE", "Focal (g=2)"),
        ("Data", "2022", "2022-2025"),
        ("Classes", "Multi", "5-class"),
        ("Augmentation", "None", "Noise+Pitch"),
        ("Scheduler", "None", "ReduceLR"),
        ("Early Stop", "No", "Yes (5)"),
    ]

    for i, row_data in enumerate(table_data):
        y = 0.92 - i * 0.09
        bg = "#E3F2FD" if i == 0 else ("#F5F5F5" if i % 2 == 0 else "white")
        fontw = "bold" if i == 0 else "normal"

        ax6.fill_between([0, 1], y - 0.04, y + 0.04, color=bg, alpha=0.8)
        ax6.text(0.02, y, row_data[0], fontsize=9, fontweight=fontw, va="center")
        ax6.text(0.38, y, row_data[1], fontsize=9, fontweight=fontw, va="center",
                 color=C_GRAY if i > 0 else "#333", ha="center")
        ax6.text(0.75, y, row_data[2], fontsize=9, fontweight=fontw, va="center",
                 color=C_ABNORM if i > 0 else "#333", ha="center")

    path = os.path.join(OUTPUT_DIR, "06_improvements_summary.png")
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 60)
    print(" 폐음 분류 시각화 생성")
    print("=" * 60)

    print("\n[1] Figure 01: 분류 파이프라인...")
    fig01_pipeline()

    print("\n[2] Figure 02: 모델 진화 히스토리...")
    fig02_evolution()

    print("\n[3] Figure 03: 데이터 구성 & 전처리...")
    fig03_data_preprocessing()

    print("\n[4] Figure 04: 정상 vs 비정상 폐음 비교...")
    fig04_normal_vs_abnormal()

    print("\n[5] Figure 05: 아키텍처 비교...")
    fig05_architecture()

    print("\n[6] Figure 06: 핵심 개선 요약...")
    fig06_improvements()

    print("\n" + "=" * 60)
    print(f" 완료! 위치: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
