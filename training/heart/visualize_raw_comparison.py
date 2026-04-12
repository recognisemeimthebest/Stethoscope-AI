"""
================================================================
정상 vs 비정상 심음 비교 — 원본 wav에서 직접 시각화
================================================================
파형 + 스펙트로그램을 원본 SR로 그려서 자연스러운 주파수 대역 표현
성인(PhysioNet) + 소아(Pediatric) 각 정상/비정상 1쌍씩
================================================================
"""

import os
import csv
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = r"G:\stetho_ai\Heart_binary_classification\figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ADULT_BASE = r"G:\stetho_ai\_misc\datasets\classification-of-heart-sound-recordings\classification-of-heart-sound-recordings"
PED_BASE = r"G:\stetho_ai\_misc\datasets\pediatric\cleaned_data"
PED_CSV = r"G:\stetho_ai\_misc\datasets\pediatric\labeled_dataset.csv"

C_NORM = "#43A047"
C_ABNORM = "#E53935"


def get_adult_files():
    """성인 데이터에서 정상/비정상 파일 수집"""
    normals, abnormals = [], []
    for folder in ["training-a", "training-b", "training-c", "training-d", "training-e", "training-f"]:
        ref_path = os.path.join(ADULT_BASE, folder, "REFERENCE.csv")
        if not os.path.exists(ref_path):
            continue
        with open(ref_path) as f:
            for row in csv.reader(f):
                wav_path = os.path.join(ADULT_BASE, folder, row[0] + ".wav")
                if row[1] == "-1":
                    normals.append(wav_path)
                elif row[1] == "1":
                    abnormals.append(wav_path)
    return normals, abnormals


def get_ped_files():
    """소아 데이터에서 정상/비정상 파일 수집"""
    normals, abnormals = [], []
    import pandas as pd
    df = pd.read_csv(PED_CSV)
    for _, row in df.iterrows():
        wav_path = os.path.join(PED_BASE, row["File_Name"])
        if not os.path.exists(wav_path):
            continue
        if row["Murmur"] == "Absent":
            normals.append(wav_path)
        elif row["Murmur"] == "Present":
            abnormals.append(wav_path)
    return normals, abnormals


# ================================================================
# Figure 4 (대체): 원본 wav 정상 vs 비정상 (파형 + 스펙트로그램)
# ================================================================
def fig4_raw_comparison():
    print("  파일 수집 중...")
    adult_norm, adult_abnorm = get_adult_files()
    ped_norm, ped_abnorm = get_ped_files()

    # 적절한 길이의 샘플 선택 (너무 짧지 않은 것)
    rng = np.random.RandomState(42)

    samples = [
        ("성인 정상", rng.choice(adult_norm), C_NORM),
        ("성인 비정상 (심잡음)", rng.choice(adult_abnorm), C_ABNORM),
        ("소아 정상", rng.choice(ped_norm), C_NORM),
        ("소아 비정상 (심잡음)", rng.choice(ped_abnorm), C_ABNORM),
    ]

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle("정상 vs 비정상 심음 비교 - 원본 오디오", fontsize=22, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(4, 3, hspace=0.5, wspace=0.3,
                           left=0.06, right=0.96, top=0.93, bottom=0.04,
                           width_ratios=[1.2, 1.2, 0.8])

    for row, (title, wav_path, color) in enumerate(samples):
        # 원본 SR로 로드
        y_orig, sr_orig = librosa.load(wav_path, sr=None)
        duration = len(y_orig) / sr_orig

        # 5초 구간 추출 (중간쯤에서)
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

        # ── 파형 ──
        ax_wave = fig.add_subplot(gs[row, 0])
        ax_wave.plot(t, y_clip, color=color, linewidth=0.4, alpha=0.8)
        ax_wave.set_title(f"{title}", fontsize=14, fontweight="bold", color=color, loc="left")
        ax_wave.set_ylabel("진폭")
        ax_wave.set_ylim(-1.05, 1.05)
        if row == 3:
            ax_wave.set_xlabel("시간 (초)")

        # 파일 정보 표시
        fname = os.path.basename(wav_path)
        ax_wave.text(0.98, 0.95, f"{fname}\nSR={sr_orig}Hz, {duration:.1f}s",
                     transform=ax_wave.transAxes, fontsize=8, ha="right", va="top",
                     color="#666", style="italic")

        # ── 스펙트로그램 (원본 SR 그대로) ──
        ax_spec = fig.add_subplot(gs[row, 1])
        S = librosa.feature.melspectrogram(y=y_clip, sr=sr_orig, n_mels=128,
                                           fmax=sr_orig // 2, hop_length=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=sr_orig, x_axis="time", y_axis="mel",
                                 ax=ax_spec, fmax=sr_orig // 2, cmap="magma",
                                 hop_length=128)
        ax_spec.set_title(f"Mel-Spectrogram (SR={sr_orig}Hz)", fontsize=12, fontweight="bold")
        ax_spec.set_ylabel("주파수 (Hz)")
        if row == 3:
            ax_spec.set_xlabel("시간 (초)")

        # ── 주파수 스펙트럼 (FFT) ──
        ax_fft = fig.add_subplot(gs[row, 2])
        fft = np.abs(np.fft.rfft(y_clip))
        freqs = np.fft.rfftfreq(len(y_clip), d=1/sr_orig)
        # 스무딩
        win = max(1, len(fft) // 200)
        fft_smooth = np.convolve(fft, np.ones(win)/win, mode="same")
        ax_fft.plot(freqs, 20*np.log10(fft_smooth + 1e-10), color=color, linewidth=0.8)
        ax_fft.set_title("주파수 분포", fontsize=12, fontweight="bold")
        ax_fft.set_ylabel("dB")
        ax_fft.set_xlim(0, sr_orig // 2)
        if row == 3:
            ax_fft.set_xlabel("주파수 (Hz)")

        # 심음 대역 표시
        ax_fft.axvspan(20, 150, alpha=0.15, color="blue")
        ax_fft.text(85, ax_fft.get_ylim()[1] - 3, "S1/S2", fontsize=8,
                    ha="center", color="blue", fontweight="bold")

    path = os.path.join(OUTPUT_DIR, "04_normal_vs_abnormal.png")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# Figure 4b: 같은 환자 여러 위치 비교 (소아, 심잡음 패턴 차이)
# ================================================================
def fig4b_multisite_comparison():
    """소아 데이터에서 같은 환자의 AV/MV/PV/TV 위치별 비교"""
    import pandas as pd
    df = pd.read_csv(PED_CSV)

    # 비정상 환자 중 4개 위치 모두 있는 환자 찾기
    abnorm = df[df["Murmur"] == "Present"]
    patients = abnorm["Patient_ID"].value_counts()
    multi_patients = patients[patients >= 4].index

    if len(multi_patients) == 0:
        print("  [!] 4개 위치 모두 있는 환자 없음, 건너뜀")
        return

    patient_id = int(multi_patients[0])
    patient_files = abnorm[abnorm["Patient_ID"] == patient_id]

    locations = ["AV", "MV", "PV", "TV"]
    loc_names = {
        "AV": "대동맥판 (Aortic)",
        "MV": "승모판 (Mitral)",
        "PV": "폐동맥판 (Pulmonary)",
        "TV": "삼첨판 (Tricuspid)",
    }

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f"같은 환자(ID: {patient_id})의 청진 위치별 심음 비교 - 비정상(심잡음)",
                 fontsize=18, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(4, 2, hspace=0.45, wspace=0.25,
                           left=0.06, right=0.96, top=0.92, bottom=0.05)

    colors_loc = ["#1565C0", "#E65100", "#2E7D32", "#7B1FA2"]

    for i, loc in enumerate(locations):
        row = patient_files[patient_files["Location"] == loc]
        if len(row) == 0:
            continue
        fname = row.iloc[0]["File_Name"]
        wav_path = os.path.join(PED_BASE, fname)
        if not os.path.exists(wav_path):
            continue

        y, sr = librosa.load(wav_path, sr=None)
        # 5초 클립
        clip_len = min(len(y), int(sr * 5))
        y_clip = y[:clip_len]
        t = np.arange(len(y_clip)) / sr
        color = colors_loc[i]

        # 파형
        ax_wave = fig.add_subplot(gs[i, 0])
        ax_wave.plot(t, y_clip, color=color, linewidth=0.5, alpha=0.8)
        ax_wave.set_title(f"{loc_names.get(loc, loc)} ({loc})", fontsize=13,
                          fontweight="bold", color=color, loc="left")
        ax_wave.set_ylabel("진폭")
        ax_wave.set_ylim(-1.05, 1.05)
        if i == 3:
            ax_wave.set_xlabel("시간 (초)")

        # 스펙트로그램
        ax_spec = fig.add_subplot(gs[i, 1])
        S = librosa.feature.melspectrogram(y=y_clip, sr=sr, n_mels=128,
                                           fmax=sr // 2, hop_length=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel",
                                 ax=ax_spec, fmax=sr // 2, cmap="magma",
                                 hop_length=128)
        ax_spec.set_title(f"Mel-Spectrogram", fontsize=12, fontweight="bold")
        if i == 3:
            ax_spec.set_xlabel("시간 (초)")

    # 하단 설명
    fig.text(0.5, 0.01,
             "같은 심잡음이라도 청진 위치(AV/MV/PV/TV)에 따라 에너지 패턴이 다름 "
             "-> 위치 정보를 활용하면 심잡음 유형 세분화 가능",
             ha="center", fontsize=12, style="italic", color="#555")

    path = os.path.join(OUTPUT_DIR, "04b_multisite_comparison.png")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  저장: {path}")


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 60)
    print(" 원본 심음 정상/비정상 비교 시각화")
    print("=" * 60)

    print("\n[1] Figure 4: 원본 wav 정상 vs 비정상...")
    fig4_raw_comparison()

    print("\n[2] Figure 4b: 위치별 비교...")
    fig4b_multisite_comparison()

    print("\n" + "=" * 60)
    print(f" 완료! 위치: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
