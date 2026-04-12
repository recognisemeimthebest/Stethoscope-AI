"""
Shannon Envelope 라벨링 품질 시각 검증
======================================
랜덤 WAV 파일 5개를 뽑아서:
  - 원본 파형
  - Shannon envelope
  - 검출된 S1 피크 (빨간 세로선)
  - beat 윈도우 (초록 음영)
를 그려서 PNG로 저장.

사용법:
  C:/Users/dwd00/anaconda3/envs/stetho_ai/python.exe 01b_verify_labels.py
"""

import os
import glob
import random
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 01_shannon_labeling.py에서 함수 임포트
from importlib.machinery import SourceFileLoader
labeler = SourceFileLoader("labeler", "G:/stetho_ai/BPM_ondevice/01_shannon_labeling.py").load_module()

DATA_ROOT = labeler.DATA_ROOT
TRAINING_DIRS = labeler.TRAINING_DIRS
TARGET_SR = labeler.TARGET_SR
WINDOW_SAMPLES = labeler.WINDOW_SAMPLES
BEAT_TOLERANCE = labeler.BEAT_TOLERANCE

OUTPUT_DIR = "G:/stetho_ai/BPM_ondevice/verify"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SAMPLES = 5  # 검증할 파일 수
PLOT_SEC = 8.0  # 앞 8초만 표시


def verify_one(wav_path, idx):
    y, sr = librosa.load(wav_path, sr=TARGET_SR)

    # 앞 PLOT_SEC 초만
    n_plot = int(sr * PLOT_SEC)
    if len(y) > n_plot:
        y_plot = y[:n_plot]
    else:
        y_plot = y
        n_plot = len(y)

    # Shannon envelope
    filtered = labeler.bandpass_filter(y_plot, sr)
    env = labeler.shannon_envelope(filtered)
    peaks = labeler.detect_s1_peaks(env, sr)

    t = np.arange(n_plot) / sr

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fname = os.path.basename(wav_path)

    # 상단: 원본 파형 + S1 피크 + beat 윈도우
    ax1 = axes[0]
    ax1.plot(t, y_plot, color='#377eb8', linewidth=0.5, alpha=0.8)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"[{idx+1}] {fname} — Waveform + S1 Peaks")

    for peak in peaks:
        peak_t = peak / sr
        ax1.axvline(peak_t, color='#e41a1c', linewidth=1.2, alpha=0.7)
        # beat tolerance 영역 (초록 음영)
        tol_start = max(0, peak - BEAT_TOLERANCE) / sr
        tol_end = min(n_plot, peak + BEAT_TOLERANCE) / sr
        ax1.axvspan(tol_start, tol_end, color='#4daf4a', alpha=0.15)

    # 하단: Shannon envelope + threshold
    ax2 = axes[1]
    ax2.plot(t, env, color='#ff7f00', linewidth=0.8)
    ax2.set_ylabel("Shannon Energy")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Shannon Envelope + Threshold")

    # threshold 표시
    med = np.median(env)
    mx = np.max(env)
    threshold = med + 0.4 * (mx - med)
    ax2.axhline(threshold, color='#984ea3', linewidth=1, linestyle='--', label=f'threshold={threshold:.4f}')
    ax2.legend(loc='upper right')

    for peak in peaks:
        ax2.axvline(peak / sr, color='#e41a1c', linewidth=1.2, alpha=0.7)

    # BPM 계산 (피크 간격)
    if len(peaks) >= 2:
        intervals = np.diff(peaks) / sr  # seconds
        avg_bpm = 60.0 / np.mean(intervals)
        fig.suptitle(f"{fname} | Detected S1: {len(peaks)} | Est. BPM: {avg_bpm:.0f}", fontsize=13, fontweight='bold')
    else:
        fig.suptitle(f"{fname} | Detected S1: {len(peaks)} | BPM: N/A", fontsize=13)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"verify_{idx+1}_{fname.replace('.wav','')}.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    # 모든 WAV 수집
    all_wavs = []
    for d in TRAINING_DIRS:
        all_wavs.extend(sorted(glob.glob(os.path.join(DATA_ROOT, d, "*.wav"))))

    print(f"총 WAV: {len(all_wavs)}")

    # 랜덤 N개 선택
    random.seed(42)
    samples = random.sample(all_wavs, min(N_SAMPLES, len(all_wavs)))

    for i, wav_path in enumerate(samples):
        print(f"\n[{i+1}/{N_SAMPLES}] {wav_path}")
        try:
            verify_one(wav_path, i)
        except Exception as e:
            print(f"  [ERROR] {e}")


if __name__ == "__main__":
    main()
