"""
Shannon Envelope 기반 S1 자동 라벨링
=====================================
PhysioNet heart sound dataset (2kHz) → Shannon energy envelope → S1 피크 검출
→ 200ms 윈도우로 beat/no-beat 라벨 생성 → .npy 저장

사용법:
  C:/Users/dwd00/anaconda3/envs/stetho_ai/python.exe 01_shannon_labeling.py
"""

import os
import glob
import numpy as np
import librosa
from scipy.signal import butter, filtfilt, find_peaks
from tqdm import tqdm

# ─── 설정 ───────────────────────────────────────────────
DATA_ROOT = "G:/stetho_ai/_misc/datasets/classification-of-heart-sound-recordings/classification-of-heart-sound-recordings"
TRAINING_DIRS = ["training-a", "training-b", "training-c", "training-d", "training-e", "training-f"]

TARGET_SR = 2000          # 원본 그대로 (이미 2kHz)
WINDOW_SEC = 0.2          # 200ms 윈도우
WINDOW_SAMPLES = int(TARGET_SR * WINDOW_SEC)  # 400 samples
HOP_SEC = 0.05            # 50ms hop (윈도우 간 이동)
HOP_SAMPLES = int(TARGET_SR * HOP_SEC)        # 100 samples

# Shannon envelope 파라미터
BP_LOW = 20.0             # bandpass 하한 (Hz)
BP_HIGH = 400.0           # bandpass 상한 (Nyquist=1000Hz이므로 400 적절)
SMOOTH_WINDOW_MS = 40     # smoothing window (ms)
SMOOTH_SAMPLES = int(TARGET_SR * SMOOTH_WINDOW_MS / 1000)  # 80 samples

# 피크 검출 파라미터
MIN_PEAK_DISTANCE_MS = 300   # S1-S1 최소 간격 (200bpm 상한)
MIN_PEAK_DISTANCE = int(TARGET_SR * MIN_PEAK_DISTANCE_MS / 1000)
BEAT_TOLERANCE_MS = 80       # 피크 ±80ms 이내 = beat 라벨
BEAT_TOLERANCE = int(TARGET_SR * BEAT_TOLERANCE_MS / 1000)

OUTPUT_DIR = "G:/stetho_ai/BPM_ondevice/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def bandpass_filter(signal, sr, low=BP_LOW, high=BP_HIGH, order=4):
    """Butterworth bandpass filter."""
    nyq = sr / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)


def shannon_envelope(signal):
    """
    Shannon energy envelope: -x^2 * log(x^2)
    작은 진폭은 억제, 큰 진폭(S1/S2)은 강조.
    """
    # normalize to [-1, 1]
    max_val = np.max(np.abs(signal))
    if max_val < 1e-10:
        return np.zeros_like(signal)
    x = signal / max_val

    # avoid log(0) — clip 아주 작은 값
    x_sq = np.clip(x ** 2, 1e-12, 1.0)
    env = -x_sq * np.log(x_sq)

    # smoothing (moving average)
    if SMOOTH_SAMPLES > 1:
        kernel = np.ones(SMOOTH_SAMPLES) / SMOOTH_SAMPLES
        env = np.convolve(env, kernel, mode='same')

    return env


def detect_s1_peaks(envelope, sr):
    """
    Shannon envelope에서 S1 피크를 검출.
    adaptive threshold: median + 0.5 * (max - median)
    """
    # adaptive threshold
    med = np.median(envelope)
    mx = np.max(envelope)
    threshold = med + 0.4 * (mx - med)

    peaks, properties = find_peaks(
        envelope,
        height=threshold,
        distance=MIN_PEAK_DISTANCE,
        prominence=0.1 * mx  # 최소 prominence
    )
    return peaks


def extract_windows(audio, sr, s1_peaks):
    """
    200ms 윈도우를 hop_sec 간격으로 추출.
    S1 피크 ±tolerance 이내에 중심이 있으면 beat=1, 아니면 beat=0.
    """
    n_samples = len(audio)
    windows = []
    labels = []

    center = 0
    while center + WINDOW_SAMPLES <= n_samples:
        start = center
        end = center + WINDOW_SAMPLES
        window = audio[start:end]

        # beat 판정: 윈도우 중심이 S1 피크 ±tolerance 이내인가?
        win_center = center + WINDOW_SAMPLES // 2
        is_beat = 0
        for peak in s1_peaks:
            if abs(win_center - peak) <= BEAT_TOLERANCE:
                is_beat = 1
                break

        windows.append(window)
        labels.append(is_beat)
        center += HOP_SAMPLES

    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int64)


def process_file(wav_path):
    """단일 WAV 파일 처리: Shannon envelope → S1 검출 → 윈도우 추출."""
    try:
        y, sr = librosa.load(wav_path, sr=TARGET_SR)
    except Exception as e:
        print(f"  [SKIP] {wav_path}: {e}")
        return None, None, 0

    if len(y) < WINDOW_SAMPLES:
        return None, None, 0

    # 1) bandpass filter
    filtered = bandpass_filter(y, sr)

    # 2) Shannon envelope
    env = shannon_envelope(filtered)

    # 3) S1 피크 검출
    peaks = detect_s1_peaks(env, sr)

    if len(peaks) < 2:
        return None, None, 0

    # 4) 윈도우 추출 + 라벨링
    windows, labels = extract_windows(y, sr, peaks)

    return windows, labels, len(peaks)


def main():
    all_windows = []
    all_labels = []
    total_files = 0
    skipped = 0
    total_peaks = 0

    for train_dir in TRAINING_DIRS:
        dir_path = os.path.join(DATA_ROOT, train_dir)
        wav_files = sorted(glob.glob(os.path.join(dir_path, "*.wav")))
        print(f"\n[{train_dir}] {len(wav_files)} WAV files")

        for wav_path in tqdm(wav_files, desc=train_dir):
            total_files += 1
            windows, labels, n_peaks = process_file(wav_path)

            if windows is None:
                skipped += 1
                continue

            all_windows.append(windows)
            all_labels.append(labels)
            total_peaks += n_peaks

    # 합치기
    X = np.concatenate(all_windows, axis=0)
    Y = np.concatenate(all_labels, axis=0)

    # 통계
    n_beat = np.sum(Y == 1)
    n_nobeat = np.sum(Y == 0)
    print(f"\n{'='*50}")
    print(f"총 파일: {total_files}, 스킵: {skipped}")
    print(f"총 S1 피크 검출: {total_peaks}")
    print(f"총 윈도우: {len(X)}")
    print(f"  beat: {n_beat} ({100*n_beat/len(X):.1f}%)")
    print(f"  no-beat: {n_nobeat} ({100*n_nobeat/len(X):.1f}%)")
    print(f"윈도우 shape: {X.shape}")
    print(f"{'='*50}")

    # 저장
    np.save(os.path.join(OUTPUT_DIR, "X_windows.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "Y_labels.npy"), Y)
    print(f"\n저장 완료: {OUTPUT_DIR}/X_windows.npy, Y_labels.npy")


if __name__ == "__main__":
    main()
