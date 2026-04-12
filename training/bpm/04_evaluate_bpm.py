"""
BPM 추정 성능 평가
==================
실제 WAV 파일에서:
  1) Shannon envelope GT BPM (라벨링 기준)
  2) ML model predicted BPM (beat_cnn_predict 시뮬레이션)
  3) DSP peak detection BPM (현재 펌웨어 방식 시뮬레이션)
3가지를 비교.

사용법:
  C:/Users/dwd00/anaconda3/envs/stetho_ai/python.exe 04_evaluate_bpm.py
"""

import os
import glob
import random
import numpy as np
import torch
import librosa
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

# Shannon labeling 함수 재사용
from importlib.machinery import SourceFileLoader
labeler = SourceFileLoader("labeler", "G:/stetho_ai/BPM_ondevice/01_shannon_labeling.py").load_module()

# PyTorch 모델 로드
trainer = SourceFileLoader("trainer", "G:/stetho_ai/BPM_ondevice/02_train_beat_detector.py").load_module()

DATA_ROOT = labeler.DATA_ROOT
TRAINING_DIRS = labeler.TRAINING_DIRS
TARGET_SR = 2000
WINDOW_SAMPLES = 400  # 200ms
HOP_SAMPLES = 100     # 50ms hop

MODEL_PATH = "G:/stetho_ai/BPM_ondevice/models/beat_cnn_best.pth"
OUTPUT_DIR = "G:/stetho_ai/BPM_ondevice/eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = trainer.TinyBeatCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def ml_detect_beats(audio, model):
    """ML 모델로 beat 감지 -> beat 타임스탬프 반환."""
    n = len(audio)
    probs = []
    centers = []

    # 배치 처리
    windows = []
    for start in range(0, n - WINDOW_SAMPLES + 1, HOP_SAMPLES):
        windows.append(audio[start:start + WINDOW_SAMPLES])
        centers.append(start + WINDOW_SAMPLES // 2)

    if not windows:
        return np.array([]), np.array([])

    X = np.array(windows, dtype=np.float32)
    X_t = torch.from_numpy(X).unsqueeze(1).to(DEVICE)  # (N, 1, 400)

    with torch.no_grad():
        logits = model(X_t)
        prob = torch.sigmoid(logits).cpu().numpy()

    probs = prob
    centers = np.array(centers)

    # threshold + NMS (non-max suppression via min distance)
    beat_mask = probs > 0.5
    beat_centers = centers[beat_mask]
    beat_probs = probs[beat_mask]

    if len(beat_centers) == 0:
        return np.array([]), probs

    # NMS: 300ms 이내 중복 제거 (가장 높은 prob 유지)
    min_dist = int(TARGET_SR * 0.3)  # 300ms
    final_beats = [beat_centers[0]]
    for i in range(1, len(beat_centers)):
        if beat_centers[i] - final_beats[-1] >= min_dist:
            final_beats.append(beat_centers[i])

    return np.array(final_beats), probs


def dsp_detect_beats(audio, sr):
    """현재 펌웨어와 유사한 DSP peak detection (envelope threshold)."""
    # Bandpass
    nyq = sr / 2.0
    b, a = butter(4, [20.0 / nyq, min(400.0, nyq * 0.95) / nyq], btype='band')
    filtered = filtfilt(b, a, audio)

    # Simple envelope (absolute + smoothing)
    env = np.abs(filtered)
    kernel = np.ones(int(sr * 0.02)) / int(sr * 0.02)  # 20ms smoothing
    env = np.convolve(env, kernel, mode='same')

    # Running max (decay)
    running_max = np.zeros_like(env)
    rm = 0.0
    for i in range(len(env)):
        rm = max(env[i], rm * 0.9998)
        running_max[i] = rm

    # Threshold: 35% of running max
    threshold = running_max * 0.35
    threshold = np.maximum(threshold, 0.0005)  # noise floor

    # Peak detection with refractory period
    refractory = int(sr * 0.3)  # 300ms = 2400 samples at 8kHz -> 600 at 2kHz
    peaks = []
    last_peak = -refractory
    for i in range(1, len(env) - 1):
        if (env[i] > threshold[i] and
            env[i] > env[i-1] and env[i] > env[i+1] and
            (i - last_peak) >= refractory):
            peaks.append(i)
            last_peak = i

    return np.array(peaks)


def beats_to_bpm(beats, sr):
    """Beat 타임스탬프 -> BPM (median interval)."""
    if len(beats) < 2:
        return 0.0
    intervals = np.diff(beats) / sr  # seconds
    # 이상치 제거 (IQR)
    q1, q3 = np.percentile(intervals, [25, 75])
    iqr = q3 - q1
    mask = (intervals >= q1 - 1.5 * iqr) & (intervals <= q3 + 1.5 * iqr)
    if np.sum(mask) == 0:
        return 60.0 / np.median(intervals)
    return 60.0 / np.median(intervals[mask])


def evaluate_file(wav_path, model):
    """단일 파일에서 GT/ML/DSP BPM 비교."""
    y, sr = librosa.load(wav_path, sr=TARGET_SR)
    if len(y) < WINDOW_SAMPLES * 2:
        return None

    # 1) Shannon GT
    filtered = labeler.bandpass_filter(y, sr)
    env = labeler.shannon_envelope(filtered)
    gt_peaks = labeler.detect_s1_peaks(env, sr)
    gt_bpm = beats_to_bpm(gt_peaks, sr)

    # 2) ML
    ml_beats, ml_probs = ml_detect_beats(y, model)
    ml_bpm = beats_to_bpm(ml_beats, sr)

    # 3) DSP
    dsp_peaks = dsp_detect_beats(y, sr)
    dsp_bpm = beats_to_bpm(dsp_peaks, sr)

    if gt_bpm < 30 or gt_bpm > 220:
        return None

    return {
        'file': os.path.basename(wav_path),
        'gt_bpm': gt_bpm,
        'ml_bpm': ml_bpm,
        'dsp_bpm': dsp_bpm,
        'gt_beats': len(gt_peaks),
        'ml_beats': len(ml_beats),
        'dsp_beats': len(dsp_peaks),
        'duration': len(y) / sr,
    }


def main():
    model = load_model()
    print(f"Model loaded on {DEVICE}")

    # 모든 WAV 수집
    all_wavs = []
    for d in TRAINING_DIRS:
        all_wavs.extend(sorted(glob.glob(os.path.join(DATA_ROOT, d, "*.wav"))))
    print(f"Total WAV files: {len(all_wavs)}")

    # 300개 랜덤 샘플로 평가
    random.seed(42)
    test_files = random.sample(all_wavs, min(300, len(all_wavs)))

    results = []
    for i, wav_path in enumerate(test_files):
        r = evaluate_file(wav_path, model)
        if r is not None:
            results.append(r)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(test_files)}...")

    print(f"\nValid results: {len(results)}/{len(test_files)}")

    # 통계 계산
    gt = np.array([r['gt_bpm'] for r in results])
    ml = np.array([r['ml_bpm'] for r in results])
    dsp = np.array([r['dsp_bpm'] for r in results])

    # BPM error (ML vs GT, DSP vs GT)
    ml_err = np.abs(ml - gt)
    dsp_err = np.abs(dsp - gt)

    # 유효 범위 (20-220 bpm)
    ml_valid = (ml > 20) & (ml < 220)
    dsp_valid = (dsp > 20) & (dsp < 220)

    print(f"\n{'='*60}")
    print(f"{'BPM Estimation Performance':^60}")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'ML':>12} {'DSP':>12}")
    print(f"{'-'*60}")
    print(f"{'Valid BPM ratio':<30} {np.mean(ml_valid):>11.1%} {np.mean(dsp_valid):>11.1%}")

    # 유효한 것만으로 에러 계산
    if np.sum(ml_valid) > 0:
        ml_mae = np.mean(ml_err[ml_valid])
        ml_med = np.median(ml_err[ml_valid])
        ml_within_5 = np.mean(ml_err[ml_valid] < 5)
        ml_within_10 = np.mean(ml_err[ml_valid] < 10)
    else:
        ml_mae = ml_med = ml_within_5 = ml_within_10 = 0

    if np.sum(dsp_valid) > 0:
        dsp_mae = np.mean(dsp_err[dsp_valid])
        dsp_med = np.median(dsp_err[dsp_valid])
        dsp_within_5 = np.mean(dsp_err[dsp_valid] < 5)
        dsp_within_10 = np.mean(dsp_err[dsp_valid] < 10)
    else:
        dsp_mae = dsp_med = dsp_within_5 = dsp_within_10 = 0

    print(f"{'MAE (bpm)':<30} {ml_mae:>12.1f} {dsp_mae:>12.1f}")
    print(f"{'Median AE (bpm)':<30} {ml_med:>12.1f} {dsp_med:>12.1f}")
    print(f"{'Within 5 bpm':<30} {ml_within_5:>11.1%} {dsp_within_5:>11.1%}")
    print(f"{'Within 10 bpm':<30} {ml_within_10:>11.1%} {dsp_within_10:>11.1%}")
    print(f"{'='*60}")

    # Correlation
    both_valid = ml_valid & dsp_valid
    if np.sum(both_valid) > 10:
        ml_corr = np.corrcoef(gt[both_valid], ml[both_valid])[0, 1]
        dsp_corr = np.corrcoef(gt[both_valid], dsp[both_valid])[0, 1]
        print(f"{'Correlation with GT':<30} {ml_corr:>12.3f} {dsp_corr:>12.3f}")

    # BPM 범위별 성능
    print(f"\n{'BPM Range Analysis':^60}")
    print(f"{'-'*60}")
    ranges = [(40, 70, 'Slow (40-70)'), (70, 100, 'Normal (70-100)'), (100, 150, 'Fast (100-150)'), (150, 220, 'Very Fast (150-220)')]
    for lo, hi, name in ranges:
        mask = (gt >= lo) & (gt < hi) & ml_valid
        n = np.sum(mask)
        if n > 5:
            mae = np.mean(ml_err[mask])
            w5 = np.mean(ml_err[mask] < 5)
            print(f"  {name:<25} n={n:>4}  MAE={mae:>5.1f}  <5bpm={w5:>5.1%}")

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) Scatter: GT vs ML
    ax = axes[0, 0]
    ax.scatter(gt[ml_valid], ml[ml_valid], alpha=0.3, s=15, c='#377eb8')
    ax.plot([30, 200], [30, 200], 'r--', linewidth=1)
    ax.set_xlabel('GT BPM (Shannon)')
    ax.set_ylabel('ML BPM')
    ax.set_title(f'ML vs GT (MAE={ml_mae:.1f}, r={ml_corr:.3f})')
    ax.set_xlim(30, 200)
    ax.set_ylim(30, 200)

    # 2) Scatter: GT vs DSP
    ax = axes[0, 1]
    ax.scatter(gt[dsp_valid], dsp[dsp_valid], alpha=0.3, s=15, c='#ff7f00')
    ax.plot([30, 200], [30, 200], 'r--', linewidth=1)
    ax.set_xlabel('GT BPM (Shannon)')
    ax.set_ylabel('DSP BPM')
    ax.set_title(f'DSP vs GT (MAE={dsp_mae:.1f}, r={dsp_corr:.3f})')
    ax.set_xlim(30, 200)
    ax.set_ylim(30, 200)

    # 3) Error distribution
    ax = axes[1, 0]
    ax.hist(ml_err[ml_valid], bins=50, alpha=0.7, color='#377eb8', label=f'ML (med={ml_med:.1f})')
    ax.hist(dsp_err[dsp_valid], bins=50, alpha=0.5, color='#ff7f00', label=f'DSP (med={dsp_med:.1f})')
    ax.set_xlabel('Absolute BPM Error')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.set_xlim(0, 50)

    # 4) Bland-Altman (ML)
    ax = axes[1, 1]
    mean_bpm = (gt[ml_valid] + ml[ml_valid]) / 2
    diff_bpm = ml[ml_valid] - gt[ml_valid]
    ax.scatter(mean_bpm, diff_bpm, alpha=0.3, s=15, c='#377eb8')
    ax.axhline(np.mean(diff_bpm), color='r', linestyle='-', label=f'Bias={np.mean(diff_bpm):.1f}')
    ax.axhline(np.mean(diff_bpm) + 1.96*np.std(diff_bpm), color='gray', linestyle='--')
    ax.axhline(np.mean(diff_bpm) - 1.96*np.std(diff_bpm), color='gray', linestyle='--')
    ax.set_xlabel('Mean BPM')
    ax.set_ylabel('ML - GT (bpm)')
    ax.set_title('Bland-Altman (ML)')
    ax.legend()

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "bpm_evaluation.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nFigure saved: {fig_path}")


if __name__ == "__main__":
    main()
