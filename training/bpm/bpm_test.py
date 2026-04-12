"""
Step 4: BPM 추정 정확도 테스트 (PC)
=====================================
WAV 파일에 전체 파이프라인 적용:
  1. 대역통과 필터 (20-150Hz)
  2. Shannon Energy Envelope 계산
  3. Noise Gate CNN 판별 (심장음 여부)
  4. Autocorrelation BPM 추정

두 방법 비교:
  A. 기존 DSP (진폭 피크 감지) ← 현재 펌웨어 방식
  B. 새 방식 (Shannon Envelope + Autocorrelation + Noise Gate)

실행:
  C:/Users/dwd00/anaconda3/envs/stetho_ai/python.exe bpm_test.py [wav_path]
"""

import os
import sys
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, find_peaks
from pathlib import Path

# ─── 경로 ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "noise_gate.tflite")
SAMPLE_WAV = r"G:\stetho_ai\sample_sound\raw_heart_sound.wav"

# ─── 파라미터 (ESP32와 동일하게 유지할 것) ────────────────────────────────────
TARGET_SR     = 8000
FRAME_SIZE    = 80      # 10ms @ 8kHz
WINDOW_FRAMES = 100     # 1초 CNN 입력
HOP_FRAMES    = 25      # 250ms 홉 (더 자주 업데이트)
BP_LOW        = 20
BP_HIGH       = 150
ENV_SR        = TARGET_SR // FRAME_SIZE  # = 100 Hz (envelope sample rate)
NOISE_GATE_THRESHOLD = 0.5


# ─── 신호처리 ──────────────────────────────────────────────────────────────────

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = fs / 2.0
    return butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')


def compute_rms_envelope(audio, frame_size=FRAME_SIZE, smooth=15):
    """
    RMS Envelope 계산 (10ms 프레임).

    테스트 결과 기반 선택:
    - Shannon envelope는 프레임별 정규화로 진폭 패턴 손상 → 자기상관 실패
    - RMS는 절대 진폭 기반 → S1/S2 vs 침묵 구간 대비가 명확 → 자기상관 성공
    - smooth=15프레임(150ms)으로 박동 단위 smoothing
    """
    n_frames = len(audio) // frame_size
    envelope = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frame = audio[i * frame_size:(i + 1) * frame_size]
        envelope[i] = float(np.sqrt(np.mean(frame ** 2)))
    kernel = np.ones(smooth) / float(smooth)
    return np.convolve(envelope, kernel, mode='same').astype(np.float32)


def compute_shannon_envelope(audio, frame_size=FRAME_SIZE):
    """Shannon Energy Envelope (레거시, 현재는 compute_rms_envelope 권장)."""
    EPS = 1e-10
    n_frames = len(audio) // frame_size
    envelope = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frame = audio[i * frame_size:(i + 1) * frame_size]
        max_val = np.max(np.abs(frame)) + EPS
        x = frame / max_val
        x2 = x ** 2
        envelope[i] = np.mean(-x2 * np.log(x2 + EPS))
    kernel = np.ones(5) / 5
    return np.convolve(envelope, kernel, mode='same').astype(np.float32)


# ─── 방법 A: 기존 DSP 피크 감지 (현재 펌웨어) ─────────────────────────────────

def estimate_bpm_dsp_peak(audio, sr=TARGET_SR):
    """
    기존 방식: 진폭 절대값 피크 감지 → BPM.
    현재 펌웨어의 DSP_PEAK_REFRACTORY=2400 @ 8kHz 방식 모방.
    """
    sos = butter_bandpass(BP_LOW, BP_HIGH, sr)
    filtered = sosfilt(sos, audio)
    amp = np.abs(filtered)

    # 불응기: 2400 samples = 300ms (분당 200 BPM 한계)
    refractory = 2400
    threshold = np.percentile(amp, 85)  # 상위 15% 진폭

    peaks, _ = find_peaks(amp, height=threshold, distance=refractory)
    if len(peaks) < 2:
        return 0, peaks

    intervals = np.diff(peaks)
    median_interval = np.median(intervals)
    bpm = (sr / median_interval) * 60
    return float(np.clip(bpm, 20, 250)), peaks


# ─── 방법 B: RMS Envelope + Autocorrelation (검증된 최종 알고리즘) ────────────

def estimate_bpm_autocorr(envelope, fs_env=ENV_SR, min_bpm=40, max_bpm=200):
    """
    자기상관 기반 BPM 추정 (테스트 검증 완료 2026-03-27).

    핵심 알고리즘:
      1. 40-200 BPM lag 범위에서 자기상관 계산
      2. 40-130 BPM 범위 내 로컬 피크만 탐색
         (130+ BPM은 S1-S2 간격 artifact — 실제 BPM이 아님)
      3. 그 중 신뢰도(자기상관값) 최고 피크 선택

    검증 결과 (내 심음 데이터):
      - DSP 기존 방식:  143-177 BPM (S1+S2 둘 다 감지, 실제의 2배)
      - 이 알고리즘:     68-87 BPM  (실제 BPM, 정상 범위 확인)

    Returns:
        bpm: 추정 심박수 (0 = 신뢰할 수 없음)
        confidence: 자기상관 피크 높이 (0~1, ≥0.3 신뢰, 0.1-0.3 낮음)
        lag_samples: 감지된 주기 lag (envelope 프레임 수)
    """
    from scipy.signal import find_peaks as _sp_fp
    n = len(envelope)
    min_lag = max(1, int(60 * fs_env / max_bpm))
    max_lag = min(n // 2, int(60 * fs_env / min_bpm))

    if min_lag >= max_lag or n < WINDOW_FRAMES:
        return 0, 0.0, 0

    e = envelope - np.mean(envelope)
    norm = float(np.dot(e, e))
    if norm < 1e-10:
        return 0, 0.0, 0

    lags = np.arange(min_lag, max_lag + 1)
    r = np.array([float(np.dot(e[:n - lag], e[lag:])) / norm for lag in lags])

    # 40-130 BPM 범위 내 로컬 피크 탐색 (S1-S2 artifact 제외)
    peak_indices, _ = _sp_fp(r, height=0.08, distance=3)
    valid = []
    for pi in peak_indices:
        lag = int(lags[pi])
        bpm = 60.0 * fs_env / lag
        if 40 <= bpm <= 130:
            valid.append((float(r[pi]), bpm, lag))

    if not valid:
        # 로컬 피크 없으면 40-130 BPM 범위 global max fallback
        candidates = [(r[i], 60.0*fs_env/lags[i], int(lags[i]))
                      for i in range(len(lags))
                      if 40 <= (60.0*fs_env/lags[i]) <= 130]
        if not candidates:
            return 0, 0.0, 0
        best = max(candidates, key=lambda x: x[0])
        return float(np.clip(best[1], 40, 130)), float(best[0]), best[2]

    # 신뢰도 최고 피크 선택
    best_conf, best_bpm, best_lag = max(valid, key=lambda x: x[0])
    if best_conf < 0.08:
        return 0, best_conf, best_lag
    return float(best_bpm), best_conf, best_lag


# ─── Noise Gate 추론 ──────────────────────────────────────────────────────────

class NoiseGate:
    """TFLite 노이즈 게이트 (ESP32와 동일한 로직)."""

    def __init__(self, model_path):
        if not os.path.exists(model_path):
            print(f"[WARN] 모델 없음: {model_path} — 노이즈 게이트 비활성화")
            self.interpreter = None
            return

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.in_det  = self.interpreter.get_input_details()[0]
        self.out_det = self.interpreter.get_output_details()[0]
        self.in_scale, self.in_zp   = self.in_det["quantization"]
        self.out_scale, self.out_zp = self.out_det["quantization"]
        print(f"Noise Gate 모델 로드: {model_path}")

    def predict(self, window):
        """
        Args:
            window: (100,) float32 정규화된 Shannon envelope
        Returns:
            probability: float [0, 1] — 심장음 존재 확률
        """
        if self.interpreter is None:
            return 1.0  # 게이트 없으면 항상 통과

        x = window.reshape(1, -1, 1).astype(np.float32)

        if self.in_scale != 0:
            x_int8 = np.clip(np.round(x / self.in_scale + self.in_zp), -128, 127).astype(np.int8)
        else:
            x_int8 = x.astype(np.int8)

        self.interpreter.set_tensor(self.in_det["index"], x_int8)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.out_det["index"])
        return float((out[0][0].astype(np.float32) - self.out_zp) * self.out_scale)


# ─── 전체 파이프라인 테스트 ────────────────────────────────────────────────────

def test_wav(wav_path, noise_gate: NoiseGate = None, plot=True):
    """
    WAV 파일에 전체 파이프라인 적용.

    Args:
        wav_path:    테스트할 WAV 파일
        noise_gate:  NoiseGate 인스턴스 (None이면 비활성화)
        plot:        파형 + envelope 플롯 표시 여부
    """
    print(f"\n{'='*60}")
    print(f"파일: {wav_path}")

    # 오디오 로드
    audio, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    duration = len(audio) / TARGET_SR
    print(f"길이: {duration:.1f}초, 샘플레이트: {sr}Hz")

    # 대역통과 필터
    sos = butter_bandpass(BP_LOW, BP_HIGH, TARGET_SR)
    filtered = sosfilt(sos, audio).astype(np.float32)

    # RMS Envelope (검증된 방식)
    envelope = compute_rms_envelope(filtered)
    n_env = len(envelope)

    # ─── 방법 A: 기존 DSP 피크 감지 ───────────────────────────────────────────
    bpm_dsp, peaks_dsp = estimate_bpm_dsp_peak(audio)
    print(f"\n[방법 A] 기존 DSP 피크 감지:")
    print(f"  감지된 피크 수: {len(peaks_dsp)}")
    print(f"  BPM: {bpm_dsp:.1f}")
    if bpm_dsp > 200:
        print(f"  ⚠️  이상값 (잡음 피크일 가능성)")

    # ─── 방법 B: Noise Gate + Autocorrelation ─────────────────────────────────
    # 창 단위 Noise Gate 판별
    gate_scores = []
    for start in range(0, n_env - WINDOW_FRAMES + 1, HOP_FRAMES):
        w = envelope[start:start + WINDOW_FRAMES]
        max_val = np.max(w) + 1e-10
        w_norm = w / max_val

        if noise_gate is not None:
            score = noise_gate.predict(w_norm)
        else:
            score = 1.0

        gate_scores.append((start, score))

    valid_gates = [(s, sc) for s, sc in gate_scores if sc >= NOISE_GATE_THRESHOLD]
    gate_ratio = len(valid_gates) / max(len(gate_scores), 1)

    print(f"\n[방법 B] Shannon Envelope + Autocorrelation:")
    print(f"  Noise Gate 통과율: {gate_ratio:.1%} ({len(valid_gates)}/{len(gate_scores)})")

    # 유효 창이 충분하면 전체 envelope으로 autocorrelation BPM
    if gate_ratio >= 0.3:
        bpm_new, conf, lag = estimate_bpm_autocorr(envelope)
        if bpm_new > 0:
            period_ms = (lag / ENV_SR) * 1000
            print(f"  자기상관 주기 lag: {lag}프레임 = {period_ms:.0f}ms")
            print(f"  BPM: {bpm_new:.1f} (신뢰도: {conf:.3f})")
        else:
            print(f"  BPM: 측정 불가 (신뢰도 낮음: {conf:.3f})")
            bpm_new = 0
    else:
        print(f"  BPM: 측정 불가 (심장음 비율 낮음: {gate_ratio:.1%})")
        bpm_new, conf = 0, 0.0

    # ─── 비교 결과 ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print(f"결과 비교:")
    print(f"  기존 DSP:  {bpm_dsp:.0f} BPM {'⚠️ 이상' if bpm_dsp > 200 else '✓'}")
    print(f"  새 방식:   {bpm_new:.0f} BPM (신뢰도: {conf:.2f}) {'⚠️ 낮음' if conf < 0.2 else '✓'}")

    # ─── 플롯 ──────────────────────────────────────────────────────────────────
    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
        t_audio = np.arange(len(audio)) / TARGET_SR
        t_env   = np.arange(n_env) / ENV_SR

        # 1. 파형
        axes[0].plot(t_audio, audio, color='#888', linewidth=0.5, alpha=0.6, label='원본')
        axes[0].plot(t_audio, filtered, color='#4fc3f7', linewidth=0.8, label='필터 (20-150Hz)')
        if len(peaks_dsp):
            axes[0].plot(peaks_dsp / TARGET_SR, audio[peaks_dsp], 'rv', markersize=6, label='DSP 피크')
        axes[0].set_title(f'파형 (기존 DSP BPM: {bpm_dsp:.0f})')
        axes[0].set_xlabel('시간 (초)')
        axes[0].legend(loc='upper right', fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # 2. Shannon Envelope + Gate 점수
        ax2 = axes[1]
        ax2.fill_between(t_env, envelope, alpha=0.6, color='#81c784', label='Shannon Envelope')
        ax2.set_title('Shannon Energy Envelope')
        ax2.set_xlabel('시간 (초)')
        ax2.set_ylabel('에너지')

        # Noise Gate 창 표시
        if gate_scores:
            for start, score in gate_scores:
                t_start = start / ENV_SR
                t_end   = (start + WINDOW_FRAMES) / ENV_SR
                color = '#2196F3' if score >= NOISE_GATE_THRESHOLD else '#F44336'
                ax2.axvspan(t_start, t_end, alpha=0.15, color=color)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. 자기상관
        n = len(envelope)
        e = envelope - np.mean(envelope)
        norm = np.dot(e, e) + 1e-10
        min_lag = int(60 * ENV_SR / 200)
        max_lag = min(n // 2, int(60 * ENV_SR / 30))
        lags = np.arange(min_lag, max_lag + 1)
        if len(lags):
            r = np.array([np.dot(e[:n - lag], e[lag:]) / norm for lag in lags])
            bpm_axis = 60.0 * ENV_SR / lags

            axes[2].plot(bpm_axis, r, color='#ce93d8', linewidth=1.5)
            if bpm_new > 0:
                axes[2].axvline(bpm_new, color='#f06292', linewidth=2, linestyle='--',
                               label=f'BPM = {bpm_new:.0f}')
            axes[2].set_title(f'자기상관 (새 방식 BPM: {bpm_new:.0f}, 신뢰도: {conf:.2f})')
            axes[2].set_xlabel('BPM')
            axes[2].set_ylabel('자기상관')
            axes[2].invert_xaxis()  # 낮은 BPM이 오른쪽 (자연스러운 표시)
            axes[2].legend(fontsize=8)
            axes[2].grid(True, alpha=0.3)

        plt.suptitle(f'{os.path.basename(wav_path)}\n'
                     f'기존 DSP: {bpm_dsp:.0f} BPM  →  새 방식: {bpm_new:.0f} BPM',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "bpm_test_result.png"), dpi=120, bbox_inches='tight')
        print(f"\n플롯 저장: {os.path.join(BASE_DIR, 'bpm_test_result.png')}")
        plt.show()

    return bpm_dsp, bpm_new, conf


def main():
    wav_path = sys.argv[1] if len(sys.argv) > 1 else SAMPLE_WAV

    if not os.path.exists(wav_path):
        print(f"[ERROR] WAV 파일 없음: {wav_path}")
        print("사용법: python bpm_test.py [wav_path]")
        return

    # Noise Gate 로드
    noise_gate = NoiseGate(MODEL_PATH)

    # 테스트
    test_wav(wav_path, noise_gate, plot=True)

    # 추가: PhysioNet 데이터로 배치 테스트 (옵션)
    heart_dir = Path(r"G:\stetho_ai\_misc\datasets\classification-of-heart-sound-recordings")
    if heart_dir.exists() and "--batch" in sys.argv:
        print("\n\n배치 테스트 (PhysioNet 데이터 20개)...")
        wav_files = list(heart_dir.rglob("*.wav"))[:20]
        for wf in wav_files:
            try:
                test_wav(str(wf), noise_gate, plot=False)
            except Exception as e:
                print(f"[WARN] {wf.name}: {e}")


if __name__ == "__main__":
    main()
