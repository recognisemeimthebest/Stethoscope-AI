"""
Step 1: Data Preparation for On-Device BPM Noise Gate Model
============================================================
논문 기반:
  - Shannon Energy Envelope (Springer et al. 2016, PhysioNet 2016 Challenge)
  - Noise-robust heart sound segmentation (PMC11469632)

학습 데이터:
  - 양성(1): PhysioNet 2016 심음 WAV → 심장음 창
  - 음성(0): ESC-50 환경음 → 잡음 창

특징:
  - 프레임: 80 samples @ 8kHz = 10ms
  - Shannon Energy per frame → envelope
  - 창 크기: 100 frames = 1초
  - 홉: 50 frames = 500ms

출력: data/ 디렉터리에 X_train.npy, y_train.npy, X_val.npy, y_val.npy
"""

import os
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt, lfilter
from sklearn.model_selection import train_test_split
from pathlib import Path
import random

# ─── 경로 설정 ───────────────────────────────────────────────────────────────
HEART_DATA_DIR = r"G:\stetho_ai\_misc\datasets\classification-of-heart-sound-recordings"
ESC50_DIR      = r"G:\stetho_ai\_misc\datasets\ESC-50-master\ESC-50-master\audio"
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), "data")

# ─── 파라미터 ─────────────────────────────────────────────────────────────────
TARGET_SR    = 8000    # ESP32 오디오 샘플레이트
FRAME_SIZE   = 80      # 10ms @ 8kHz
WINDOW_FRAMES = 100    # 1초 (CNN 입력)
HOP_FRAMES   = 50      # 500ms 홉
BP_LOW       = 20      # 심음 대역통과 하한 Hz
BP_HIGH      = 150     # 심음 대역통과 상한 Hz
RANDOM_SEED  = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─── 신호 처리 함수 ───────────────────────────────────────────────────────────

def butter_bandpass_sos(lowcut, highcut, fs, order=4):
    """Butterworth 대역통과 필터 (SOS 형식)."""
    nyq = fs / 2.0
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
    return sos


def load_and_resample(filepath, target_sr=TARGET_SR):
    """WAV/OGG 파일 로드 후 모노 + 목표 샘플레이트로 변환."""
    try:
        audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
        return audio.astype(np.float32)
    except Exception as e:
        print(f"  [WARN] 로드 실패 {filepath}: {e}")
        return None


def compute_shannon_envelope(audio, frame_size=FRAME_SIZE, smooth_frames=5):
    """
    Shannon Energy Envelope 계산.

    논문: Springer et al. (2016) - Logistic Regression-HSMM-based Heart Sound Segmentation
    수식: E(n) = -x²(n) · log(x²(n))  (정규화 후 적용)

    Args:
        audio:        1D float32 오디오 신호 (정규화됨, -1~1)
        frame_size:   프레임 샘플 수 (default: 80 = 10ms @ 8kHz)
        smooth_frames: 이동평균 평활화 창 크기

    Returns:
        envelope: 프레임별 Shannon energy (1D array)
    """
    n_frames = len(audio) // frame_size
    if n_frames == 0:
        return np.array([])

    envelope = np.zeros(n_frames, dtype=np.float32)
    EPS = 1e-10

    for i in range(n_frames):
        frame = audio[i * frame_size:(i + 1) * frame_size]

        # 프레임 내 최대값으로 정규화
        max_val = np.max(np.abs(frame)) + EPS
        x = frame / max_val

        # Shannon energy: E = -x² log(x²)
        x2 = x ** 2
        se = -x2 * np.log(x2 + EPS)
        envelope[i] = np.mean(se)

    # 이동평균 평활화
    if smooth_frames > 1:
        kernel = np.ones(smooth_frames, dtype=np.float32) / smooth_frames
        envelope = np.convolve(envelope, kernel, mode='same')

    return envelope


def extract_windows(envelope, window_frames=WINDOW_FRAMES, hop_frames=HOP_FRAMES):
    """Envelope에서 슬라이딩 창 추출."""
    windows = []
    n = len(envelope)
    for start in range(0, n - window_frames + 1, hop_frames):
        w = envelope[start:start + window_frames]
        # 0~1 정규화 (창 내 최대값 기준)
        max_val = np.max(w) + 1e-10
        w = w / max_val
        windows.append(w)
    return windows


def augment_window(window):
    """
    데이터 증강 (학습 시 과적합 방지).
    - 진폭 스케일 랜덤 변화 ±20%
    - 가우시안 잡음 추가
    - 역전 (부정적 Shannon energy는 없으므로 amplitude만)
    """
    augmented = []

    # 원본
    augmented.append(window.copy())

    # 진폭 변화
    for scale in [0.8, 1.2]:
        w = window * scale
        w = np.clip(w, 0, None)
        w = w / (np.max(w) + 1e-10)
        augmented.append(w)

    # 가우시안 잡음 (SNR ~20dB)
    noise = np.random.normal(0, 0.05, window.shape).astype(np.float32)
    w = np.abs(window + noise)
    w = w / (np.max(w) + 1e-10)
    augmented.append(w)

    return augmented


# ─── 데이터 수집 ──────────────────────────────────────────────────────────────

def collect_heart_sound_windows():
    """PhysioNet 2016 심음 데이터에서 양성 창 추출."""
    sos = butter_bandpass_sos(BP_LOW, BP_HIGH, TARGET_SR)
    positive_windows = []

    heart_dir = Path(HEART_DATA_DIR)
    if not heart_dir.exists():
        print(f"[WARN] 심음 데이터 디렉터리 없음: {HEART_DATA_DIR}")
        print("       PhysioNet 2016 데이터를 해당 경로에 준비해주세요.")
        return positive_windows

    # training-a ~ training-f 서브디렉터리 순회
    wav_files = list(heart_dir.rglob("*.wav"))
    print(f"[INFO] 심음 WAV 파일 수: {len(wav_files)}")

    for wav_path in wav_files[:3000]:  # 최대 3000개
        audio = load_and_resample(str(wav_path))
        if audio is None or len(audio) < TARGET_SR:
            continue

        # 대역통과 필터
        filtered = sosfilt(sos, audio).astype(np.float32)

        # Shannon envelope
        envelope = compute_shannon_envelope(filtered)
        if len(envelope) < WINDOW_FRAMES:
            continue

        # 창 추출 + 증강
        windows = extract_windows(envelope)
        for w in windows:
            augmented = augment_window(w)
            positive_windows.extend(augmented)

    print(f"[INFO] 양성 창 수 (심음): {len(positive_windows)}")
    return positive_windows


def collect_noise_windows():
    """ESC-50 환경음에서 음성 창 추출."""
    negative_windows = []

    esc_dir = Path(ESC50_DIR)
    if not esc_dir.exists():
        print(f"[WARN] ESC-50 디렉터리 없음: {ESC50_DIR}")
        print("       랜덤 가우시안 잡음으로 대체합니다.")
        # 랜덤 잡음으로 대체
        for _ in range(2000):
            noise = np.abs(np.random.normal(0, 1, WINDOW_FRAMES)).astype(np.float32)
            noise = noise / (np.max(noise) + 1e-10)
            negative_windows.append(noise)
        return negative_windows

    audio_files = list(esc_dir.glob("*.ogg")) + list(esc_dir.glob("*.wav"))
    print(f"[INFO] ESC-50 파일 수: {len(audio_files)}")

    for audio_path in audio_files:
        audio = load_and_resample(str(audio_path))
        if audio is None or len(audio) < TARGET_SR:
            continue

        # 심음 주파수 대역 제거 (20-150Hz는 일부 환경음에도 있으므로 그대로 사용)
        envelope = compute_shannon_envelope(audio)
        if len(envelope) < WINDOW_FRAMES:
            continue

        windows = extract_windows(envelope)
        negative_windows.extend(windows)

    # 랜덤 가우시안 잡음도 추가 (다양성)
    for _ in range(500):
        noise = np.abs(np.random.normal(0, 1, WINDOW_FRAMES)).astype(np.float32)
        noise = noise / (np.max(noise) + 1e-10)
        negative_windows.append(noise)

    print(f"[INFO] 음성 창 수 (잡음): {len(negative_windows)}")
    return negative_windows


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("온디바이스 BPM 노이즈 게이트 모델 데이터 준비")
    print("=" * 60)

    print("\n[1/3] 심음 창 추출 (양성 레이블=1)...")
    pos_windows = collect_heart_sound_windows()

    print("\n[2/3] 잡음 창 추출 (음성 레이블=0)...")
    neg_windows = collect_noise_windows()

    if not pos_windows or not neg_windows:
        print("[ERROR] 데이터가 없습니다. 경로를 확인해주세요.")
        return

    # 클래스 균형 맞추기
    n_samples = min(len(pos_windows), len(neg_windows))
    print(f"\n[3/3] 데이터셋 생성 (클래스당 {n_samples}개)...")

    random.shuffle(pos_windows)
    random.shuffle(neg_windows)
    pos_windows = pos_windows[:n_samples]
    neg_windows = neg_windows[:n_samples]

    X = np.array(pos_windows + neg_windows, dtype=np.float32)
    y = np.array([1] * len(pos_windows) + [0] * len(neg_windows), dtype=np.float32)

    # 셔플
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # CNN 입력 형태: (samples, window_frames, 1)
    X = X.reshape(-1, WINDOW_FRAMES, 1)

    # 학습/검증 분할 (80:20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # 저장
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"),   y_val)

    print(f"\n완료!")
    print(f"  학습 세트:  {X_train.shape} (양성: {int(y_train.sum())}, 음성: {int((1-y_train).sum())})")
    print(f"  검증 세트:  {X_val.shape}")
    print(f"  저장 경로:  {OUTPUT_DIR}")
    print(f"\n다음 단계: python train.py")


if __name__ == "__main__":
    main()
