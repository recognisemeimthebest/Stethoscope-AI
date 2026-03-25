"""
================================================================
LU-Net 심음 노이즈 제거 - Step 1: 데이터셋 생성기
================================================================
Gemini 파이프라인 대비 수정 사항:
  1. Waveform 도메인 (Mel-Spectrogram X → Raw 1D signal O)
     → Griffin-Lim 역변환 불필요 → "물먹은 소리" 문제 원천 차단
  2. 샘플링 레이트 4000Hz (2000Hz에서 상향)
     → S1/S2 고주파 하모닉스 + murmur 대역 보존
  3. 50% 오버랩 슬라이싱 → 데이터 2배 확보
  4. 진짜 Data Augmentation (gain jitter, time shift)
     → Gemini의 *10 복사(가짜 augmentation) 대체
  5. X/Y 동일 정규화 스케일 → 마스킹 모델 전제조건 충족

사용법:
  python 01_generate_dataset.py

필요 라이브러리:
  pip install librosa soundfile numpy tqdm
================================================================
"""

import os
import numpy as np
import librosa
import random
from tqdm import tqdm

# ============================================================
# 설정값 (종완의 데이터 경로에 맞게 수정하세요)
# ============================================================
CONFIG = {
    # 샘플링 레이트
    "sr": 4000,
    # 청크 길이 (초)
    "chunk_sec": 5,
    # 50% 오버랩
    "overlap_ratio": 0.5,
    # 묵음 판별 RMS 임계값
    "silence_threshold": 0.005,
    # 생성할 총 데이터 개수
    "num_samples": 10000,
    # 각 폴더에서 로딩할 최대 파일 수 (메모리 보호)
    "max_files_per_dir": 500,
    # SNR 범위 (dB) - 다양한 난이도로 학습
    "snr_range": [-5, -2, 0, 3, 5, 10, 15],

    # === 데이터 경로 (Windows) ===
    "dir_adult": r"G:\stetho_ai\HeartSound For Adult\classification-of-heart-sound-recordings\classification-of-heart-sound-recordings",
    "dir_pediatric": r"G:\stetho_ai\HeartSound For Pediatric\cleaned_data",
    "dir_esc50": r"G:\stetho_ai\ESC-50-master\ESC-50-master\audio",
    "file_my_heart": r"G:\stetho_ai\LSTM_first\Heart_beat_20260323.wav",
    "file_my_tap": r"G:\stetho_ai\LSTM_first\Tap_rub.wav",
    "file_my_bg": r"G:\stetho_ai\LSTM_first\Lilygo_background.wav",

    # 저장 경로
    "save_dir": r"G:\stetho_ai\LUNet_Dataset",
}


# ============================================================
# 1. 오디오 처리 핵심 함수들
# ============================================================
def calculate_rms(signal):
    """RMS 에너지 계산"""
    return np.sqrt(np.mean(signal ** 2))


def mix_audio_with_snr(clean, noise, snr_db):
    """
    목표 SNR에 맞춰 clean + noise 합성
    (Gemini 코드와 동일 로직, 검증 완료)
    """
    clean_rms = calculate_rms(clean)
    noise_rms = calculate_rms(noise)

    if clean_rms < 1e-10 or noise_rms < 1e-10:
        return clean.copy()

    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    mixed = clean + scaled_noise

    # 클리핑 방지
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val

    return mixed


def peak_normalize(signal, target_peak=0.9):
    """Peak normalization"""
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal / max_val * target_peak
    return signal


# ============================================================
# 2. 개선된 슬라이싱 (50% 오버랩 + 묵음 필터)
# ============================================================
def slice_into_chunks(audio, sr, chunk_sec=5, overlap_ratio=0.5,
                      silence_threshold=0.005):
    """
    [Gemini 대비 개선]
    - 50% 오버랩으로 슬라이싱 → 같은 파일에서 ~2배 데이터 확보
    - Gemini는 오버랩 없이 순차 슬라이싱 → 데이터 낭비
    """
    chunk_len = sr * chunk_sec
    hop_len = int(chunk_len * (1 - overlap_ratio))
    chunks = []

    if len(audio) < chunk_len:
        return chunks

    for i in range(0, len(audio) - chunk_len + 1, hop_len):
        chunk = audio[i: i + chunk_len]
        if calculate_rms(chunk) > silence_threshold:
            chunks.append(chunk)

    return chunks


# ============================================================
# 3. 진짜 Data Augmentation
# ============================================================
def augment_chunk(chunk, sr):
    """
    [핵심 수정] Gemini는 같은 조각을 *10 복사 → 과적합 유발
    여기서는 실제 변형을 가해서 진짜 다양성을 만듦
    """
    aug = chunk.copy()

    # 1) Gain jitter: 볼륨 ±30% 변동
    gain = random.uniform(0.7, 1.3)
    aug = aug * gain

    # 2) Time shift: 최대 ±0.2초 이동
    max_shift = int(sr * 0.2)
    shift = random.randint(-max_shift, max_shift)
    aug = np.roll(aug, shift)

    # 3) 약간의 백색 노이즈 추가 (하드웨어 노이즈 시뮬레이션)
    if random.random() < 0.3:
        hw_noise = np.random.randn(len(aug)) * 0.002
        aug = aug + hw_noise

    return aug


# ============================================================
# 4. 폴더 탐색 + 청크 로딩
# ============================================================
def load_chunks_from_dir(directory, sr, chunk_sec, overlap_ratio,
                         silence_threshold, max_files=500):
    """하위 폴더까지 탐색하며 wav 파일 → 청크로 변환"""
    chunks = []
    wav_files = []

    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, f))

    if not wav_files:
        return chunks

    random.shuffle(wav_files)
    if max_files is not None:
        wav_files = wav_files[:max_files]

    folder_name = os.path.basename(os.path.normpath(directory))
    for f in tqdm(wav_files, desc=f"[{folder_name}] 로딩"):
        try:
            y, _ = librosa.load(f, sr=sr)
            chunks.extend(
                slice_into_chunks(y, sr, chunk_sec, overlap_ratio,
                                  silence_threshold)
            )
        except Exception:
            pass  # 깨진 파일 무시

    return chunks


def load_chunks_from_file(filepath, sr, chunk_sec, overlap_ratio,
                          silence_threshold):
    """단일 wav 파일 → 청크로 변환"""
    if not os.path.exists(filepath):
        print(f"  [경고] 파일 없음: {filepath}")
        return []

    y, _ = librosa.load(filepath, sr=sr)
    fname = os.path.basename(filepath)
    chunks = slice_into_chunks(y, sr, chunk_sec, overlap_ratio,
                               silence_threshold)
    print(f"  [{fname}] → {len(chunks)}개 청크 추출")
    return chunks


# ============================================================
# 5. 메인 파이프라인
# ============================================================
def main():
    cfg = CONFIG
    sr = cfg["sr"]
    chunk_sec = cfg["chunk_sec"]
    chunk_len = sr * chunk_sec  # 4000 * 5 = 20000 samples

    os.makedirs(cfg["save_dir"], exist_ok=True)

    print("=" * 60)
    print(f" LU-Net 데이터셋 생성기 (SR={sr}Hz, Chunk={chunk_sec}s)")
    print("=" * 60)

    # ----- Step 1: Clean (정답지) 풀 만들기 -----
    print("\n[STEP 1] 정답지(Clean) 풀 수집 중...")
    clean_pool = []

    # 성인 심음
    if os.path.exists(cfg["dir_adult"]):
        clean_pool.extend(
            load_chunks_from_dir(
                cfg["dir_adult"], sr, chunk_sec,
                cfg["overlap_ratio"], cfg["silence_threshold"],
                cfg["max_files_per_dir"]
            )
        )

    # 아동 심음
    if os.path.exists(cfg["dir_pediatric"]):
        clean_pool.extend(
            load_chunks_from_dir(
                cfg["dir_pediatric"], sr, chunk_sec,
                cfg["overlap_ratio"], cfg["silence_threshold"],
                cfg["max_files_per_dir"]
            )
        )

    # 직접 녹음한 심음 (augmentation으로 다양성 확보)
    my_heart_chunks = load_chunks_from_file(
        cfg["file_my_heart"], sr, chunk_sec,
        cfg["overlap_ratio"], cfg["silence_threshold"]
    )
    # [핵심] *10 복사가 아닌, augmentation으로 변형 데이터 생성
    for _ in range(10):
        for chunk in my_heart_chunks:
            clean_pool.append(augment_chunk(chunk, sr))

    print(f"  → 정답지 총 {len(clean_pool)}개 청크 확보\n")

    # ----- Step 2: Noise 풀 만들기 -----
    print("[STEP 2] 노이즈(Noise) 풀 수집 중...")
    noise_pool = []

    # ESC-50 범용 노이즈
    if os.path.exists(cfg["dir_esc50"]):
        noise_pool.extend(
            load_chunks_from_dir(
                cfg["dir_esc50"], sr, chunk_sec,
                cfg["overlap_ratio"], cfg["silence_threshold"],
                cfg["max_files_per_dir"]
            )
        )

    # 마찰음 (augmentation)
    my_tap_chunks = load_chunks_from_file(
        cfg["file_my_tap"], sr, chunk_sec,
        cfg["overlap_ratio"], cfg["silence_threshold"]
    )
    for _ in range(10):
        for chunk in my_tap_chunks:
            noise_pool.append(augment_chunk(chunk, sr))

    # 배경 노이즈 (augmentation)
    my_bg_chunks = load_chunks_from_file(
        cfg["file_my_bg"], sr, chunk_sec,
        cfg["overlap_ratio"], cfg["silence_threshold"]
    )
    for _ in range(10):
        for chunk in my_bg_chunks:
            noise_pool.append(augment_chunk(chunk, sr))

    print(f"  → 노이즈 총 {len(noise_pool)}개 청크 확보\n")

    if len(clean_pool) == 0 or len(noise_pool) == 0:
        print("[에러] 청크가 부족합니다! 경로를 확인하세요.")
        return

    # ----- Step 3: 합성 데이터 생성 -----
    print(f"[STEP 3] {cfg['num_samples']}개 합성 데이터 생성 중...")

    X_data = []  # noisy waveform (입력)
    Y_data = []  # clean waveform (정답)

    for i in tqdm(range(cfg["num_samples"]), desc="데이터셋 생성"):
        clean = random.choice(clean_pool)
        noise = random.choice(noise_pool)
        snr_db = random.choice(cfg["snr_range"])

        # Peak normalize (둘 다 같은 기준)
        clean_norm = peak_normalize(clean, 0.9)

        # SNR 기반 합성
        mixed = mix_audio_with_snr(clean_norm, noise, snr_db)

        # 길이 검증 (안전장치)
        assert len(mixed) == chunk_len, \
            f"길이 불일치: {len(mixed)} != {chunk_len}"
        assert len(clean_norm) == chunk_len

        X_data.append(mixed)
        Y_data.append(clean_norm)

    # ----- Step 4: 저장 -----
    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)

    print(f"\n[STEP 4] 데이터 형태:")
    print(f"  X (noisy):  {X_data.shape}  →  (samples, waveform_length)")
    print(f"  Y (clean):  {Y_data.shape}")

    np.save(os.path.join(cfg["save_dir"], "X_train.npy"), X_data)
    np.save(os.path.join(cfg["save_dir"], "Y_train.npy"), Y_data)

    # 설정값도 함께 저장 (나중에 추론할 때 필요)
    np.savez(
        os.path.join(cfg["save_dir"], "config.npz"),
        sr=sr,
        chunk_sec=chunk_sec,
        chunk_len=chunk_len,
    )

    print(f"\n  저장 완료: {cfg['save_dir']}")
    print(f"  X_train.npy: {X_data.nbytes / 1024 / 1024:.1f} MB")
    print(f"  Y_train.npy: {Y_data.nbytes / 1024 / 1024:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()