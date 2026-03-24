import os
import numpy as np
import librosa
import random
from tqdm import tqdm  # 진행률 바 표시 (설치 필요: pip install tqdm)


# ==========================================
# 1. 오디오 처리 핵심 엔진
# ==========================================
def calculate_rms(signal):
    return np.sqrt(np.mean(signal ** 2))


def mix_audio_with_snr(clean_signal, noise_signal, snr_db):
    clean_rms = calculate_rms(clean_signal)
    noise_rms = calculate_rms(noise_signal)
    if clean_rms == 0 or noise_rms == 0:
        return clean_signal
    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    scaled_noise = noise_signal * (target_noise_rms / noise_rms)
    mixed_signal = clean_signal + scaled_noise
    return mixed_signal


def normalize_audio(signal):
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal / max_val * 0.9
    return signal


def slice_into_chunks(audio_array, sr, chunk_sec=5, silence_threshold=0.001):
    chunk_length = sr * chunk_sec
    chunks = []
    if len(audio_array) < chunk_length:
        return chunks
    for i in range(0, len(audio_array) - chunk_length + 1, chunk_length):
        chunk = audio_array[i: i + chunk_length]
        if calculate_rms(chunk) > silence_threshold:
            chunks.append(chunk)
    return chunks


# ==========================================
# 2. 스펙트로그램 변환 엔진
# ==========================================
def audio_to_melspectrogram(signal, sr):
    mel_spec = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=512, hop_length=128, n_mels=64, fmax=1000
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # ⚠️ 중요: LSTM은 (Time_steps, Features) 형태를 좋아합니다!
    # Librosa는 기본적으로 (Features, Time_steps)로 출력하므로 전치(Transpose) 해줍니다.
    return mel_spec_db.T


# ==========================================
# 3. 데이터셋 생성 파이프라인 실행
# ==========================================
if __name__ == "__main__":
    TARGET_SR = 2000
    CHUNK_SEC = 5

    # 데이터 경로
    CLEAN_FILE = r"G:\stetho_ai\LSTM_first\Heart_beat_20260323.wav"
    NOISE_FILE = r"G:\stetho_ai\LSTM_first\Tap_rub.wav"
    SAVE_DIR = r"G:\stetho_ai\LSTM_first\Dataset"  # numpy 파일이 저장될 최종 폴더

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("1. 데이터 로딩 및 깍둑썰기 중...")
    y_clean, _ = librosa.load(CLEAN_FILE, sr=TARGET_SR)
    clean_chunks = slice_into_chunks(y_clean, TARGET_SR, CHUNK_SEC)

    y_noise, _ = librosa.load(NOISE_FILE, sr=TARGET_SR)
    noise_chunks = slice_into_chunks(y_noise, TARGET_SR, CHUNK_SEC)

    # 생성할 총 데이터 개수 설정 (테스트용으로 100개만 만들어봅시다)
    NUM_SAMPLES = 100

    X_data = []  # 문제지 (Noisy)
    Y_data = []  # 정답지 (Clean)

    print(f"\n2. 총 {NUM_SAMPLES}개의 데이터셋 생성 시작...")

    for i in tqdm(range(NUM_SAMPLES)):
        clean_sample = random.choice(clean_chunks)
        noise_sample = random.choice(noise_chunks)
        random_snr = random.choice([-5, 0, 5, 10, 15])

        # 합성 및 볼륨 정규화
        mixed_audio = mix_audio_with_snr(clean_sample, noise_sample, snr_db=random_snr)
        mixed_audio_norm = normalize_audio(mixed_audio)
        clean_sample_norm = normalize_audio(clean_sample)

        # 스펙트로그램 변환 (형태: 시간 프레임 수 x 64개 멜 주파수 대역)
        noisy_spec = audio_to_melspectrogram(mixed_audio_norm, TARGET_SR)
        clean_spec = audio_to_melspectrogram(clean_sample_norm, TARGET_SR)

        X_data.append(noisy_spec)
        Y_data.append(clean_spec)

    # 리스트를 Numpy 배열로 꽉 압축하기
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    print(f"\n3. 데이터 형태 확인:")
    print(f" - X shape: {X_data.shape} (데이터 개수, 시간축 길이, 주파수 대역 수)")
    print(f" - Y shape: {Y_data.shape}")

    # Numpy 파일로 저장 (.npy)
    np.save(os.path.join(SAVE_DIR, 'X_train.npy'), X_data)
    np.save(os.path.join(SAVE_DIR, 'Y_train.npy'), Y_data)

    print(f"\n완성! '{SAVE_DIR}' 폴더에 X_train.npy 와 Y_train.npy 가 저장되었습니다.")