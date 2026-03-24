import os
import numpy as np
import librosa
import random
from tqdm import tqdm


# ==========================================
# 1. 오디오 처리 및 스펙트로그램 엔진
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


def audio_to_melspectrogram(signal, sr):
    mel_spec = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=512, hop_length=128, n_mels=64, fmax=1000
    )
    return librosa.power_to_db(mel_spec, ref=np.max).T


# ==========================================
# 2. 대량의 폴더 탐색 및 로딩 함수
# ==========================================
def load_chunks_from_dir(directory, sr, chunk_sec, max_files=500):
    """폴더와 하위 폴더를 뒤져서 wav 파일을 찾고 조각으로 만듭니다."""
    chunks = []
    wav_files = []

    # 하위 폴더(training-a,b,c...)까지 모두 탐색
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, f))

    if not wav_files:
        return chunks

    # 파일이 너무 많으면 메모리가 터지거나 하루 종일 걸릴 수 있으므로 섞어서 일부만 추출
    random.shuffle(wav_files)
    if max_files is not None:
        wav_files = wav_files[:max_files]

    folder_name = os.path.basename(os.path.normpath(directory))
    for f in tqdm(wav_files, desc=f"[{folder_name}] 로딩 중"):
        try:
            y, _ = librosa.load(f, sr=sr)
            chunks.extend(slice_into_chunks(y, sr, chunk_sec))
        except Exception:
            pass  # 깨진 파일은 무시
    return chunks


# ==========================================
# 3. 메인 실행: 데이터 영혼까지 끌어모으기
# ==========================================
if __name__ == "__main__":
    TARGET_SR = 2000
    CHUNK_SEC = 5
    NUM_SAMPLES = 10000  # 🔥 최종 생성할 데이터 개수 (1만 개!)

    # 각 폴더에서 불러올 최대 파일 수 (원하면 늘려도 되지만 로딩 시간이 길어집니다)
    MAX_FILES = 500

    # 전체 데이터 경로 세팅
    DIR_ADULT = r"G:\stetho_ai\HeartSound For Adult\classification-of-heart-sound-recordings\classification-of-heart-sound-recordings"
    DIR_PEDIATRIC = r"G:\stetho_ai\HeartSound For Pediatric\cleaned_data"
    DIR_ESC50 = r"G:\stetho_ai\ESC-50-master\ESC-50-master\audio"

    FILE_MY_HEART = r"G:\stetho_ai\LSTM_first\Heart_beat_20260323.wav"
    FILE_MY_TAP = r"G:\stetho_ai\LSTM_first\Tap_rub.wav"

    # 혹시 Lilygo_background 파일이 있다면 여기 경로를 적어주세요 (없으면 None)
    FILE_MY_BG = r"G:\stetho_ai\LSTM_first\Lilygo_background.wav"

    SAVE_DIR = r"G:\stetho_ai\LSTM_first\Dataset_10000"
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("🚀 [STEP 1] 정답지(Clean) 풀(Pool) 만들기 시작!")
    clean_pool = []
    # 1. 성인 심음 로딩
    clean_pool.extend(load_chunks_from_dir(DIR_ADULT, TARGET_SR, CHUNK_SEC, MAX_FILES))
    # 2. 아동 심음 로딩
    clean_pool.extend(load_chunks_from_dir(DIR_PEDIATRIC, TARGET_SR, CHUNK_SEC, MAX_FILES))
    # 3. 직접 녹음한 심음 로딩
    if os.path.exists(FILE_MY_HEART):
        y, _ = librosa.load(FILE_MY_HEART, sr=TARGET_SR)
        my_clean_chunks = slice_into_chunks(y, TARGET_SR, CHUNK_SEC)
        # 직접 녹음한 데이터는 모델에게 아주 중요하므로 가중치를 주기 위해 10번 반복해서 듬뿍 넣어줍니다.
        clean_pool.extend(my_clean_chunks * 10)

    print(f"✅ 정답지 조각 총 {len(clean_pool)}개 확보 완료!\n")

    print("🚀 [STEP 2] 노이즈(Noise) 풀(Pool) 만들기 시작!")
    noise_pool = []
    # 1. ESC-50 범용 환경 노이즈 로딩
    noise_pool.extend(load_chunks_from_dir(DIR_ESC50, TARGET_SR, CHUNK_SEC, MAX_FILES))
    # 2. 마찰음 로딩
    if os.path.exists(FILE_MY_TAP):
        y, _ = librosa.load(FILE_MY_TAP, sr=TARGET_SR)
        my_noise_chunks = slice_into_chunks(y, TARGET_SR, CHUNK_SEC)
        noise_pool.extend(my_noise_chunks * 10)  # 중요 데이터 듬뿍
    # 3. 백그라운드 노이즈 로딩
    if os.path.exists(FILE_MY_BG):
        y, _ = librosa.load(FILE_MY_BG, sr=TARGET_SR)
        bg_chunks = slice_into_chunks(y, TARGET_SR, CHUNK_SEC)
        noise_pool.extend(bg_chunks * 10)

    print(f"✅ 노이즈 조각 총 {len(noise_pool)}개 확보 완료!\n")

    print(f"🚀 [STEP 3] 대규모 합성 및 스펙트로그램 변환 (총 {NUM_SAMPLES}개)...")
    X_data = []
    Y_data = []

    for i in tqdm(range(NUM_SAMPLES), desc="데이터셋 굽는 중"):
        clean_sample = random.choice(clean_pool)
        noise_sample = random.choice(noise_pool)

        # -5dB(노이즈 아주 큼) ~ 15dB(노이즈 아주 작음) 사이에서 극단적으로 훈련
        random_snr = random.choice([-5, -2, 0, 5, 10, 15])

        mixed_audio = mix_audio_with_snr(clean_sample, noise_sample, snr_db=random_snr)

        mixed_audio_norm = normalize_audio(mixed_audio)
        clean_sample_norm = normalize_audio(clean_sample)

        noisy_spec = audio_to_melspectrogram(mixed_audio_norm, TARGET_SR)
        clean_spec = audio_to_melspectrogram(clean_sample_norm, TARGET_SR)

        X_data.append(noisy_spec)
        Y_data.append(clean_spec)

    X_data = np.array(X_data, dtype=np.float32)  # 용량 최적화를 위해 float32 사용
    Y_data = np.array(Y_data, dtype=np.float32)

    print("\n🚀 [STEP 4] Numpy 파일로 저장 중... (시간이 조금 걸립니다)")
    np.save(os.path.join(SAVE_DIR, 'X_train.npy'), X_data)
    np.save(os.path.join(SAVE_DIR, 'Y_train.npy'), Y_data)

    print(f"\n🎉 완벽하게 끝났습니다! X shape: {X_data.shape}, Y shape: {Y_data.shape}")