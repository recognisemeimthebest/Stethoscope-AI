import os
import numpy as np
import librosa
import soundfile as sf
import random


# ==========================================
# 1. 핵심 엔진 + 볼륨 증폭(Normalization)
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
    """소리를 사람이 듣기 좋게 최대 볼륨의 90% 수준으로 키워줍니다."""
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal / max_val * 0.9
    return signal


# ==========================================
# 2. 5초 깍둑썰기 (묵음 필터링 추가 ✂️)
# ==========================================
def slice_into_chunks(audio_array, sr, chunk_sec=5, silence_threshold=0.001):
    """5초로 자르되, 소리가 너무 작은(묵음) 조각은 버립니다."""
    chunk_length = sr * chunk_sec
    chunks = []
    if len(audio_array) < chunk_length:
        return chunks

    for i in range(0, len(audio_array) - chunk_length + 1, chunk_length):
        chunk = audio_array[i: i + chunk_length]
        # 해당 5초 조각의 볼륨(RMS)이 기준치 이상일 때만 합격!
        if calculate_rms(chunk) > silence_threshold:
            chunks.append(chunk)
    return chunks


# ==========================================
# 3. 실전 데이터 생성기
# ==========================================
if __name__ == "__main__":
    TARGET_SR = 2000
    CHUNK_SEC = 5

    CLEAN_FILE = r"G:\stetho_ai\LSTM_first\Heart_beat_20260323.wav"
    NOISE_FILE = r"G:\stetho_ai\LSTM_first\Tap_rub.wav"
    SAVE_DIR = r"G:\stetho_ai\Preprocessed_Mix_Test"

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("1. 파일 로딩 및 묵음 제거 중...")

    y_clean, _ = librosa.load(CLEAN_FILE, sr=TARGET_SR)
    clean_chunks = slice_into_chunks(y_clean, TARGET_SR, CHUNK_SEC)

    y_noise, _ = librosa.load(NOISE_FILE, sr=TARGET_SR)
    noise_chunks = slice_into_chunks(y_noise, TARGET_SR, CHUNK_SEC)

    print(f" -> 알짜배기 심음 조각: {len(clean_chunks)}개 추출됨")
    print(f" -> 알짜배기 노이즈 조각: {len(noise_chunks)}개 추출됨")

    if len(clean_chunks) == 0 or len(noise_chunks) == 0:
        print("경고: 추출된 조각이 없습니다! silence_threshold 값을 더 낮춰보세요.")
    else:
        print("\n2. 무작위 노이즈 합성을 시작합니다...")
        num_test_files = min(10, len(clean_chunks))

        for i in range(num_test_files):
            clean_sample = random.choice(clean_chunks)
            noise_sample = random.choice(noise_chunks)

            random_snr = random.choice([-5, 0, 5, 10, 15])

            # 1단계: 섞기
            mixed_audio = mix_audio_with_snr(clean_sample, noise_sample, snr_db=random_snr)

            # 2단계: 빵빵하게 볼륨 키우기 (중요!)
            mixed_audio = normalize_audio(mixed_audio)

            # subtype을 'PCM_16'으로 명시하여 윈도우 기본 플레이어 호환성 높임
            save_name = f"mixed_sample_{i + 1}_SNR_{random_snr}dB.wav"
            sf.write(os.path.join(SAVE_DIR, save_name), mixed_audio, TARGET_SR, subtype='PCM_16')
            print(f"저장 완료: {save_name}")

        print(f"\n성공! 결과물을 다시 확인해보세요.")