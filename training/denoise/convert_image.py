import os
import numpy as np
import librosa
import soundfile as sf
import random
import matplotlib.pyplot as plt
import librosa.display


# ==========================================
# 1 & 2. 기존 엔진 및 깍둑썰기 함수 (이전과 동일)
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
# 🌟 4. 스펙트로그램 변환 및 시각화 함수 (NEW!)
# ==========================================
def audio_to_melspectrogram(signal, sr):
    """1차원 오디오 파형을 2차원 멜 스펙트로그램(dB 스케일)으로 변환합니다."""
    # 심음 데이터에 맞춘 파라미터 설정 (수정 가능)
    # 2000Hz 기준 n_fft=512, hop_length=128 정도가 저주파 분석에 좋습니다.
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=512,  # 한 번에 분석할 윈도우 크기
        hop_length=128,  # 윈도우가 이동하는 보폭
        n_mels=64,  # y축(주파수)을 몇 개의 밴드로 나눌 것인가
        fmax=1000  # 나이퀴스트 주파수 (2000Hz의 절반)
    )
    # 에너지를 사람이 듣는 방식과 유사한 데시벨(dB) 스케일로 변환
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def save_spectrogram_image(clean_spec, noisy_spec, save_path):
    """정답지(Clean)와 문제지(Noisy)를 비교하는 이미지를 저장합니다."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    librosa.display.specshow(noisy_spec, x_axis='time', y_axis='mel', sr=2000, fmax=1000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Noisy Heart Sound (Model Input X)')

    plt.subplot(1, 2, 2)
    librosa.display.specshow(clean_spec, x_axis='time', y_axis='mel', sr=2000, fmax=1000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Clean Heart Sound (Target Y)')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ==========================================
# 3. 실전 데이터 생성기 (파이프라인 실행)
# ==========================================
if __name__ == "__main__":
    TARGET_SR = 2000
    CHUNK_SEC = 5

    CLEAN_FILE = r"G:\stetho_ai\LSTM_first\Heart_beat_20260323.wav"
    NOISE_FILE = r"G:\stetho_ai\LSTM_first\Tap_rub.wav"
    SAVE_DIR = r"G:\stetho_ai\Preprocessed_Mix_Test"

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("1. 데이터 로딩 및 깍둑썰기 중...")
    y_clean, _ = librosa.load(CLEAN_FILE, sr=TARGET_SR)
    clean_chunks = slice_into_chunks(y_clean, TARGET_SR, CHUNK_SEC)

    y_noise, _ = librosa.load(NOISE_FILE, sr=TARGET_SR)
    noise_chunks = slice_into_chunks(y_noise, TARGET_SR, CHUNK_SEC)

    print("\n2. 노이즈 합성 및 스펙트로그램 변환을 시작합니다...")
    num_test_files = min(3, len(clean_chunks))  # 시각화 테스트는 3개만 해봅시다.

    for i in range(num_test_files):
        clean_sample = random.choice(clean_chunks)
        noise_sample = random.choice(noise_chunks)
        random_snr = random.choice([-5, 0, 5, 10])

        # 1. 섞기 (문제지 만들기)
        mixed_audio = mix_audio_with_snr(clean_sample, noise_sample, snr_db=random_snr)

        # 2. 볼륨 정규화 (모델이 일관된 크기를 학습하게 함)
        mixed_audio_norm = normalize_audio(mixed_audio)
        clean_sample_norm = normalize_audio(clean_sample)  # 정답지도 똑같이 정규화

        # 3. 멜 스펙트로그램으로 변환 (1D -> 2D)
        noisy_spec = audio_to_melspectrogram(mixed_audio_norm, TARGET_SR)
        clean_spec = audio_to_melspectrogram(clean_sample_norm, TARGET_SR)

        # 4. 결과물을 이미지로 저장해서 눈으로 확인
        img_save_name = f"spectrogram_compare_SNR_{random_snr}dB_sample{i + 1}.png"
        save_spectrogram_image(clean_spec, noisy_spec, os.path.join(SAVE_DIR, img_save_name))

        print(f"이미지 저장 완료: {img_save_name} (노이즈 섞인 정도: {random_snr}dB)")

    print(f"\n완성입니다! '{SAVE_DIR}' 폴더에서 PNG 이미지 파일들을 열어보세요.")