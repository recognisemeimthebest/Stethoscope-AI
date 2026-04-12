import numpy as np
import librosa
import soundfile as sf


def calculate_rms(signal):
    """오디오 신호의 RMS(Root Mean Square) 에너지를 계산하여 소리의 크기를 구합니다."""
    return np.sqrt(np.mean(signal ** 2))


def mix_audio_with_snr(clean_signal, noise_signal, snr_db):
    """
    원하는 SNR(dB)에 맞춰 깨끗한 심음과 노이즈를 합성합니다.

    clean_signal: 깨끗한 심음 (1D numpy array)
    noise_signal: 노이즈 (1D numpy array)
    snr_db: 목표 신호 대 잡음비 (예: 5, 0, -5)
    """
    # 1. 두 신호의 현재 소리 크기(RMS) 계산
    clean_rms = calculate_rms(clean_signal)
    noise_rms = calculate_rms(noise_signal)

    # 노이즈가 완전히 없는 무음 구간일 경우 예외 처리
    if clean_rms == 0 or noise_rms == 0:
        return clean_signal

    # 2. 목표 SNR을 달성하기 위해 노이즈가 가져야 할 목표 RMS 계산
    # 공식 변환: target_noise_rms = clean_rms / (10 ^ (snr_db / 20))
    target_noise_rms = clean_rms / (10 ** (snr_db / 20))

    # 3. 목표치에 맞게 노이즈 볼륨 조절(Scaling)
    scaled_noise = noise_signal * (target_noise_rms / noise_rms)

    # 4. 심음과 조절된 노이즈 합성
    mixed_signal = clean_signal + scaled_noise

    # 5. 오디오 클리핑(Clipping) 방지: 소리가 너무 커져서 깨지는 것을 막음
    max_val = np.max(np.abs(mixed_signal))
    if max_val > 1.0:
        mixed_signal = mixed_signal / max_val

    return mixed_signal


# ==========================================
# 🧪 테스트 실행 예시 (가상의 시나리오)
# ==========================================
if __name__ == "__main__":
    # 타겟 샘플링 레이트
    TARGET_SR = 2000

    # 1. 데이터 불러오기 및 2000Hz로 다운샘플링 (예시 경로)
    # y_clean, _ = librosa.load("clean_heart_sound.wav", sr=TARGET_SR)
    # y_noise, _ = librosa.load("tap_rub_noise.wav", sr=TARGET_SR)

    # 임시로 더미 데이터(랜덤 노이즈)를 만들어 테스트해 봅니다 (5초 길이)
    y_clean = np.random.randn(TARGET_SR * 5) * 0.5  # 가상의 심음
    y_noise = np.random.randn(TARGET_SR * 5) * 0.1  # 가상의 노이즈

    # 2. 노이즈 합성 (예: SNR 0dB - 심음과 노이즈 크기가 같게 빡세게 설정!)
    mixed_audio = mix_audio_with_snr(y_clean, y_noise, snr_db=0)

    print("노이즈 합성이 완료되었습니다!")
    # sf.write('mixed_output_0dB.wav', mixed_audio, TARGET_SR) # 결과물 저장