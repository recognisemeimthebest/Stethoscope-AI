import os
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
import glob

# ==========================================
# 1. 훈련된 모델과 스케일링 파라미터 불러오기
# ==========================================
DATA_DIR = r"G:\stetho_ai\LSTM_first\Dataset_10000"

print("1. 인공지능 모델을 깨우는 중...")
model = tf.keras.models.load_model(os.path.join(DATA_DIR, 'pro_denoising_crnn.h5'))
Y_min, Y_max = np.load(os.path.join(DATA_DIR, 'scaling_params.npy'))


# ==========================================
# 2. 오디오 복원 함수 (노이즈 캔슬링 프로세스)
# ==========================================
def remove_noise_from_audio(audio_path, save_path):
    TARGET_SR = 2000

    # 1. 오디오 로딩 및 5초(10000샘플) 길이 맞추기
    y, _ = librosa.load(audio_path, sr=TARGET_SR)
    if len(y) > TARGET_SR * 5:
        y = y[:TARGET_SR * 5]  # 5초보다 길면 자름
    else:
        y = np.pad(y, (0, TARGET_SR * 5 - len(y)))  # 짧으면 무음으로 채움

    # 2. 스펙트로그램으로 변환 (인공지능의 눈)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=TARGET_SR, n_fft=512, hop_length=128, n_mels=64, fmax=1000
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).T

    # 3. 모델이 먹기 좋게 스케일링 (0 ~ 1)
    X_min, X_max = np.min(mel_spec_db), np.max(mel_spec_db)
    mel_scaled = (mel_spec_db - X_min) / (X_max - X_min + 1e-8)

    # 형태 맞추기 (Batch Size 1 추가 -> (1, 79, 64))
    mel_input = np.expand_dims(mel_scaled, axis=0)

    # 4. 🔥 인공지능 예측! (노이즈 닦아내기)
    pred_scaled = model.predict(mel_input, verbose=0)[0]

    # 5. 원래 데시벨(dB) 스케일로 복원
    pred_db = pred_scaled * (Y_max - Y_min + 1e-8) + Y_min

    # 6. 스펙트로그램을 다시 오디오 파형(소리)으로 역변환 (Griffin-Lim 알고리즘)
    pred_power = librosa.db_to_power(pred_db.T)
    y_clean = librosa.feature.inverse.mel_to_audio(
        pred_power, sr=TARGET_SR, n_fft=512, hop_length=128
    )

    # 🔥 [여기에 추가됨!] 6.5. 쪼그라든 볼륨 다시 빵빵하게 키워주기 (Normalization)
    max_val = np.max(np.abs(y_clean))
    if max_val > 0:
        y_clean = (y_clean / max_val) * 0.9  # 사람이 듣기 좋게 최대 볼륨의 90%로 증폭!

    # 7. 깨끗해진 파일 저장
    sf.write(save_path, y_clean, TARGET_SR, subtype='PCM_16')
    print(f" -> 완료! 복원된 오디오가 저장되었습니다: {os.path.basename(save_path)}")


# ==========================================
# 3. 실제 테스트 진행!
# ==========================================
if __name__ == "__main__":
    # 테스트 파일이 있는 폴더 경로
    TEST_DIR = r"G:\stetho_ai\Preprocessed_Mix_Test"
    SAVE_FILE = r"G:\stetho_ai\LSTM_first\denoised_result.wav"

    # 폴더 안의 .wav 파일을 자동으로 모두 찾습니다.
    wav_files = glob.glob(os.path.join(TEST_DIR, "*.wav"))

    if not wav_files:
        print(f"앗! '{TEST_DIR}' 폴더에 테스트할 wav 파일이 없습니다. 경로를 확인해 주세요!")
    else:
        # 찾은 파일 중 첫 번째 파일을 자동으로 선택합니다.
        TEST_NOISY_FILE = wav_files[0]

        print(f"\n2. 노이즈 캔슬링 시작! (테스트 파일: {os.path.basename(TEST_NOISY_FILE)})")
        remove_noise_from_audio(TEST_NOISY_FILE, SAVE_FILE)

        print(f"\n🎉 작업 완료! '{SAVE_FILE}' 파일을 열어서 들어보세요!")