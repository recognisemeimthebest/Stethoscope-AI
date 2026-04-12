import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==========================================
# 1. 모델과 1만 개 데이터셋 불러오기
# ==========================================
DATA_DIR = r"G:\stetho_ai\LSTM_first\Dataset_10000"

print("1. 채점 준비 중... (데이터 및 모델 로드)")
model = tf.keras.models.load_model(os.path.join(DATA_DIR, 'masking_denoising_crnn.h5'))

X = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
Y = np.load(os.path.join(DATA_DIR, 'Y_train.npy'))
Y_min, Y_max = np.load(os.path.join(DATA_DIR, 'scaling_params.npy'))

# 전체 데이터를 스케일링 (훈련할 때와 동일하게)
X_scaled = (X - np.min(X)) / (np.max(X) - np.min(X) + 1e-8)
Y_scaled = (Y - Y_min) / (Y_max - Y_min + 1e-8)

# 훈련 때 쓰지 않은 '검증용 데이터(Validation Set)' 20%만 분리해서 진짜 실력을 테스트합니다.
_, X_test, _, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

print(f" -> 총 {len(X_test)}개의 테스트 데이터로 채점을 시작합니다!\n")

# ==========================================
# 2. 모델 예측 (노이즈 닦아내기)
# ==========================================
print("2. 인공지능이 2000개의 문제지를 풀고 있습니다...")
# 한 번에 다 넣으면 메모리가 터질 수 있으니 batch_size를 설정하여 예측
Pred_scaled = model.predict(X_test, batch_size=64, verbose=1)

# 예측값과 원본값을 원래 데시벨(dB) 스케일로 복원
Pred_db = Pred_scaled * (Y_max - Y_min + 1e-8) + Y_min
X_db = X_test * (np.max(X) - np.min(X) + 1e-8) + np.min(X)
Y_db = Y_test * (Y_max - Y_min + 1e-8) + Y_min

# ==========================================
# 3. 성능 지표 (복원율) 계산
# ==========================================
print("\n3. 최종 성적표 계산 중...")

# [지표 1] 노이즈 제거율 (MAE 기반)
# 원래 껴있던 노이즈 양 = |노이즈 낀 심음 - 깨끗한 심음|
original_noise_mae = np.mean(np.abs(X_db - Y_db))

# 모델이 못 지우고 남긴 노이즈 양 = |모델이 닦아낸 심음 - 깨끗한 심음|
remaining_noise_mae = np.mean(np.abs(Pred_db - Y_db))

# 복원율 계산
noise_reduction_rate = (1 - (remaining_noise_mae / original_noise_mae)) * 100


# [지표 2] 신호 대 잡음비 (SNR) 향상도 측정
def calculate_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0: return 0
    return 10 * np.log10(signal_power / noise_power)


input_snrs = []
output_snrs = []

for i in tqdm(range(len(X_test)), desc="SNR 계산"):
    clean_signal = Y_db[i]

    # 노이즈 = 전체 소리 - 깨끗한 소리
    original_noise = X_db[i] - clean_signal
    remaining_noise = Pred_db[i] - clean_signal

    input_snrs.append(calculate_snr(clean_signal, original_noise))
    output_snrs.append(calculate_snr(clean_signal, remaining_noise))

avg_snr_improvement = np.mean(output_snrs) - np.mean(input_snrs)

# ==========================================
# 4. 최종 성적표 출력
# ==========================================
print("\n" + "=" * 50)
print(" 🏆 인공지능 노이즈 캔슬링 최종 성적표 🏆")
print("=" * 50)
print(f"✔️ 평가 데이터 수: {len(X_test)}개 (훈련에 안 쓴 낯선 데이터)")
print(f"✔️ 초기 평균 노이즈 오차: {original_noise_mae:.2f} dB")
print(f"✔️ 모델 통과 후 잔여 오차: {remaining_noise_mae:.2f} dB")
print("-" * 50)
print(f"🔥 전체 복원율 (노이즈 제거율): {noise_reduction_rate:.2f} %")
print(f"🔥 평균 음질 향상도 (ΔSNR):  +{avg_snr_improvement:.2f} dB")
print("=" * 50)