"""
LU-Net 복원율 계산 (Gemini 방식과 동일한 지표)
기존 평가 결과가 있으면 재사용, 없으면 새로 계산
"""
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATA_DIR = r"G:\stetho_ai\LUNet_Dataset"

print("모델 및 데이터 로드 중...")
model = tf.keras.models.load_model(os.path.join(DATA_DIR, "lunet_best.h5"))

X = np.load(os.path.join(DATA_DIR, "X_train.npy"))
Y = np.load(os.path.join(DATA_DIR, "Y_train.npy"))

_, X_test, _, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"평가 데이터: {X_test.shape[0]}개")
print("모델 예측 중...")
X_input = X_test[..., np.newaxis]
Y_pred = model.predict(X_input, batch_size=64, verbose=1)
Y_pred = Y_pred.squeeze(-1)

# =============================================
# 복원율 계산 (Gemini 방식과 동일)
# =============================================
# 원래 노이즈 양 = |noisy - clean|의 평균
original_noise = np.mean(np.abs(X_test - Y_test))

# 남은 노이즈 양 = |denoised - clean|의 평균
remaining_noise = np.mean(np.abs(Y_pred - Y_test))

# 복원율
restoration_rate = (1 - remaining_noise / original_noise) * 100

# SNR도 같이 계산
def snr_db(clean, signal):
    s_power = np.sum(clean ** 2)
    n_power = np.sum((clean - signal) ** 2)
    if n_power < 1e-10:
        return 50.0
    return 10 * np.log10(s_power / (n_power + 1e-10))

input_snrs = []
output_snrs = []
for i in tqdm(range(len(X_test)), desc="SNR 계산"):
    input_snrs.append(snr_db(Y_test[i], X_test[i]))
    output_snrs.append(snr_db(Y_test[i], Y_pred[i]))

avg_improvement = np.mean(np.array(output_snrs) - np.array(input_snrs))

print("\n" + "=" * 60)
print("        최종 비교 성적표")
print("=" * 60)
print(f"  평가 데이터 수:        {len(X_test)}개")
print(f"  초기 평균 노이즈 오차: {original_noise:.4f}")
print(f"  모델 통과 후 잔여 오차: {remaining_noise:.4f}")
print("-" * 60)
print(f"  복원율 (노이즈 제거율): {restoration_rate:.2f} %")
print(f"  평균 SNR 향상 (ΔSNR):  +{avg_improvement:.2f} dB")
print("=" * 60)
print()
print("  [비교표]")
print(f"  {'모델':<30} {'복원율':>8} {'ΔSNR':>10}")
print(f"  {'-'*50}")
print(f"  {'Gemini Bi-LSTM (스펙트로그램)':<30} {'53.65%':>8} {'+4.86 dB':>10}")
print(f"  {'Gemini 마스킹 CRNN':<30} {'54.10%':>8} {'+5.22 dB':>10}")
print(f"  {'LU-Net (waveform, ours)':<30} {f'{restoration_rate:.2f}%':>8} {f'+{avg_improvement:.2f} dB':>10}")
print("=" * 60)