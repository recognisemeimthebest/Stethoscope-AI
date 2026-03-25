"""
================================================================
LU-Net v2 평가 - 전체 모델 비교표
================================================================
v1 baseline + v2 (Attention/Residual/SI-SNR) 비교
================================================================
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ============================================================
# 설정
# ============================================================
DATA_DIR = r"G:\stetho_ai\LUNet_Dataset"
MODEL_PATH = os.path.join(DATA_DIR, "lunet_v2_best.h5")


# ============================================================
# 커스텀 손실 함수 재정의 (모델 로드에 필요)
# ============================================================
def si_snr_loss(y_true, y_pred):
    y_true = K.squeeze(y_true, axis=-1)
    y_pred = K.squeeze(y_pred, axis=-1)
    y_true = y_true - K.mean(y_true, axis=-1, keepdims=True)
    y_pred = y_pred - K.mean(y_pred, axis=-1, keepdims=True)
    dot = K.sum(y_pred * y_true, axis=-1, keepdims=True)
    s_target = dot * y_true / (K.sum(y_true ** 2, axis=-1, keepdims=True) + 1e-8)
    e_noise = y_pred - s_target
    si_snr = 10 * K.log(
        K.sum(s_target ** 2, axis=-1) / (K.sum(e_noise ** 2, axis=-1) + 1e-8)
    ) / K.log(10.0)
    return -K.mean(si_snr)


def combined_loss(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))
    si_snr = si_snr_loss(y_true, y_pred)
    return 0.5 * mse + 0.5 * (si_snr / 20.0)


# ============================================================
# 평가 지표
# ============================================================
def snr_db(clean, signal):
    s_power = np.sum(clean ** 2)
    n_power = np.sum((clean - signal) ** 2)
    if n_power < 1e-10:
        return 50.0
    return 10 * np.log10(s_power / (n_power + 1e-10))


def si_snr_db_numpy(clean, denoised):
    clean = clean - np.mean(clean)
    denoised = denoised - np.mean(denoised)
    dot = np.sum(denoised * clean)
    s_target = dot * clean / (np.sum(clean ** 2) + 1e-10)
    e_noise = denoised - s_target
    return 10 * np.log10(
        np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + 1e-10)
    )


# ============================================================
# 메인
# ============================================================
def main():
    print("=" * 60)
    print(" LU-Net v2 성능 평가")
    print(" (Attention + Residual + SI-SNR Loss)")
    print("=" * 60)

    # ----- 로드 -----
    print("\n[1] 모델 및 데이터 로드 중...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "combined_loss": combined_loss,
            "si_snr_loss": si_snr_loss,
        }
    )

    X = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    Y = np.load(os.path.join(DATA_DIR, "Y_train.npy"))

    _, X_test, _, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    print(f"    평가 데이터: {X_test.shape[0]}개")

    # ----- 예측 -----
    print("\n[2] 모델 예측 중...")
    X_input = X_test[..., np.newaxis]
    Y_pred = model.predict(X_input, batch_size=64, verbose=1)
    Y_pred = Y_pred.squeeze(-1)

    # ----- 복원율 -----
    original_noise = np.mean(np.abs(X_test - Y_test))
    remaining_noise = np.mean(np.abs(Y_pred - Y_test))
    restoration_rate = (1 - remaining_noise / original_noise) * 100

    # ----- SNR / SI-SNR -----
    print("\n[3] SNR 계산 중...")
    input_snrs = []
    output_snrs = []
    si_snrs = []

    for i in tqdm(range(len(X_test)), desc="평가"):
        clean = Y_test[i]
        noisy = X_test[i]
        denoised = Y_pred[i]

        input_snrs.append(snr_db(clean, noisy))
        output_snrs.append(snr_db(clean, denoised))
        si_snrs.append(si_snr_db_numpy(clean, denoised))

    improvements = np.array(output_snrs) - np.array(input_snrs)

    # ----- 결과 -----
    print("\n" + "=" * 60)
    print("     LU-Net v2 최종 성적표")
    print("=" * 60)
    print(f"  평가 데이터 수:         {len(X_test)}개")
    print(f"  초기 평균 노이즈 오차:  {original_noise:.4f}")
    print(f"  모델 통과 후 잔여 오차: {remaining_noise:.4f}")
    print("-" * 60)
    print(f"  복원율 (노이즈 제거율): {restoration_rate:.2f} %")
    print(f"  평균 SNR 향상 (ΔSNR):  +{np.mean(improvements):.2f} dB")
    print(f"  평균 SI-SNR:           {np.mean(si_snrs):.2f} dB")
    print(f"  중앙값 SNR 향상:       +{np.median(improvements):.2f} dB")
    print(f"  최대 SNR 향상:         +{np.max(improvements):.2f} dB")
    print("=" * 60)

    # ----- 전체 비교표 -----
    print()
    print("=" * 75)
    print("                         전체 모델 비교표")
    print("=" * 75)
    print(f"  {'모델':<40} {'복원율':>8} {'ΔSNR':>10} {'SI-SNR':>10}")
    print(f"  {'-'*68}")
    print(f"  {'Gemini Bi-LSTM (스펙트로그램)':<40} {'53.65%':>8} {'+4.86 dB':>10} {'N/A':>10}")
    print(f"  {'Gemini 마스킹 CRNN':<40} {'54.10%':>8} {'+5.22 dB':>10} {'N/A':>10}")
    print(f"  {'LU-Net v1 5초 (baseline)':<40} {'54.20%':>8} {'+5.94 dB':>10} {'9.82 dB':>10}")
    print(f"  {'LU-Net v1 2초':<40} {'50.44%':>8} {'+5.57 dB':>10} {'9.25 dB':>10}")
    print(f"  {'LU-Net v2 5초 (Attn+Res+SI-SNR)':<40} {f'{restoration_rate:.2f}%':>8} {f'+{np.mean(improvements):.2f} dB':>10} {f'{np.mean(si_snrs):.2f} dB':>10}")
    print("=" * 75)

    # v1 대비 개선폭
    delta_snr_v1 = 5.94
    delta_snr_v2 = np.mean(improvements)
    print(f"\n  v1 baseline 대비 ΔSNR 변화: {delta_snr_v2 - delta_snr_v1:+.2f} dB")

    si_snr_v1 = 9.82
    si_snr_v2 = np.mean(si_snrs)
    print(f"  v1 baseline 대비 SI-SNR 변화: {si_snr_v2 - si_snr_v1:+.2f} dB")

    # ----- 저장 -----
    np.savez(
        os.path.join(DATA_DIR, "evaluation_results_v2.npz"),
        input_snrs=input_snrs,
        output_snrs=output_snrs,
        si_snrs=si_snrs,
        improvements=improvements,
        restoration_rate=restoration_rate,
    )
    print(f"\n  상세 결과 저장됨: evaluation_results_v2.npz")


if __name__ == "__main__":
    main()