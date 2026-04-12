"""
================================================================
LU-Net 심음 노이즈 제거 - Step 3: 평가 (2초 버전)
================================================================
5초 vs 2초 비교표 자동 출력
================================================================
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ============================================================
# 설정
# ============================================================
DATA_DIR = r"G:\stetho_ai\LUNet_Dataset_2sec"
MODEL_PATH = os.path.join(DATA_DIR, "lunet_best_2sec.h5")


# ============================================================
# 평가 지표
# ============================================================
def snr_db(clean, signal):
    s_power = np.sum(clean ** 2)
    n_power = np.sum((clean - signal) ** 2)
    if n_power < 1e-10:
        return 50.0
    return 10 * np.log10(s_power / (n_power + 1e-10))


def si_snr_db(clean, denoised):
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
    print(" LU-Net 성능 평가 (2초 버전)")
    print("=" * 60)

    # ----- 로드 -----
    print("\n[1] 모델 및 데이터 로드 중...")
    model = tf.keras.models.load_model(MODEL_PATH)

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

    # ----- 복원율 (Gemini 방식 MAE 기반) -----
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
        si_snrs.append(si_snr_db(clean, denoised))

    improvements = np.array(output_snrs) - np.array(input_snrs)

    # ----- 결과 출력 -----
    print("\n" + "=" * 60)
    print("     LU-Net 2초 버전 최종 성적표")
    print("=" * 60)
    print(f"  평가 데이터 수:        {len(X_test)}개")
    print(f"  초기 평균 노이즈 오차: {original_noise:.4f}")
    print(f"  모델 통과 후 잔여 오차: {remaining_noise:.4f}")
    print("-" * 60)
    print(f"  복원율 (노이즈 제거율): {restoration_rate:.2f} %")
    print(f"  평균 SNR 향상 (ΔSNR):  +{np.mean(improvements):.2f} dB")
    print(f"  평균 SI-SNR:           {np.mean(si_snrs):.2f} dB")
    print(f"  중앙값 SNR 향상:       +{np.median(improvements):.2f} dB")
    print("=" * 60)

    # ----- 5초 vs 2초 비교표 -----
    print()
    print("=" * 65)
    print("                    5초 vs 2초 비교표")
    print("=" * 65)
    print(f"  {'모델':<35} {'복원율':>8} {'ΔSNR':>10} {'SI-SNR':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Gemini Bi-LSTM (스펙트로그램)':<35} {'53.65%':>8} {'+4.86 dB':>10} {'N/A':>10}")
    print(f"  {'Gemini 마스킹 CRNN':<35} {'54.10%':>8} {'+5.22 dB':>10} {'N/A':>10}")
    print(f"  {'LU-Net 5초 (waveform)':<35} {'54.20%':>8} {'+5.94 dB':>10} {'9.82 dB':>10}")
    print(f"  {'LU-Net 2초 (waveform)':<35} {f'{restoration_rate:.2f}%':>8} {f'+{np.mean(improvements):.2f} dB':>10} {f'{np.mean(si_snrs):.2f} dB':>10}")
    print("=" * 65)

    # ----- 저장 -----
    np.savez(
        os.path.join(DATA_DIR, "evaluation_results_2sec.npz"),
        input_snrs=input_snrs,
        output_snrs=output_snrs,
        si_snrs=si_snrs,
        improvements=improvements,
        restoration_rate=restoration_rate,
    )
    print(f"\n  상세 결과 저장됨: evaluation_results_2sec.npz")


if __name__ == "__main__":
    main()