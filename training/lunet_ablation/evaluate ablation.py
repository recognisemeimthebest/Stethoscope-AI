"""
================================================================
전체 Ablation Study 평가 - 4개 모델 비교
================================================================
LU-Net v1 (LSTM + MSE)
LU-Net v2 (LSTM + Attn + Res + 90:10)
TU-Net v1 (TCN + MSE)
TU-Net v2 (TCN + Attn + Res + 90:10)
================================================================
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATA_DIR = r"G:\stetho_ai\LUNet_Dataset"


# ============================================================
# Custom loss (모델 로드용)
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
    return 0.9 * mse + 0.1 * (si_snr / 20.0)

CUSTOM = {"combined_loss": combined_loss, "si_snr_loss": si_snr_loss}


# ============================================================
# 평가 지표
# ============================================================
def snr_db(clean, signal):
    s = np.sum(clean ** 2)
    n = np.sum((clean - signal) ** 2)
    if n < 1e-10: return 50.0
    return 10 * np.log10(s / (n + 1e-10))

def si_snr_np(clean, denoised):
    clean = clean - np.mean(clean)
    denoised = denoised - np.mean(denoised)
    dot = np.sum(denoised * clean)
    s_target = dot * clean / (np.sum(clean ** 2) + 1e-10)
    e_noise = denoised - s_target
    return 10 * np.log10(np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + 1e-10))


def evaluate_model(model, X_test, Y_test, name):
    """단일 모델 평가"""
    X_input = X_test[..., np.newaxis]
    Y_pred = model.predict(X_input, batch_size=64, verbose=0)
    Y_pred = Y_pred.squeeze(-1)

    original_noise = np.mean(np.abs(X_test - Y_test))
    remaining_noise = np.mean(np.abs(Y_pred - Y_test))
    restoration = (1 - remaining_noise / original_noise) * 100

    improvements = []
    si_snrs = []
    for i in range(len(X_test)):
        inp_snr = snr_db(Y_test[i], X_test[i])
        out_snr = snr_db(Y_test[i], Y_pred[i])
        improvements.append(out_snr - inp_snr)
        si_snrs.append(si_snr_np(Y_test[i], Y_pred[i]))

    return {
        "name": name,
        "restoration": restoration,
        "delta_snr": np.mean(improvements),
        "si_snr": np.mean(si_snrs),
        "median_snr": np.median(improvements),
    }


# ============================================================
# 메인
# ============================================================
def main():
    print("=" * 75)
    print("           Ablation Study - 전체 모델 비교")
    print("=" * 75)

    # 데이터
    print("\n데이터 로드 중...")
    X = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    Y = np.load(os.path.join(DATA_DIR, "Y_train.npy"))
    _, X_test, _, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(f"  평가 데이터: {X_test.shape[0]}개\n")

    # 모델 목록
    models_to_eval = [
        ("lunet_v2_90_10.h5", "LU-Net v2 (LSTM+Attn+Res+90:10)", CUSTOM),
        ("tunet_v1_best.h5", "TU-Net v1 (TCN+MSE)", None),
        ("tunet_v2_best.h5", "TU-Net v2 (TCN+Attn+Res+90:10)", CUSTOM),
    ]

    results = []

    # 기존 결과 하드코딩 (v1 모델 파일이 덮어씌워졌으므로)
    results.append({
        "name": "LU-Net v1 (LSTM+MSE)",
        "restoration": 54.20,
        "delta_snr": 5.94,
        "si_snr": 9.82,
        "median_snr": 5.90,
    })

    for filename, name, custom in models_to_eval:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"  [{name}] 모델 파일 없음, 건너뜀")
            continue

        print(f"  [{name}] 평가 중...")
        if custom:
            model = tf.keras.models.load_model(path, custom_objects=custom)
        else:
            model = tf.keras.models.load_model(path)

        r = evaluate_model(model, X_test, Y_test, name)
        results.append(r)
        print(f"    ΔSNR: +{r['delta_snr']:.2f} dB, SI-SNR: {r['si_snr']:.2f} dB")

    # 비교표
    print("\n\n")
    print("=" * 80)
    print("                      Ablation Study 결과")
    print("=" * 80)
    print(f"  {'모델':<42} {'복원율':>8} {'ΔSNR':>10} {'SI-SNR':>10}")
    print(f"  {'-'*72}")

    # 외부 비교군
    print(f"  {'Gemini Bi-LSTM (스펙트로그램)':<42} {'53.65%':>8} {'+4.86 dB':>10} {'N/A':>10}")
    print(f"  {'원본 LU-Net 논문 (Ali 2023)':<42} {'N/A':>8} {'+5.575 dB':>10} {'N/A':>10}")
    print(f"  {'-'*72}")

    for r in results:
        print(f"  {r['name']:<42} {f\"{r['restoration']:.2f}%\":>8} {f\"+{r['delta_snr']:.2f} dB\":>10} {f\"{r['si_snr']:.2f} dB\":>10}")

    print("=" * 80)

    # LSTM vs TCN 비교
    lstm_v1 = next((r for r in results if "LU-Net v1" in r["name"]), None)
    tcn_v1 = next((r for r in results if "TU-Net v1" in r["name"]), None)
    lstm_v2 = next((r for r in results if "LU-Net v2" in r["name"]), None)
    tcn_v2 = next((r for r in results if "TU-Net v2" in r["name"]), None)

    if lstm_v1 and tcn_v1:
        print(f"\n  [LSTM vs TCN - v1 (순수 구조)]")
        print(f"    ΔSNR: LSTM {lstm_v1['delta_snr']:+.2f} vs TCN {tcn_v1['delta_snr']:+.2f} ({tcn_v1['delta_snr']-lstm_v1['delta_snr']:+.2f} dB)")
        print(f"    SI-SNR: LSTM {lstm_v1['si_snr']:.2f} vs TCN {tcn_v1['si_snr']:.2f} ({tcn_v1['si_snr']-lstm_v1['si_snr']:+.2f} dB)")

    if lstm_v2 and tcn_v2:
        print(f"\n  [LSTM vs TCN - v2 (Attn+Res+90:10)]")
        print(f"    ΔSNR: LSTM {lstm_v2['delta_snr']:+.2f} vs TCN {tcn_v2['delta_snr']:+.2f} ({tcn_v2['delta_snr']-lstm_v2['delta_snr']:+.2f} dB)")
        print(f"    SI-SNR: LSTM {lstm_v2['si_snr']:.2f} vs TCN {tcn_v2['si_snr']:.2f} ({tcn_v2['si_snr']-lstm_v2['si_snr']:+.2f} dB)")

    print("=" * 80)


if __name__ == "__main__":
    main()