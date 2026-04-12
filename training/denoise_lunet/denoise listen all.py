"""
================================================================
4개 모델 청음 비교 - Ablation Study 전체
================================================================
LU-Net v2 (LSTM+Attn+Res+90:10)
TU-Net v1 (TCN+MSE)
TU-Net v2 (TCN+Attn+Res+90:10)
+ 노이즈 원본

입력: Preprocessed_Mix_Test 폴더의 테스트 파일
출력: G:\stetho_ai\LUNet_Dataset\ 에 wav 파일 생성
================================================================
"""

import os
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
import tensorflow.keras.backend as K

SR = 4000
CHUNK_LEN = SR * 5  # 20000

INPUT_FILE = r"G:\stetho_ai\_misc\datasets\Preprocessed_Mix_Test\mixed_sample_7_SNR_-5dB.wav"
OUTPUT_DIR = r"G:\stetho_ai\_misc\datasets\LUNet_Dataset"


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
# 디노이징
# ============================================================
def denoise_with_model(model, audio, chunk_len):
    """5초 단위로 잘라서 denoising 후 이어붙이기"""
    original_length = len(audio)
    chunks = []
    for i in range(0, len(audio), chunk_len):
        chunk = audio[i: i + chunk_len]
        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
        chunks.append(chunk)

    denoised = []
    for chunk in chunks:
        x = chunk[np.newaxis, :, np.newaxis]
        pred = model.predict(x, verbose=0)
        denoised.append(pred.squeeze())

    result = np.concatenate(denoised)[:original_length]
    max_val = np.max(np.abs(result))
    if max_val > 0:
        result = result / max_val * 0.9
    return result


# ============================================================
# 메인
# ============================================================
def main():
    print("=" * 60)
    print(" 4개 모델 청음 비교")
    print("=" * 60)

    # 입력
    print(f"\n입력: {os.path.basename(INPUT_FILE)}")
    y, _ = librosa.load(INPUT_FILE, sr=SR)
    print(f"  길이: {len(y)/SR:.1f}초, SR: {SR}Hz")

    # 모델 목록: (파일명, 출력명, custom_objects, 설명)
    models = [
        ("lunet_v2_90_10.h5", "denoised_lunet_v2", CUSTOM,
         "LU-Net v2 (LSTM+Attn+Res+90:10) - ΔSNR +5.59, SI-SNR 12.76"),
        ("tunet_v1_best.h5", "denoised_tunet_v1", None,
         "TU-Net v1 (TCN+MSE) - ΔSNR +4.35, SI-SNR 8.14"),
        ("tunet_v2_best.h5", "denoised_tunet_v2", CUSTOM,
         "TU-Net v2 (TCN+Attn+Res+90:10) - ΔSNR +5.63, SI-SNR 12.59"),
    ]

    outputs = []

    for filename, out_name, custom, desc in models:
        path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(path):
            print(f"\n  [{desc}] 모델 파일 없음, 건너뜀")
            continue

        print(f"\n  [{desc}]")
        print(f"    로드 중... {filename}")
        if custom:
            model = tf.keras.models.load_model(path, custom_objects=custom)
        else:
            model = tf.keras.models.load_model(path)

        y_denoised = denoise_with_model(model, y, CHUNK_LEN)

        out_path = os.path.join(OUTPUT_DIR, f"{out_name}_SNR-5dB.wav")
        sf.write(out_path, y_denoised, SR, subtype="PCM_16")
        print(f"    저장: {os.path.basename(out_path)}")
        outputs.append((out_path, desc))

        # 메모리 해제
        del model
        tf.keras.backend.clear_session()

    # 원본 저장
    y_norm = y.copy()
    max_val = np.max(np.abs(y_norm))
    if max_val > 0:
        y_norm = y_norm / max_val * 0.9
    out_orig = os.path.join(OUTPUT_DIR, "original_noisy_SNR-5dB.wav")
    sf.write(out_orig, y_norm, SR, subtype="PCM_16")

    # 결과 안내
    print("\n" + "=" * 60)
    print(" 파일을 순서대로 들어보세요:")
    print(f"  0. {os.path.basename(out_orig)}  (노이즈 원본)")
    for i, (path, desc) in enumerate(outputs, 1):
        print(f"  {i}. {os.path.basename(path)}")
        print(f"     → {desc}")
    print("=" * 60)

    print("\n [청음 포인트]")
    print("  - S1/S2 심음이 또렷하게 들리는가?")
    print("  - 배경 노이즈가 얼마나 줄었는가?")
    print("  - 심음 파형이 왜곡 없이 보존되었는가?")
    print("  - v1(MSE only) vs v2(+Attn+Res+SI-SNR) 차이")
    print("  - LSTM vs TCN 차이")


if __name__ == "__main__":
    main()