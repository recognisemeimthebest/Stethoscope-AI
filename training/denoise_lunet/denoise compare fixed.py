"""
v2 (90:10) vs v2 (50:50) 청음 비교
"""
import os
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
import tensorflow.keras.backend as K

SR = 4000
CHUNK_LEN = SR * 5

INPUT_FILE = r"G:\stetho_ai\Preprocessed_Mix_Test\mixed_sample_7_SNR_-5dB.wav"
OUTPUT_DIR = r"G:\stetho_ai\LUNet_Dataset"


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


def combined_loss_5050(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))
    si_snr = si_snr_loss(y_true, y_pred)
    return 0.5 * mse + 0.5 * (si_snr / 20.0)


CUSTOM_OBJECTS = {
    "combined_loss": combined_loss,
    "si_snr_loss": si_snr_loss,
}

CUSTOM_OBJECTS_5050 = {
    "combined_loss": combined_loss_5050,
    "si_snr_loss": si_snr_loss,
}


def denoise_with_model(model, audio, chunk_len):
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


def main():
    print("=" * 60)
    print(" v2 (90:10) vs v2 (50:50) 청음 비교")
    print("=" * 60)

    # 입력
    print(f"\n입력: {os.path.basename(INPUT_FILE)}")
    y, _ = librosa.load(INPUT_FILE, sr=SR)
    print(f"  길이: {len(y)/SR:.1f}초, SR: {SR}Hz")

    # ----- v2 90:10 -----
    path_9010 = os.path.join(OUTPUT_DIR, "lunet_v2_90_10.h5")
    print(f"\n[1] v2 (90:10) 로드 중... {os.path.basename(path_9010)}")
    model_9010 = tf.keras.models.load_model(path_9010, custom_objects=CUSTOM_OBJECTS)
    y_9010 = denoise_with_model(model_9010, y, CHUNK_LEN)

    out_9010 = os.path.join(OUTPUT_DIR, "denoised_v2_90_10_SNR-5dB.wav")
    sf.write(out_9010, y_9010, SR, subtype="PCM_16")
    print(f"  저장: {os.path.basename(out_9010)}")

    # ----- v2 50:50 -----
    path_5050 = os.path.join(OUTPUT_DIR, "lunet_v2_best.h5")
    if os.path.exists(path_5050):
        print(f"\n[2] v2 (50:50) 로드 중... {os.path.basename(path_5050)}")
        model_5050 = tf.keras.models.load_model(path_5050, custom_objects=CUSTOM_OBJECTS_5050)
        y_5050 = denoise_with_model(model_5050, y, CHUNK_LEN)

        out_5050 = os.path.join(OUTPUT_DIR, "denoised_v2_50_50_SNR-5dB.wav")
        sf.write(out_5050, y_5050, SR, subtype="PCM_16")
        print(f"  저장: {os.path.basename(out_5050)}")
    else:
        print(f"\n[2] v2 (50:50) 파일 없음, 건너뜀")
        out_5050 = None

    # ----- 원본 -----
    y_norm = y.copy()
    max_val = np.max(np.abs(y_norm))
    if max_val > 0:
        y_norm = y_norm / max_val * 0.9

    out_orig = os.path.join(OUTPUT_DIR, "original_noisy_SNR-5dB.wav")
    sf.write(out_orig, y_norm, SR, subtype="PCM_16")

    # 결과
    print("\n" + "=" * 60)
    print(" 파일을 순서대로 들어보세요:")
    print(f"  1. {os.path.basename(out_orig)}           (노이즈 원본)")
    print(f"  2. {os.path.basename(out_9010)}   (v2 90:10 - ΔSNR +5.59, SI-SNR 12.76)")
    if out_5050:
        print(f"  3. {os.path.basename(out_5050)}   (v2 50:50 - ΔSNR +0.12, SI-SNR 12.44)")
    print("=" * 60)


if __name__ == "__main__":
    main()