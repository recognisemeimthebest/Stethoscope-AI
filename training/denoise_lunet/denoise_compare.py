"""
v1 vs v2 청음 비교 - 동일 파일로 두 모델 결과를 나란히 생성
"""
import os
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
import tensorflow.keras.backend as K

SR = 4000
CHUNK_LEN = SR * 5  # 20000

INPUT_FILE = r"G:\stetho_ai\Preprocessed_Mix_Test\mixed_sample_7_SNR_-5dB.wav"
OUTPUT_DIR = r"G:\stetho_ai\LUNet_Dataset"

# v2 로드에 필요한 커스텀 손실 함수
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


def denoise_with_model(model, audio, sr, chunk_len):
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
        x = chunk[np.newaxis, :, np.newaxis]  # (1, 20000, 1)
        pred = model.predict(x, verbose=0)
        denoised.append(pred.squeeze())

    result = np.concatenate(denoised)[:original_length]

    # Peak normalize
    max_val = np.max(np.abs(result))
    if max_val > 0:
        result = result / max_val * 0.9
    return result


def main():
    print("=" * 60)
    print(" v1 vs v2 청음 비교")
    print("=" * 60)

    # ----- 입력 파일 로드 -----
    print(f"\n입력: {os.path.basename(INPUT_FILE)}")
    y, _ = librosa.load(INPUT_FILE, sr=SR)
    print(f"  길이: {len(y)/SR:.1f}초, SR: {SR}Hz")

    # ----- v1 모델 -----
    print("\n[v1] LU-Net baseline 로드 중...")
    model_v1 = tf.keras.models.load_model(
        os.path.join(OUTPUT_DIR, "lunet_best.h5")
    )
    y_v1 = denoise_with_model(model_v1, y, SR, CHUNK_LEN)

    out_v1 = os.path.join(OUTPUT_DIR, "denoised_v1_SNR-5dB.wav")
    sf.write(out_v1, y_v1, SR, subtype="PCM_16")
    print(f"  저장: {os.path.basename(out_v1)}")

    # ----- v2 모델 -----
    print("\n[v2] LU-Net v2 (Attn+Res+SI-SNR) 로드 중...")
    model_v2 = tf.keras.models.load_model(
        os.path.join(OUTPUT_DIR, "lunet_v2_best.h5"),
        custom_objects={
            "combined_loss": combined_loss,
            "si_snr_loss": si_snr_loss,
        }
    )
    y_v2 = denoise_with_model(model_v2, y, SR, CHUNK_LEN)

    out_v2 = os.path.join(OUTPUT_DIR, "denoised_v2_SNR-5dB.wav")
    sf.write(out_v2, y_v2, SR, subtype="PCM_16")
    print(f"  저장: {os.path.basename(out_v2)}")

    # ----- 원본도 같은 SR로 저장 (비교용) -----
    y_norm = y.copy()
    max_val = np.max(np.abs(y_norm))
    if max_val > 0:
        y_norm = y_norm / max_val * 0.9

    out_orig = os.path.join(OUTPUT_DIR, "original_noisy_SNR-5dB.wav")
    sf.write(out_orig, y_norm, SR, subtype="PCM_16")
    print(f"\n  원본(노이즈): {os.path.basename(out_orig)}")

    print("\n" + "=" * 60)
    print(" 3개 파일을 순서대로 들어보세요:")
    print(f"  1. {os.path.basename(out_orig)}  (노이즈 원본)")
    print(f"  2. {os.path.basename(out_v1)}    (v1 baseline)")
    print(f"  3. {os.path.basename(out_v2)}    (v2 Attn+Res)")
    print("=" * 60)


if __name__ == "__main__":
    main()