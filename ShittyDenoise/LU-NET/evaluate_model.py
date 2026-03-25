"""
================================================================
LU-Net 심음 노이즈 제거 - Step 3: 평가 + 추론
================================================================
Gemini 평가 코드 대비 수정 사항:
  1. Waveform 도메인에서 SNR 계산 (스펙트로그램 도메인 X)
     → 실제 "귀로 듣는" 음질과 일치하는 정확한 측정
  2. SI-SNR (Scale-Invariant SNR) 추가
     → 오디오 분리/향상 분야 표준 지표
  3. 추론 시 정규화 일관성 보장
     → Gemini 코드는 추론마다 다른 min/max 사용 → 여기선 불필요

사용법:
  python 03_evaluate.py

필요 라이브러리:
  pip install tensorflow numpy soundfile librosa tqdm
================================================================
"""

import os
import glob
import numpy as np
import tensorflow as tf
import soundfile as sf
import librosa
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ============================================================
# 설정
# ============================================================
DATA_DIR = r"G:\stetho_ai\LUNet_Dataset"
SR = 4000
CHUNK_LEN = SR * 5  # 20000


# ============================================================
# 1. 평가 지표 함수들
# ============================================================
def snr_db(clean, denoised):
    """
    Signal-to-Noise Ratio (dB)
    = 10 * log10( ||clean||^2 / ||clean - denoised||^2 )
    """
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum((clean - denoised) ** 2)
    if noise_power < 1e-10:
        return 50.0  # 사실상 완벽 복원
    return 10 * np.log10(signal_power / (noise_power + 1e-10))


def si_snr_db(clean, denoised):
    """
    Scale-Invariant SNR (dB) — 오디오 분리 표준 지표
    볼륨 차이에 영향받지 않고 파형의 "형태" 일치도만 측정
    """
    clean = clean - np.mean(clean)
    denoised = denoised - np.mean(denoised)

    # s_target = <denoised, clean> * clean / ||clean||^2
    dot = np.sum(denoised * clean)
    s_target = dot * clean / (np.sum(clean ** 2) + 1e-10)

    # e_noise = denoised - s_target
    e_noise = denoised - s_target

    si_snr = 10 * np.log10(
        np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + 1e-10)
    )
    return si_snr


# ============================================================
# 2. 전체 데이터셋 평가
# ============================================================
def evaluate_model():
    print("=" * 60)
    print(" LU-Net 성능 평가")
    print("=" * 60)

    # ----- 모델 & 데이터 로드 -----
    print("\n[1] 모델 및 데이터 로드 중...")
    model = tf.keras.models.load_model(
        os.path.join(DATA_DIR, "lunet_best.h5")
    )

    X = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    Y = np.load(os.path.join(DATA_DIR, "Y_train.npy"))

    # 검증 셋만 사용 (훈련에 안 쓴 데이터)
    _, X_test, _, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print(f"    평가 데이터: {X_test.shape[0]}개\n")

    # 채널 차원 추가 for model input
    X_input = X_test[..., np.newaxis]

    # ----- 모델 예측 -----
    print("[2] 모델 예측 중...")
    Y_pred = model.predict(X_input, batch_size=64, verbose=1)
    Y_pred = Y_pred.squeeze(-1)  # (N, 20000, 1) → (N, 20000)

    # ----- SNR 계산 -----
    print("\n[3] SNR 계산 중...")

    input_snrs = []    # 입력(noisy) SNR
    output_snrs = []   # 출력(denoised) SNR
    si_snrs = []       # SI-SNR
    improvements = []  # ΔSNR

    for i in tqdm(range(len(X_test)), desc="평가"):
        clean = Y_test[i]
        noisy = X_test[i]
        denoised = Y_pred[i]

        snr_in = snr_db(clean, noisy)
        snr_out = snr_db(clean, denoised)
        si = si_snr_db(clean, denoised)

        input_snrs.append(snr_in)
        output_snrs.append(snr_out)
        si_snrs.append(si)
        improvements.append(snr_out - snr_in)

    # ----- 결과 출력 -----
    print("\n" + "=" * 60)
    print("     LU-Net 노이즈 캔슬링 최종 성적표")
    print("=" * 60)
    print(f"  평가 데이터 수:      {len(X_test)}개")
    print(f"  입력 평균 SNR:       {np.mean(input_snrs):.2f} dB")
    print(f"  출력 평균 SNR:       {np.mean(output_snrs):.2f} dB")
    print("-" * 60)
    print(f"  평균 SNR 향상 (ΔSNR): +{np.mean(improvements):.2f} dB")
    print(f"  평균 SI-SNR:          {np.mean(si_snrs):.2f} dB")
    print(f"  중앙값 SNR 향상:      +{np.median(improvements):.2f} dB")
    print(f"  최대 SNR 향상:        +{np.max(improvements):.2f} dB")
    print(f"  최소 SNR 향상:        +{np.min(improvements):.2f} dB")
    print("=" * 60)

    # 결과 저장
    np.savez(
        os.path.join(DATA_DIR, "evaluation_results.npz"),
        input_snrs=input_snrs,
        output_snrs=output_snrs,
        si_snrs=si_snrs,
        improvements=improvements,
    )
    print(f"\n  상세 결과 저장됨: evaluation_results.npz")

    return np.mean(improvements)


# ============================================================
# 3. 단일 파일 노이즈 제거 (추론)
# ============================================================
def denoise_audio(input_path, output_path):
    """
    wav 파일 하나를 노이즈 제거하여 저장

    [Gemini 대비 개선점]
    - Mel-Spectrogram 변환/역변환 없음 → 음질 손실 없음
    - Griffin-Lim 불필요 → "물먹은 소리" 제거
    - 별도 스케일링 불필요 → 정규화 일관성 문제 없음
    """
    print(f"\n  입력: {os.path.basename(input_path)}")

    # 모델 로드
    model = tf.keras.models.load_model(
        os.path.join(DATA_DIR, "lunet_best.h5")
    )

    # 오디오 로드
    y, _ = librosa.load(input_path, sr=SR)
    original_length = len(y)

    # 5초 단위로 자르기 (긴 파일 대응)
    chunks = []
    for i in range(0, len(y), CHUNK_LEN):
        chunk = y[i: i + CHUNK_LEN]
        if len(chunk) < CHUNK_LEN:
            # 마지막 청크: 패딩
            chunk = np.pad(chunk, (0, CHUNK_LEN - len(chunk)))
        chunks.append(chunk)

    # 각 청크 denoising
    denoised_chunks = []
    for chunk in chunks:
        x = chunk[np.newaxis, :, np.newaxis]  # (1, 20000, 1)
        pred = model.predict(x, verbose=0)
        denoised_chunks.append(pred.squeeze())

    # 이어붙이기 & 원본 길이로 자르기
    y_clean = np.concatenate(denoised_chunks)[:original_length]

    # Peak normalize
    max_val = np.max(np.abs(y_clean))
    if max_val > 0:
        y_clean = y_clean / max_val * 0.9

    # 저장
    sf.write(output_path, y_clean, SR, subtype="PCM_16")
    print(f"  출력: {os.path.basename(output_path)}")
    print(f"  길이: {len(y_clean) / SR:.1f}초")


# ============================================================
# 4. 메인
# ============================================================
def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--denoise":
        # 단일 파일 denoising 모드
        # 사용법: python 03_evaluate.py --denoise input.wav output.wav
        if len(sys.argv) < 4:
            print("사용법: python 03_evaluate.py --denoise input.wav output.wav")
            return
        denoise_audio(sys.argv[2], sys.argv[3])
    else:
        # 기본: 전체 평가 모드
        avg_improvement = evaluate_model()

        # 테스트 파일이 있으면 자동으로 하나 denoising 해보기
        test_dir = r"G:\stetho_ai\Preprocessed_Mix_Test"
        if os.path.exists(test_dir):
            wav_files = glob.glob(os.path.join(test_dir, "*.wav"))
            if wav_files:
                print("\n\n[보너스] 실제 파일 denoising 테스트:")
                denoise_audio(
                    wav_files[0],
                    os.path.join(DATA_DIR, "denoised_lunet_result.wav"),
                )
                print("\n  원본과 비교해서 들어보세요!")


if __name__ == "__main__":
    main()