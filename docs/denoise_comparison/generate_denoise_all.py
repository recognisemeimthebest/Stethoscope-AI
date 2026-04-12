"""
디노이즈 모델 3종 성능 시각화 + WAV 저장
- LU-Net v2: LSTM + Attn + Res + SI-SNR
- TU-Net v1: TCN + MSE
- TU-Net v2: TCN + Attn + Res + SI-SNR
"""
import os
import sys
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# TF 로깅 억제
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow.keras.backend as K

# ── 경로 ──
DATASET_DIR = r"G:\stetho_ai\_misc\datasets\LUNet_Dataset"
HEART_DATASET = r"G:\stetho_ai\_misc\datasets\classification-of-heart-sound-recordings\classification-of-heart-sound-recordings\training-a"
CLEAN_FILE = os.path.join(HEART_DATASET, "a0007.wav")
SAVE_DIR = r"G:\stetho_ai\denoise_demo"
os.makedirs(SAVE_DIR, exist_ok=True)

# LU-Net/TU-Net은 4000Hz, 20000 samples (5초), 파형 기반
SR = 4000
DURATION = 5
CHUNK_LEN = SR * DURATION  # 20000


# ── 커스텀 loss 등록 ──
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


custom_objects = {
    'si_snr_loss': si_snr_loss,
    'combined_loss': combined_loss,
}


def snr_db(clean, signal):
    noise = signal - clean
    return 10 * np.log10(np.sum(clean**2) / (np.sum(noise**2) + 1e-10))


def denoise_waveform(model, y_noisy):
    """파형 기반 모델 추론 (20000 samples → 20000 samples)"""
    x = y_noisy.reshape(1, -1, 1).astype(np.float32)
    pred = model.predict(x, verbose=0)
    return pred.squeeze()


# ── 1. 원본 + 노이즈 ──
print("1. 원본 심음 로드 + 노이즈 합성...")
y_clean, _ = librosa.load(CLEAN_FILE, sr=SR, duration=DURATION)
if len(y_clean) < CHUNK_LEN:
    y_clean = np.pad(y_clean, (0, CHUNK_LEN - len(y_clean)))
else:
    y_clean = y_clean[:CHUNK_LEN]
peak = np.max(np.abs(y_clean))
if peak > 0:
    y_clean = y_clean / peak * 0.9

np.random.seed(42)
white_noise = np.random.randn(CHUNK_LEN) * 0.08
t = np.linspace(0, DURATION, CHUNK_LEN)
hum = 0.05 * np.sin(2 * np.pi * 50 * t)
y_noisy = np.clip(y_clean + white_noise + hum, -1, 1).astype(np.float32)

# ── 2. 모델별 추론 ──
models_config = [
    ("LU-Net v2\nLSTM + Attn + Res + SI-SNR",
     os.path.join(DATASET_DIR, "lunet_v2_90_10.h5"),
     "4_denoised_lunet_v2.wav", "#58a6ff"),
    ("TU-Net v1\nTCN + MSE",
     os.path.join(DATASET_DIR, "tunet_v1_best.h5"),
     "5_denoised_tunet_v1.wav", "#f0883e"),
    ("TU-Net v2\nTCN + Attn + Res + SI-SNR",
     os.path.join(DATASET_DIR, "tunet_v2_best.h5"),
     "6_denoised_tunet_v2.wav", "#a371f7"),
]

results = []
for name, model_path, wav_name, color in models_config:
    print(f"\n2. {name.split(chr(10))[0]} 로드 및 추론...")
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    y_den = denoise_waveform(model, y_noisy)

    # 길이 맞추기
    min_len = min(len(y_clean), len(y_den))
    y_den = y_den[:min_len]

    # 정규화
    d_peak = np.max(np.abs(y_den))
    if d_peak > 0:
        y_den = y_den / d_peak * 0.9

    # SNR
    snr_val = snr_db(y_clean[:min_len], y_den)
    snr_noisy = snr_db(y_clean[:min_len], y_noisy[:min_len])
    delta_snr = snr_val - snr_noisy
    print(f"   SNR: {snr_val:.1f} dB (delta: +{delta_snr:.1f} dB)")

    # WAV 저장
    sf.write(os.path.join(SAVE_DIR, wav_name), y_den, SR, subtype='PCM_16')

    results.append((name, y_den, snr_val, delta_snr, color, min_len))

    # 메모리 해제
    del model
    tf.keras.backend.clear_session()

# 원본/노이즈도 저장 (4000Hz 버전)
sf.write(os.path.join(SAVE_DIR, "1_clean_4k.wav"), y_clean, SR, subtype='PCM_16')
sf.write(os.path.join(SAVE_DIR, "2_noisy_4k.wav"), y_noisy, SR, subtype='PCM_16')

snr_noisy_val = snr_db(y_clean, y_noisy)

# ── 3. 시각화 (모델당 1행, 5행 총) ──
print("\n3. 시각화 생성 중...")

n_rows = 2 + len(results)  # clean + noisy + 3 models
fig, axes = plt.subplots(n_rows, 2, figsize=(20, n_rows * 3.2))
fig.patch.set_facecolor('#0d1117')

all_signals = [
    ("원본 심음 (Clean)", y_clean, "#58a6ff", None, None),
    (f"노이즈 추가 (SNR: {snr_noisy_val:.1f} dB)", y_noisy, "#f85149", None, None),
]
for name, y_den, snr_val, delta_snr, color, min_len in results:
    label = f"{name.split(chr(10))[0]} (SNR: {snr_val:.1f} dB, +{delta_snr:.1f})"
    all_signals.append((label, y_den, color, snr_val, delta_snr))

t_axis = np.linspace(0, DURATION, CHUNK_LEN)

for row, (label, y, color, snr_val, delta_snr) in enumerate(all_signals):
    y_plot = y[:CHUNK_LEN]

    # 파형
    ax1 = axes[row][0]
    ax1.set_facecolor('#161b22')
    ax1.plot(t_axis[:len(y_plot)], y_plot, color=color, linewidth=0.4)
    ax1.set_ylabel(label.split('(')[0].strip(), fontsize=11, fontweight='bold',
                    color=color, labelpad=5)
    ax1.set_ylim(-1, 1)
    ax1.set_xlabel('시간 (초)', fontsize=9, color='#aaa')
    ax1.tick_params(colors='#aaa', labelsize=8)
    for spine in ax1.spines.values():
        spine.set_color('#444')
    if row == 0:
        ax1.set_title('파형 (Waveform)', fontsize=14, fontweight='bold',
                       color='white', pad=10)

    # Mel-spectrogram
    ax2 = axes[row][1]
    ax2.set_facecolor('#161b22')
    S = librosa.feature.melspectrogram(y=y_plot, sr=SR, n_mels=64,
                                        hop_length=256, fmax=SR // 2)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=SR, hop_length=256,
                                    x_axis='time', y_axis='mel',
                                    ax=ax2, cmap='cividis', fmax=SR // 2)
    ax2.set_xlabel('시간 (초)', fontsize=9, color='#aaa')
    ax2.set_ylabel('Mel', fontsize=9, color='#aaa')
    ax2.tick_params(colors='#aaa', labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color('#444')
    if row == 0:
        ax2.set_title('Mel-Spectrogram', fontsize=14, fontweight='bold',
                       color='white', pad=10)

    # SNR 표시 (오른쪽 상단)
    if snr_val is not None:
        ax2.text(0.98, 0.92, f'SNR: {snr_val:.1f} dB\n(+{delta_snr:.1f})',
                 transform=ax2.transAxes, fontsize=10, color=color,
                 ha='right', va='top', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117', alpha=0.8))

fig.suptitle('디노이즈 모델 성능 비교  —  LU-Net v2 / TU-Net v1 / TU-Net v2',
             fontsize=20, fontweight='bold', color='white', y=0.99)

plt.tight_layout(rect=[0, 0, 1, 0.97])

# 각 모델별 이미지 개별 저장 + 통합 이미지
for i, (name, y_den, snr_val, delta_snr, color, min_len) in enumerate(results):
    short_name = name.split('\n')[0].replace(' ', '_').replace('-', '')
    # 개별 이미지
    fig_single, ax_s = plt.subplots(2, 2, figsize=(18, 9))
    fig_single.patch.set_facecolor('#0d1117')

    single_data = [
        ("원본 (Clean)", y_clean, "#58a6ff"),
        (f"노이즈 (SNR: {snr_noisy_val:.1f} dB)", y_noisy, "#f85149"),
        (f"{name.split(chr(10))[0]} 디노이즈", y_den, color),
    ]

    for r in range(2):
        for c in range(2):
            ax_s[r][c].set_facecolor('#161b22')
            ax_s[r][c].tick_params(colors='#aaa', labelsize=8)
            for spine in ax_s[r][c].spines.values():
                spine.set_color('#444')

    # 상단 좌: 원본 파형 vs 디노이즈 파형 겹치기
    ax_s[0][0].plot(t_axis, y_clean, color='#58a6ff', linewidth=0.5, alpha=0.6, label='원본')
    ax_s[0][0].plot(t_axis[:len(y_den)], y_den, color=color, linewidth=0.5, alpha=0.8, label='디노이즈')
    ax_s[0][0].set_title('원본 vs 디노이즈 파형', fontsize=13, fontweight='bold', color='white')
    ax_s[0][0].set_ylim(-1, 1)
    ax_s[0][0].legend(fontsize=10, loc='upper right', facecolor='#161b22', edgecolor='#444', labelcolor='white')

    # 상단 우: 노이즈 파형 vs 디노이즈 파형
    ax_s[0][1].plot(t_axis, y_noisy, color='#f85149', linewidth=0.3, alpha=0.5, label='노이즈')
    ax_s[0][1].plot(t_axis[:len(y_den)], y_den, color=color, linewidth=0.5, alpha=0.8, label='디노이즈')
    ax_s[0][1].set_title('노이즈 vs 디노이즈 파형', fontsize=13, fontweight='bold', color='white')
    ax_s[0][1].set_ylim(-1, 1)
    ax_s[0][1].legend(fontsize=10, loc='upper right', facecolor='#161b22', edgecolor='#444', labelcolor='white')

    # 하단 좌: 원본 mel
    S_clean = librosa.power_to_db(librosa.feature.melspectrogram(y=y_clean, sr=SR, n_mels=64, hop_length=256, fmax=SR//2), ref=np.max)
    librosa.display.specshow(S_clean, sr=SR, hop_length=256, x_axis='time', y_axis='mel', ax=ax_s[1][0], cmap='cividis', fmax=SR//2)
    ax_s[1][0].set_title('원본 Mel-Spectrogram', fontsize=13, fontweight='bold', color='#58a6ff')

    # 하단 우: 디노이즈 mel
    S_den = librosa.power_to_db(librosa.feature.melspectrogram(y=y_den, sr=SR, n_mels=64, hop_length=256, fmax=SR//2), ref=np.max)
    librosa.display.specshow(S_den, sr=SR, hop_length=256, x_axis='time', y_axis='mel', ax=ax_s[1][1], cmap='cividis', fmax=SR//2)
    ax_s[1][1].set_title(f'디노이즈 Mel-Spectrogram (SNR: {snr_val:.1f} dB)', fontsize=13, fontweight='bold', color=color)

    model_title = name.replace('\n', ' — ')
    fig_single.suptitle(f'{model_title}  |  SNR 개선: +{delta_snr:.1f} dB',
                         fontsize=17, fontweight='bold', color='white', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_name = f"denoise_{short_name}.png"
    fig_single.savefig(os.path.join(r"G:\stetho_ai", save_name), dpi=300,
                        bbox_inches='tight', facecolor=fig_single.get_facecolor())
    print(f"   Saved: {save_name}")
    plt.close(fig_single)

# 통합 이미지 저장
fig.savefig(r"G:\stetho_ai\denoise_all_comparison.png", dpi=300,
            bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved: denoise_all_comparison.png")
plt.close(fig)

# ── 결과 요약 ──
print("\n" + "=" * 50)
print("디노이즈 성능 요약")
print("=" * 50)
print(f"{'모델':<30} {'SNR (dB)':>10} {'Delta':>10}")
print("-" * 50)
print(f"{'노이즈':<30} {snr_noisy_val:>10.1f} {'':>10}")
for name, _, snr_val, delta_snr, _, _ in results:
    short = name.split('\n')[0]
    print(f"{short:<30} {snr_val:>10.1f} {'+' + f'{delta_snr:.1f}':>10}")
print("=" * 50)

print(f"\nWAV 파일 저장 위치: {SAVE_DIR}")
print("완료!")
