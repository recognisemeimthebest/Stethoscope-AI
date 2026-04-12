"""
LU-Net v1 (LSTM + MSE) 디노이즈 성능 — 개별 이미지 + WAV 저장
generate_denoise_all.py와 동일 형식
"""
import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

DATASET_DIR = r"G:\stetho_ai\_misc\datasets\LUNet_Dataset"
HEART_DATASET = r"G:\stetho_ai\_misc\datasets\classification-of-heart-sound-recordings\classification-of-heart-sound-recordings\training-a"
CLEAN_FILE = os.path.join(HEART_DATASET, "a0007.wav")
SAVE_DIR = r"G:\stetho_ai\denoise_demo"
MODEL_PATH = os.path.join(DATASET_DIR, "lunet_best.h5")

SR = 4000
DURATION = 5
CHUNK_LEN = SR * DURATION


def snr_db(clean, signal):
    noise = signal - clean
    return 10 * np.log10(np.sum(clean**2) / (np.sum(noise**2) + 1e-10))


# ── 1. 원본 + 노이즈 (동일 시드) ──
print("1. 원본 + 노이즈 준비...")
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

snr_noisy = snr_db(y_clean, y_noisy)

# ── 2. 모델 추론 ──
print("2. LU-Net v1 로드 및 추론...")
model = tf.keras.models.load_model(MODEL_PATH)
x_input = y_noisy.reshape(1, -1, 1).astype(np.float32)
y_den = model.predict(x_input, verbose=0).squeeze()

min_len = min(len(y_clean), len(y_den))
y_den = y_den[:min_len]
d_peak = np.max(np.abs(y_den))
if d_peak > 0:
    y_den = y_den / d_peak * 0.9

snr_val = snr_db(y_clean[:min_len], y_den)
delta_snr = snr_val - snr_noisy
print(f"   SNR: {snr_val:.1f} dB (delta: +{delta_snr:.1f} dB)")

# ── 3. WAV 저장 ──
sf.write(os.path.join(SAVE_DIR, "3_denoised_lunet_v1.wav"), y_den, SR, subtype='PCM_16')
print(f"   WAV 저장: 3_denoised_lunet_v1.wav")

# ── 4. 개별 이미지 (4분할) ──
print("3. 이미지 생성 중...")
color = "#3fb950"
t_axis = np.linspace(0, DURATION, CHUNK_LEN)

fig, ax_s = plt.subplots(2, 2, figsize=(18, 9))
fig.patch.set_facecolor('#0d1117')

for r in range(2):
    for c in range(2):
        ax_s[r][c].set_facecolor('#161b22')
        ax_s[r][c].tick_params(colors='#aaa', labelsize=8)
        for spine in ax_s[r][c].spines.values():
            spine.set_color('#444')

# 상단 좌: 원본 vs 디노이즈 파형
ax_s[0][0].plot(t_axis, y_clean, color='#58a6ff', linewidth=0.5, alpha=0.6, label='원본')
ax_s[0][0].plot(t_axis[:len(y_den)], y_den, color=color, linewidth=0.5, alpha=0.8, label='디노이즈')
ax_s[0][0].set_title('원본 vs 디노이즈 파형', fontsize=13, fontweight='bold', color='white')
ax_s[0][0].set_ylim(-1, 1)
ax_s[0][0].legend(fontsize=10, loc='upper right', facecolor='#161b22', edgecolor='#444', labelcolor='white')

# 상단 우: 노이즈 vs 디노이즈 파형
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

fig.suptitle(f'LU-Net v1 — LSTM + MSE  |  SNR 개선: +{delta_snr:.1f} dB',
             fontsize=17, fontweight='bold', color='white', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(r"G:\stetho_ai\denoise_LUNet_v1.png", dpi=300,
            bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved: denoise_LUNet_v1.png")
plt.close()
