"""
LU-Net v1 (LSTM + MSE) 디노이즈 성능 시각화
원본 심음 → 노이즈 추가 → 디노이즈 결과 비교
Mel-Spectrogram + 파형 + SNR 수치
"""
import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import butter, filtfilt
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 경로 ──
DATASET = r"G:\stetho_ai\_misc\datasets\classification-of-heart-sound-recordings\classification-of-heart-sound-recordings\training-a"
CLEAN_FILE = os.path.join(DATASET, "a0007.wav")  # 정상 심음
MODEL_DIR = r"G:\stetho_ai\LSTM_first\Dataset_10000"

# v1 파라미터 (denoising_test.py 기준)
V1_SR = 2000
V1_DURATION = 5
V1_N_FFT = 512
V1_HOP = 128
V1_N_MELS = 64
V1_FMAX = 1000

import tensorflow as tf


def snr(clean, noisy):
    """Signal-to-Noise Ratio (dB)"""
    noise = noisy - clean
    return 10 * np.log10(np.sum(clean**2) / (np.sum(noise**2) + 1e-10))


# ── 1. 원본 로드 ──
print("1. 원본 심음 로드...")
y_clean, _ = librosa.load(CLEAN_FILE, sr=V1_SR, duration=V1_DURATION)
if len(y_clean) < V1_SR * V1_DURATION:
    y_clean = np.pad(y_clean, (0, V1_SR * V1_DURATION - len(y_clean)))
else:
    y_clean = y_clean[:V1_SR * V1_DURATION]

# 정규화
peak = np.max(np.abs(y_clean))
if peak > 0:
    y_clean = y_clean / peak * 0.9

# ── 2. 노이즈 추가 (백색 잡음 + 저주파 험) ──
print("2. 노이즈 합성...")
np.random.seed(42)
white_noise = np.random.randn(len(y_clean)) * 0.08
# 저주파 험 (50Hz 전원 노이즈 시뮬레이션)
t = np.linspace(0, V1_DURATION, len(y_clean))
hum = 0.05 * np.sin(2 * np.pi * 50 * t)
y_noisy = y_clean + white_noise + hum
y_noisy = np.clip(y_noisy, -1, 1).astype(np.float32)

# ── 3. v1 모델로 디노이즈 ──
print("3. LU-Net v1 모델 로드 및 추론...")
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'pro_denoising_crnn.h5'))
Y_min, Y_max = np.load(os.path.join(MODEL_DIR, 'scaling_params.npy'))

# 노이즈 오디오 → mel spectrogram
mel_noisy = librosa.feature.melspectrogram(
    y=y_noisy, sr=V1_SR, n_fft=V1_N_FFT, hop_length=V1_HOP,
    n_mels=V1_N_MELS, fmax=V1_FMAX
)
mel_noisy_db = librosa.power_to_db(mel_noisy, ref=np.max).T

# 스케일링
X_min, X_max = np.min(mel_noisy_db), np.max(mel_noisy_db)
mel_scaled = (mel_noisy_db - X_min) / (X_max - X_min + 1e-8)
mel_input = np.expand_dims(mel_scaled, axis=0)

# 예측
pred_scaled = model.predict(mel_input, verbose=0)[0]
pred_db = pred_scaled * (Y_max - Y_min + 1e-8) + Y_min

# 역변환
pred_power = librosa.db_to_power(pred_db.T)
y_denoised = librosa.feature.inverse.mel_to_audio(
    pred_power, sr=V1_SR, n_fft=V1_N_FFT, hop_length=V1_HOP
)

# 길이 맞추기
min_len = min(len(y_clean), len(y_noisy), len(y_denoised))
y_clean = y_clean[:min_len]
y_noisy = y_noisy[:min_len]
y_denoised = y_denoised[:min_len]

# 디노이즈 정규화
d_peak = np.max(np.abs(y_denoised))
if d_peak > 0:
    y_denoised = y_denoised / d_peak * 0.9

# SNR 계산
snr_noisy = snr(y_clean, y_noisy)
snr_denoised = snr(y_clean, y_denoised)

print(f"   SNR (노이즈): {snr_noisy:.1f} dB")
print(f"   SNR (디노이즈): {snr_denoised:.1f} dB")
print(f"   SNR 개선: +{snr_denoised - snr_noisy:.1f} dB")

# ── 4. WAV 저장 ──
print("4. WAV 파일 저장...")
SAVE_DIR = r"G:\stetho_ai\denoise_demo"
os.makedirs(SAVE_DIR, exist_ok=True)
sf.write(os.path.join(SAVE_DIR, "1_clean.wav"), y_clean, V1_SR, subtype='PCM_16')
sf.write(os.path.join(SAVE_DIR, "2_noisy.wav"), y_noisy, V1_SR, subtype='PCM_16')
sf.write(os.path.join(SAVE_DIR, "3_denoised_v1.wav"), y_denoised, V1_SR, subtype='PCM_16')
print(f"   저장 완료: {SAVE_DIR}")

# ── 5. 시각화 ──
print("5. 시각화 생성 중...")

fig, axes = plt.subplots(3, 3, figsize=(22, 16),
                         gridspec_kw={'width_ratios': [1, 1, 0.8]})
fig.patch.set_facecolor('#0d1117')

signals = [
    (y_clean, "원본 심음 (Clean)", "#58a6ff"),
    (y_noisy, f"노이즈 추가 (SNR: {snr_noisy:.1f} dB)", "#f85149"),
    (y_denoised, f"LU-Net v1 디노이즈 (SNR: {snr_denoised:.1f} dB)", "#f0883e"),
]

t_axis = np.linspace(0, min_len / V1_SR, min_len)

for row, (y, title, color) in enumerate(signals):
    # ── 1열: 파형 ──
    ax_wave = axes[row][0]
    ax_wave.set_facecolor('#161b22')
    ax_wave.plot(t_axis, y, color=color, linewidth=0.4, alpha=0.9)
    ax_wave.set_title(title if row == 0 else '', fontsize=14, fontweight='bold',
                       color=color, pad=10)
    ax_wave.set_ylabel(title.split('(')[0].strip(), fontsize=12, fontweight='bold',
                        color=color, labelpad=10)
    ax_wave.set_xlabel('시간 (초)', fontsize=10, color='white')
    ax_wave.set_ylim(-1, 1)
    ax_wave.tick_params(colors='#aaa')
    for spine in ax_wave.spines.values():
        spine.set_color('#444')
    if row == 0:
        ax_wave.set_title('파형 (Waveform)', fontsize=14, fontweight='bold',
                           color='white', pad=10)

    # ── 2열: Mel-Spectrogram ──
    ax_mel = axes[row][1]
    ax_mel.set_facecolor('#161b22')
    S = librosa.feature.melspectrogram(y=y, sr=V1_SR, n_fft=V1_N_FFT,
                                        hop_length=V1_HOP, n_mels=V1_N_MELS,
                                        fmax=V1_FMAX)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=V1_SR, hop_length=V1_HOP,
                                    x_axis='time', y_axis='mel',
                                    ax=ax_mel, cmap='cividis', fmax=V1_FMAX)
    ax_mel.set_xlabel('시간 (초)', fontsize=10, color='white')
    ax_mel.set_ylabel('Mel 주파수', fontsize=10, color='white')
    ax_mel.tick_params(colors='#aaa')
    for spine in ax_mel.spines.values():
        spine.set_color('#444')
    cb = fig.colorbar(img, ax=ax_mel, pad=0.02, shrink=0.85)
    cb.ax.tick_params(colors='#aaa')
    cb.set_label('dB', color='#aaa', fontsize=9)
    if row == 0:
        ax_mel.set_title('Mel-Spectrogram', fontsize=14, fontweight='bold',
                          color='white', pad=10)

# ── 3열: SNR 비교 바 차트 + 설명 ──
# 상단: SNR 바 차트
ax_snr = axes[0][2]
ax_snr.set_facecolor('#161b22')
bars = ax_snr.barh(['노이즈', '디노이즈'], [snr_noisy, snr_denoised],
                    color=['#f85149', '#f0883e'], height=0.5)
ax_snr.set_xlabel('SNR (dB) — 높을수록 깨끗', fontsize=11, color='white')
ax_snr.set_title('SNR 비교', fontsize=14, fontweight='bold', color='white', pad=10)
ax_snr.tick_params(colors='#aaa')
for spine in ax_snr.spines.values():
    spine.set_color('#444')
for bar, val in zip(bars, [snr_noisy, snr_denoised]):
    ax_snr.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f} dB', va='center', fontsize=12, color='white', fontweight='bold')
# 개선량 표시
ax_snr.text(0.5, -0.15, f'SNR 개선: +{snr_denoised - snr_noisy:.1f} dB',
            transform=ax_snr.transAxes, fontsize=13, color='#58a6ff',
            ha='center', fontweight='bold')

# 중단: 모델 구조 설명
ax_info = axes[1][2]
ax_info.set_facecolor('#0d1117')
ax_info.axis('off')
info_text = (
    "LU-Net v1 구조\n"
    "━━━━━━━━━━━━━━━\n"
    "Conv1D (64) x 2\n"
    "  + BatchNorm\n"
    "  + Dropout 0.2\n"
    "━━━━━━━━━━━━━━━\n"
    "Bi-LSTM (128)\n"
    "  + Dropout 0.3\n"
    "Bi-LSTM (64)\n"
    "  + Dropout 0.3\n"
    "━━━━━━━━━━━━━━━\n"
    "Dense (64, sigmoid)\n"
    "━━━━━━━━━━━━━━━\n"
    "Loss: MSE\n"
    "SR: 2000 Hz\n"
    "학습: 10,000 샘플"
)
ax_info.text(0.5, 0.5, info_text, transform=ax_info.transAxes,
             fontsize=11, color='white', ha='center', va='center',
             linespacing=1.4, fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#161b22',
                       edgecolor='#f0883e', linewidth=2))

# 하단: 한계점
ax_limit = axes[2][2]
ax_limit.set_facecolor('#0d1117')
ax_limit.axis('off')
limit_text = (
    "v1 한계점\n\n"
    "- 2000Hz로 고주파 손실\n"
    "- Griffin-Lim 역변환\n"
    "  → 위상 정보 손실\n"
    "- MSE Loss\n"
    "  → 과도한 스무딩\n"
    "- 스펙트로그램 기반\n"
    "  → 시간 해상도 제한"
)
ax_limit.text(0.5, 0.5, limit_text, transform=ax_limit.transAxes,
              fontsize=11, color='#f85149', ha='center', va='center',
              linespacing=1.4,
              bbox=dict(boxstyle='round,pad=0.8', facecolor='#161b22',
                        edgecolor='#f85149', linewidth=2))

fig.suptitle('LU-Net v1 (LSTM + MSE) 디노이즈 성능',
             fontsize=21, fontweight='bold', color='white', y=0.98)
fig.text(0.5, 0.945,
         '정상 심음에 백색 잡음 + 전원 험을 추가한 뒤, v1 모델로 노이즈를 제거한 결과입니다',
         fontsize=12, color='#8b949e', ha='center')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(r"G:\stetho_ai\denoise_v1_performance.png", dpi=300,
            bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved: denoise_v1_performance.png")
plt.close()
