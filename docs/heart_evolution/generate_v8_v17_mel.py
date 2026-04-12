"""
V8 vs V17 — 같은 심음의 Mel-Spectrogram 전처리 차이 비교
V8:  1000Hz, 64 mel, 64x64
V17: 4000Hz, 64 mel, 64x79, Peak Normalization
"""
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

DATASET = r"G:\stetho_ai\_misc\datasets\classification-of-heart-sound-recordings\classification-of-heart-sound-recordings\training-a"
NORMAL_FILE = os.path.join(DATASET, "a0007.wav")
ABNORMAL_FILE = os.path.join(DATASET, "a0001.wav")

# ── V8 전처리 파라미터 ──
V8_SR = 1000
V8_N_MELS = 64
V8_HOP = 64       # 64x64 되도록
V8_DURATION = 5

# ── V17 전처리 파라미터 ──
V17_SR = 4000
V17_N_MELS = 64
V17_HOP = 256
V17_DURATION = 5
V17_REF = 1.0


def make_mel_v8(wav_path):
    y, _ = librosa.load(wav_path, sr=V8_SR, duration=V8_DURATION)
    # V8: 정규화 없음
    S = librosa.feature.melspectrogram(y=y, sr=V8_SR, n_mels=V8_N_MELS,
                                       hop_length=V8_HOP, fmax=V8_SR // 2)
    return librosa.power_to_db(S, ref=np.max)


def make_mel_v17(wav_path):
    y, _ = librosa.load(wav_path, sr=V17_SR, duration=V17_DURATION)
    # V17: Peak Normalization
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    S = librosa.feature.melspectrogram(y=y, sr=V17_SR, n_mels=V17_N_MELS,
                                       hop_length=V17_HOP, fmax=V17_SR // 2)
    return librosa.power_to_db(S, ref=V17_REF)


fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.patch.set_facecolor('#0d1117')

samples = [
    (NORMAL_FILE, "정상 심음 (Normal)", 0),
    (ABNORMAL_FILE, "비정상 심음 (Abnormal)", 1),
]

for wav_path, label, row in samples:
    mel_v8 = make_mel_v8(wav_path)
    mel_v17 = make_mel_v17(wav_path)

    # ── V8 ──
    ax1 = axes[row][0]
    ax1.set_facecolor('#161b22')
    img1 = librosa.display.specshow(mel_v8, sr=V8_SR, hop_length=V8_HOP,
                                     x_axis='time', y_axis='mel',
                                     ax=ax1, cmap='cividis', fmax=V8_SR // 2)
    if row == 0:
        ax1.set_title('V8 전처리\n1000Hz  |  64x64  |  정규화 없음',
                       fontsize=14, fontweight='bold', color='#8b949e', pad=12)
    ax1.set_ylabel(label, fontsize=13, fontweight='bold',
                    color='#58a6ff' if row == 0 else '#f0883e', labelpad=10)
    ax1.set_xlabel('시간 (초)', fontsize=11, color='white')
    ax1.tick_params(colors='#aaa')
    for spine in ax1.spines.values():
        spine.set_color('#555')
    cb1 = fig.colorbar(img1, ax=ax1, pad=0.02, shrink=0.85)
    cb1.ax.tick_params(colors='#aaa')
    cb1.set_label('dB', color='#aaa', fontsize=9)

    # 해상도 표시
    ax1.text(0.02, 0.05, f'해상도: {mel_v8.shape[0]} x {mel_v8.shape[1]}',
             transform=ax1.transAxes, fontsize=10, color='#8b949e',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117', alpha=0.8))

    # ── V17 ──
    ax2 = axes[row][1]
    ax2.set_facecolor('#161b22')
    img2 = librosa.display.specshow(mel_v17, sr=V17_SR, hop_length=V17_HOP,
                                     x_axis='time', y_axis='mel',
                                     ax=ax2, cmap='cividis', fmax=V17_SR // 2)
    if row == 0:
        ax2.set_title('V17 전처리\n4000Hz  |  64x79  |  Peak Normalization',
                       fontsize=14, fontweight='bold', color='#58a6ff', pad=12)
    ax2.set_xlabel('시간 (초)', fontsize=11, color='white')
    ax2.tick_params(colors='#aaa')
    for spine in ax2.spines.values():
        spine.set_color('#555')
    cb2 = fig.colorbar(img2, ax=ax2, pad=0.02, shrink=0.85)
    cb2.ax.tick_params(colors='#aaa')
    cb2.set_label('dB', color='#aaa', fontsize=9)

    # 해상도 표시
    ax2.text(0.02, 0.05, f'해상도: {mel_v17.shape[0]} x {mel_v17.shape[1]}',
             transform=ax2.transAxes, fontsize=10, color='#58a6ff',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117', alpha=0.8))

fig.suptitle('같은 심음, 다른 전처리  —  V8 vs V17 Mel-Spectrogram 비교',
             fontsize=20, fontweight='bold', color='white', y=0.98)
fig.text(0.5, 0.94,
         'V17은 4배 높은 샘플레이트로 더 많은 주파수 정보를 보고, Peak Normalization으로 음량 편차를 제거합니다',
         fontsize=12, color='#8b949e', ha='center')

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(r"G:\stetho_ai\v8_v17_mel_comparison.png", dpi=300,
            bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved: v8_v17_mel_comparison.png")
plt.close()
