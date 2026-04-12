"""
정상 vs 비정상 심음 비교 시각화 (일반인 친화 버전)
상단: Shannon Envelope (심장 박동 에너지) / 하단: Mel-Spectrogram
"""
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import butter, filtfilt
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

DATASET = r"G:\stetho_ai\_misc\datasets\classification-of-heart-sound-recordings\classification-of-heart-sound-recordings\training-a"
NORMAL_FILE = f"{DATASET}\\a0007.wav"    # label -1 = 정상
ABNORMAL_FILE = f"{DATASET}\\a0001.wav"  # label  1 = 비정상

SR = 4000
DURATION = 5.0
N_MELS = 128


def shannon_envelope(y, sr, frame_len=0.02, hop_len=0.01):
    """Shannon energy envelope: -x^2 * log(x^2)"""
    frame_samples = int(frame_len * sr)
    hop_samples = int(hop_len * sr)

    # 정규화
    y_norm = y / (np.max(np.abs(y)) + 1e-10)

    envelope = []
    for start in range(0, len(y_norm) - frame_samples, hop_samples):
        frame = y_norm[start:start + frame_samples]
        x2 = frame ** 2 + 1e-12  # log(0) 방지
        se = -np.mean(x2 * np.log(x2))
        envelope.append(se)

    envelope = np.array(envelope)
    # 0~1 정규화
    envelope = (envelope - envelope.min()) / (envelope.max() - envelope.min() + 1e-10)

    # 스무딩 (로우패스 필터)
    nyq = (sr / hop_samples) / 2
    cutoff = min(8.0, nyq * 0.8)  # 8Hz 이하만 통과
    b, a = butter(2, cutoff / nyq, btype='low')
    envelope = filtfilt(b, a, envelope)
    envelope = np.clip(envelope, 0, None)
    envelope = envelope / (envelope.max() + 1e-10)

    t = np.linspace(0, len(y) / sr, len(envelope))
    return t, envelope


fig, axes = plt.subplots(2, 2, figsize=(18, 12),
                         gridspec_kw={'height_ratios': [1, 1.8]})
fig.patch.set_facecolor('#1a1a2e')

configs = [
    (0, NORMAL_FILE,   "정상 심음 (Normal)",   "#2ecc71",
     "S1·S2 박동이 규칙적으로 반복\n→ 건강한 심장의 '덥-딱' 리듬",
     "밝은 세로줄이 일정한 간격 =\n규칙적인 심장 박동"),
    (1, ABNORMAL_FILE, "비정상 심음 (Abnormal)", "#e74c3c",
     "박동 사이에 비정상 에너지 존재\n→ 심잡음(murmur) 가능성",
     "세로줄 사이가 흐릿하게 차있음 =\n심잡음(murmur) 존재"),
]

for col, wav_path, title, color, env_desc, mel_desc in configs:
    y, sr = librosa.load(wav_path, sr=SR, duration=DURATION)

    # ── 상단: Shannon Envelope ──
    ax_env = axes[0][col]
    ax_env.set_facecolor('#16213e')
    t_env, env = shannon_envelope(y, sr)

    ax_env.fill_between(t_env, env, alpha=0.3, color=color)
    ax_env.plot(t_env, env, color=color, linewidth=1.5)
    ax_env.set_title(title, fontsize=17, fontweight='bold', color=color, pad=12)
    ax_env.set_xlabel('시간 (초)', fontsize=11, color='white')
    ax_env.set_ylabel('심장 박동 에너지', fontsize=11, color='white')
    ax_env.set_ylim(0.1, 1.15)
    ax_env.tick_params(colors='white')
    for spine in ax_env.spines.values():
        spine.set_color('#444')

    # 설명 박스
    ax_env.text(0.98, 0.95, env_desc, transform=ax_env.transAxes,
                fontsize=11, color='white', ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.25))

    # 비정상: 피크 사이 비정상 에너지에 동그라미 표시
    if col == 1:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(env, height=0.35, distance=int(0.25 * len(env) / DURATION))
        # 피크 사이 구간에서 중간 에너지(0.15~0.45) 범프 찾기
        for i in range(len(peaks) - 1):
            seg_start = peaks[i]
            seg_end = peaks[i + 1]
            seg = env[seg_start:seg_end]
            # 피크 사이 구간의 중간 지점에서 에너지가 올라온 부분 찾기
            mid_peaks, props = find_peaks(seg, height=(0.15, 0.50), prominence=0.03)
            for mp in mid_peaks:
                idx = seg_start + mp
                from matplotlib.patches import Ellipse
                # 시간축/에너지축 스케일이 다르므로 Ellipse 사용
                ell = Ellipse((t_env[idx], env[idx]), width=0.25, height=0.15,
                              fill=False, edgecolor='#f1c40f',
                              linewidth=2.5, linestyle='--')
                ax_env.add_patch(ell)
        # 동그라미 범례 텍스트
        ax_env.text(0.02, 0.95, '--- 비정상 에너지',
                    transform=ax_env.transAxes, fontsize=11,
                    color='#f1c40f', ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#f1c40f', alpha=0.15))

    # ── 하단: Mel-Spectrogram ──
    ax_mel = axes[1][col]
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=sr // 2)
    S_dB = librosa.power_to_db(S, ref=np.max)

    img = librosa.display.specshow(
        S_dB, sr=sr, x_axis='time', y_axis='mel',
        ax=ax_mel, cmap='magma', fmax=sr // 2
    )
    ax_mel.set_xlabel('시간 (초)', fontsize=11, color='white')
    ax_mel.set_ylabel('주파수 (높은음↑ 낮은음↓)', fontsize=11, color='white')
    ax_mel.tick_params(colors='white')
    for spine in ax_mel.spines.values():
        spine.set_color('#444')
    cb = fig.colorbar(img, ax=ax_mel, format='%+2.0f dB', pad=0.02)
    cb.ax.tick_params(colors='white')
    cb.set_label('소리 세기 (dB)', color='white', fontsize=10)
    # 설명 박스
    ax_mel.text(0.98, 0.95, mel_desc, transform=ax_mel.transAxes,
                fontsize=11, color='white', ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.25))

# ── 상단 제목 ──
fig.text(0.5, 0.97,
         '심음 AI 분석 시각화  —  심장 소리를 눈으로 보다',
         fontsize=18, fontweight='bold', color='white', ha='center')
fig.text(0.5, 0.935,
         '위: Shannon Envelope (심장 박동 에너지 곡선)  |  아래: Mel-Spectrogram (AI가 보는 데이터)',
         fontsize=12, color='#aaa', ha='center')

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(r"G:\stetho_ai\mel_spectrogram_comparison.png", dpi=300,
            bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved: G:\\stetho_ai\\mel_spectrogram_comparison.png")
plt.close()
