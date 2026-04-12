"""
ICBHI 사이클 단위 학습 데이터 생성기
- ICBHI 어노테이션의 호흡 사이클(1-3초)을 그대로 사용
- 사이클 길이가 다르므로 고정 길이(FIXED_SEC)로 패딩/트림
- ESC-50 → Unknown
"""

import os
import random
import librosa
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
from multiprocessing import Pool

warnings.filterwarnings('ignore')

# ── 파라미터 ─────────────────────────────────────────────────
SR = 8000
FIXED_SEC = 2.0                      # 사이클을 2초로 통일 (패딩/트림)
FIXED_SAMPLES = int(SR * FIXED_SEC)  # 16000
MIN_ENERGY = 0.001

N_MELS = 64
FMIN = 50
FMAX = 3500
HOP_LENGTH = 40     # 5ms @ 8kHz
N_FFT = 256          # 32ms @ 8kHz

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── 경로 ─────────────────────────────────────────────────────
ICBHI_DIR = r"G:\stetho_ai\_misc\lung\datasets\ICBHI_respiratory_db\Respiratory_Sound_Database\Respiratory_Sound_Database\audio_and_txt_files"
ESC50_AUDIO = r"G:\stetho_ai\_misc\datasets\ESC-50-master\ESC-50-master\audio"
ESC50_META = r"G:\stetho_ai\_misc\datasets\ESC-50-master\ESC-50-master\meta\esc50.csv"
CACHE_DIR = r"G:\stetho_ai\Lung_classification\cache"

EXCLUDE_CATEGORIES = {'breathing', 'coughing', 'snoring', 'sneezing'}

CLASSES = ['Crackle', 'Normal', 'Unknown', 'Wheeze']
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}


def parse_annotation(txt_path):
    annotations = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                start, end = float(parts[0]), float(parts[1])
                crackle, wheeze = int(parts[2]), int(parts[3])
                annotations.append((start, end, crackle, wheeze))
    return annotations


def cycle_label(crackle, wheeze):
    if crackle == 1:
        return 'Crackle'
    elif wheeze == 1:
        return 'Wheeze'
    else:
        return 'Normal'


def make_mel(y_seg):
    """오디오 세그먼트 → PCEN mel-spectrogram"""
    S = librosa.feature.melspectrogram(
        y=y_seg, sr=SR, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX
    )
    return librosa.pcen(S * (2**31), sr=SR, hop_length=HOP_LENGTH, bias=10).astype(np.float32)


def process_icbhi_wav(args):
    """ICBHI WAV + 어노테이션 → (mel, label) 리스트"""
    wav_path, txt_path = args
    try:
        annotations = parse_annotation(txt_path)
        y, _ = librosa.load(wav_path, sr=SR)
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak

        results = []
        for start, end, crackle, wheeze in annotations:
            s_idx = int(start * SR)
            e_idx = int(end * SR)
            segment = y[s_idx:e_idx]

            # 무음 스킵
            if np.sqrt(np.mean(segment ** 2)) < MIN_ENERGY:
                continue

            # 고정 길이로 맞추기
            segment = librosa.util.fix_length(segment, size=FIXED_SAMPLES)

            label = cycle_label(crackle, wheeze)
            mel = make_mel(segment)
            results.append((mel, CLASS_TO_IDX[label]))

        return results
    except Exception:
        return []


def process_esc_wav(wav_path):
    """ESC-50 WAV → 2초 단위 mel 리스트"""
    try:
        y, _ = librosa.load(wav_path, sr=SR)
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak

        mels = []
        # 5초 → 2초씩 (2개 + 나머지)
        for start in range(0, len(y) - FIXED_SAMPLES + 1, FIXED_SAMPLES):
            seg = y[start:start + FIXED_SAMPLES]
            if np.sqrt(np.mean(seg ** 2)) >= MIN_ENERGY:
                mels.append(make_mel(seg))
        return mels
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs(CACHE_DIR, exist_ok=True)
    NUM_WORKERS = 8

    print("=" * 60)
    print(f"사이클 단위 데이터 생성 (고정 {FIXED_SEC}초, 8kHz PCEN)")
    print("=" * 60)

    # ICBHI 페어
    wav_files = sorted([f for f in os.listdir(ICBHI_DIR) if f.endswith('.wav')])
    icbhi_pairs = []
    for wf in wav_files:
        txt = wf.replace('.wav', '.txt')
        if os.path.exists(os.path.join(ICBHI_DIR, txt)):
            icbhi_pairs.append((os.path.join(ICBHI_DIR, wf), os.path.join(ICBHI_DIR, txt)))
    print(f"ICBHI: {len(icbhi_pairs)}개")

    # ESC-50
    esc_meta = pd.read_csv(ESC50_META)
    excluded = set(esc_meta[esc_meta['category'].isin(EXCLUDE_CATEGORIES)]['filename'].tolist())
    esc_wavs = [os.path.join(ESC50_AUDIO, f) for f in os.listdir(ESC50_AUDIO) if f.endswith('.wav') and f not in excluded]
    print(f"ESC-50 Unknown: {len(esc_wavs)}개")

    # 8:1:1
    random.shuffle(icbhi_pairs)
    random.shuffle(esc_wavs)

    def split_811(lst):
        n = len(lst)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        return lst[:n_train], lst[n_train:n_train+n_val], lst[n_train+n_val:]

    icbhi_train, icbhi_val, icbhi_test = split_811(icbhi_pairs)
    esc_train, esc_val, esc_test = split_811(esc_wavs)

    print(f"분할: ICBHI {len(icbhi_train)}/{len(icbhi_val)}/{len(icbhi_test)}, ESC-50 {len(esc_train)}/{len(esc_val)}/{len(esc_test)}")

    # 추출
    print(f"\n{'='*60}")
    print(f"mel-spectrogram 추출 (워커 {NUM_WORKERS}개)")
    print(f"{'='*60}")

    splits = {
        'train': (icbhi_train, esc_train),
        'val':   (icbhi_val,   esc_val),
        'test':  (icbhi_test,  esc_test),
    }

    for split_name, (icbhi_split, esc_split) in splits.items():
        print(f"\n[{split_name.upper()}]")
        all_mels, all_labels = [], []

        with Pool(NUM_WORKERS) as pool:
            results = pool.map(process_icbhi_wav, icbhi_split)

        cls_counts = defaultdict(int)
        for cycle_list in results:
            for mel, label_idx in cycle_list:
                all_mels.append(mel)
                all_labels.append(label_idx)
                cls_counts[CLASSES[label_idx]] += 1

        for cls in ['Crackle', 'Normal', 'Wheeze']:
            print(f"  {cls}: {cls_counts[cls]} 사이클")

        with Pool(NUM_WORKERS) as pool:
            esc_results = pool.map(process_esc_wav, esc_split)

        unk_count = 0
        unk_idx = CLASS_TO_IDX['Unknown']
        for mel_list in esc_results:
            for mel in mel_list:
                all_mels.append(mel)
                all_labels.append(unk_idx)
                unk_count += 1
        print(f"  Unknown: {unk_count} 세그먼트")

        X = np.array(all_mels)
        y = np.array(all_labels)
        np.save(os.path.join(CACHE_DIR, f'frame_{split_name}_data.npy'), X)
        np.save(os.path.join(CACHE_DIR, f'frame_{split_name}_labels.npy'), y)

        total = len(all_labels)
        print(f"  저장: frame_{split_name}_data.npy {X.shape} ({X.nbytes/1024**2:.0f}MB)")
        print(f"  총: {total}개")

    print(f"\n완료!")
