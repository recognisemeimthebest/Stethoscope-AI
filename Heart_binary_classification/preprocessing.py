import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

# 1. 경로 설정
ADULT_BASE = r"G:\stetho_ai\HeartSound For Adult\classification-of-heart-sound-recordings\classification-of-heart-sound-recordings"
PED_BASE = r"G:\stetho_ai\HeartSound For Pediatric"
CACHE_DIR = r"G:\stetho_ai\Heart_binary_classification"

# 파라미터 (4000Hz 설정)
SR = 4000
WINDOW_DURATION = 5  # 윈도우 크기: 5초
STRIDE_DURATION = 2  # 이동 간격: 2초 (3초씩 중첩)
SAMPLES_PER_WINDOW = SR * WINDOW_DURATION
STRIDE_SAMPLES = SR * STRIDE_DURATION
N_MELS = 64
HOP_LENGTH = 256

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


def get_metadata():
    adult_list = []
    folders = [f'training-{c}' for c in ['a', 'b', 'c', 'd', 'e', 'f']]

    print(f"성인 데이터 스캔 중: {ADULT_BASE}")
    for f in folders:
        folder_path = os.path.join(ADULT_BASE, f)
        csv_path = os.path.join(folder_path, 'REFERENCE.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, header=None, names=['File_Name', 'Label'])
            df['Label'] = df['Label'].map({-1: 0, 1: 1})
            df['File_Path'] = df['File_Name'].apply(lambda x: os.path.join(folder_path, x + '.wav'))
            df['Source'] = 'Adult'
            adult_list.append(df)

    adult_df = pd.concat(adult_list, ignore_index=True)
    ped_csv = os.path.join(PED_BASE, 'labeled_dataset.csv')
    ped_df = pd.read_csv(ped_csv)
    ped_df['Label'] = ped_df['Murmur'].map({'Absent': 0, 'Present': 1})
    ped_df['File_Path'] = ped_df['File_Name'].apply(lambda x: os.path.join(PED_BASE, 'cleaned_data', x))
    ped_df['Source'] = 'Pediatric'

    return adult_df, ped_df


try:
    print("메타데이터 수집 및 균형 샘플링 시작...")
    adult_df, ped_df = get_metadata()

    # 4개 그룹 분리 (성인/아동 x 정상/비정상)
    a_norm = adult_df[adult_df['Label'] == 0]
    a_abnorm = adult_df[adult_df['Label'] == 1]
    p_norm = ped_df[ped_df['Label'] == 0]
    p_abnorm = ped_df[ped_df['Label'] == 1]

    # 5:5:5:5 비율 맞추기 (원본 파일 개수 기준)
    min_count = min(len(a_norm), len(a_abnorm), len(p_norm), len(p_abnorm))
    print(f"\n파일 기준 최소 그룹 개수({min_count})에 맞춰 샘플링합니다.")

    balanced_df = pd.concat([
        a_norm.sample(min_count, random_state=42),
        a_abnorm.sample(min_count, random_state=42),
        p_norm.sample(min_count, random_state=42),
        p_abnorm.sample(min_count, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    # 슬라이딩 윈도우 기반 특징 추출
    features, labels = [], []
    print(f"\n윈도윙 추출 시작 (Window: 5s, Stride: 2s, SR: {SR}Hz)...")

    for _, row in tqdm(balanced_df.iterrows(), total=len(balanced_df)):
        try:
            # 오디오 로드
            y, _ = librosa.load(row['File_Path'], sr=SR)

            # 파일이 5초보다 짧으면 패딩 후 1개 세그먼트만 생성
            if len(y) < SAMPLES_PER_WINDOW:
                y_padded = np.pad(y, (0, SAMPLES_PER_WINDOW - len(y)))
                segments = [y_padded]
            else:
                # 슬라이딩 윈도우 적용
                segments = []
                for start in range(0, len(y) - SAMPLES_PER_WINDOW + 1, STRIDE_SAMPLES):
                    segments.append(y[start: start + SAMPLES_PER_WINDOW])

            # 각 세그먼트별 Mel-Spectrogram 생성
            for seg in segments:
                mel_spec = librosa.feature.melspectrogram(y=seg, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
                features.append(librosa.power_to_db(mel_spec, ref=np.max))
                labels.append(row['Label'])

        except Exception as e:
            continue

    # 데이터 저장
    X = np.expand_dims(np.array(features), axis=1)  # (Total_Segments, 1, 64, 79)
    y = np.array(labels)

    np.save(os.path.join(CACHE_DIR, 'x_data.npy'), X)
    np.save(os.path.join(CACHE_DIR, 'y_data.npy'), y)

    print(f"\n성공적으로 완료되었습니다!")
    print(f"생성된 총 세그먼트 수: {len(X)}")
    print(f"최종 데이터 텐서 모양: {X.shape}")

except Exception as e:
    print(f"\n오류 발생: {e}")