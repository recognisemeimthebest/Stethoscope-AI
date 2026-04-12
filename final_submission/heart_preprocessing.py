import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

# 1. 경로 설정
ADULT_BASE = r"G:\stetho_ai\_misc\datasets\classification-of-heart-sound-recordings\classification-of-heart-sound-recordings"
PED_BASE = r"G:\stetho_ai\_misc\datasets\pediatric"
UNKNOWN_BASE = r"G:\stetho_ai\_misc\datasets\ESC-50-master\ESC-50-master\audio"
DEVICE_HEART_BASE = r"G:\stetho_ai\_misc\heart\analysis\my_heart_data\heart\good"
DEVICE_NOISE_BASE = r"G:\stetho_ai\_misc\heart\analysis\my_heart_data\noise"
BLE_HEART_BASE = r"G:\stetho_ai\heartwav"
CACHE_DIR = r"G:\stetho_ai\Heart_binary_classification"

# 파라미터 (4000Hz 설정)
SR = 4000
WINDOW_DURATION = 5  # 윈도우 크기: 5초
STRIDE_DURATION = 2  # 이동 간격: 2초 (3초씩 중첩)
SAMPLES_PER_WINDOW = SR * WINDOW_DURATION
STRIDE_SAMPLES = SR * STRIDE_DURATION
N_MELS = 64
HOP_LENGTH = 256

# 고정 ref 값 (dB 변환 시 일관된 기준 사용)
FIXED_REF = 1.0

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


def get_unknown_metadata():
    """ESC-50 환경음 데이터를 Unknown(label=2) 클래스로 로드"""
    unknown_files = [
        os.path.join(UNKNOWN_BASE, f)
        for f in os.listdir(UNKNOWN_BASE) if f.endswith('.wav')
    ]
    unknown_df = pd.DataFrame({
        'File_Name': [os.path.splitext(os.path.basename(f))[0] for f in unknown_files],
        'Label': 2,
        'File_Path': unknown_files,
        'Source': 'ESC-50_Unknown'
    })
    print(f"Unknown(ESC-50) 데이터: {len(unknown_df)}개 파일")
    return unknown_df


def get_device_metadata():
    """ICS43434 기기로 녹음한 실제 심음(Normal), 노이즈(Unknown) 데이터"""
    heart_files = [
        os.path.join(DEVICE_HEART_BASE, f)
        for f in os.listdir(DEVICE_HEART_BASE) if f.endswith('.wav')
    ]
    noise_files = [
        os.path.join(DEVICE_NOISE_BASE, f)
        for f in os.listdir(DEVICE_NOISE_BASE) if f.endswith('.wav')
    ]
    heart_df = pd.DataFrame({
        'File_Name': [os.path.splitext(os.path.basename(f))[0] for f in heart_files],
        'Label': 0,  # Normal
        'File_Path': heart_files,
        'Source': 'Device_Heart'
    })
    noise_df = pd.DataFrame({
        'File_Name': [os.path.splitext(os.path.basename(f))[0] for f in noise_files],
        'Label': 2,  # Unknown
        'File_Path': noise_files,
        'Source': 'Device_Noise'
    })
    print(f"기기 심음(Normal) 데이터: {len(heart_df)}개 파일")
    print(f"기기 노이즈(Unknown) 데이터: {len(noise_df)}개 파일")
    return heart_df, noise_df


def get_ble_metadata():
    """BLE 전송된 실제 청진 녹음 (Normal) — 디노이즈 파일 제외"""
    ble_files = [
        os.path.join(BLE_HEART_BASE, f)
        for f in os.listdir(BLE_HEART_BASE)
        if f.endswith('.wav') and 'denoised' not in f
    ]
    ble_df = pd.DataFrame({
        'File_Name': [os.path.splitext(os.path.basename(f))[0] for f in ble_files],
        'Label': 0,  # Normal
        'File_Path': ble_files,
        'Source': 'BLE_Heart'
    })
    print(f"BLE 심음(Normal) 데이터: {len(ble_df)}개 파일")
    return ble_df


try:
    print("메타데이터 수집 및 균형 샘플링 시작...")
    adult_df, ped_df = get_metadata()
    unknown_df = get_unknown_metadata()
    device_heart_df, device_noise_df = get_device_metadata()
    ble_heart_df = get_ble_metadata()

    # 5개 그룹 분리 (성인/아동 x 정상/비정상 + Unknown)
    a_norm = adult_df[adult_df['Label'] == 0]
    a_abnorm = adult_df[adult_df['Label'] == 1]
    p_norm = ped_df[ped_df['Label'] == 0]
    p_abnorm = ped_df[ped_df['Label'] == 1]

    # 5그룹 균형 맞추기 (원본 파일 개수 기준, 기기 데이터 제외)
    min_count = min(len(a_norm), len(a_abnorm), len(p_norm), len(p_abnorm), len(unknown_df))
    print(f"\n파일 기준 최소 그룹 개수({min_count})에 맞춰 샘플링합니다.")

    balanced_df = pd.concat([
        a_norm.sample(min_count, random_state=42),
        a_abnorm.sample(min_count, random_state=42),
        p_norm.sample(min_count, random_state=42),
        p_abnorm.sample(min_count, random_state=42),
        unknown_df.sample(min_count, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    # 기기 데이터 + BLE 녹음 전량 추가
    device_df = pd.concat([device_heart_df, device_noise_df, ble_heart_df], ignore_index=True)
    print(f"기기 데이터 {len(device_df)}개 파일 전량 추가 (기기심음+노이즈+BLE녹음)")

    all_df = pd.concat([balanced_df, device_df], ignore_index=True)
    all_df = all_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 슬라이딩 윈도우 기반 특징 추출
    features, labels, file_ids = [], [], []
    print(f"\n윈도윙 추출 시작 (Window: 5s, Stride: 2s, SR: {SR}Hz)...")
    print(f"ref=FIXED_REF({FIXED_REF}) 사용 (기존 ref=np.max에서 변경)")

    for file_idx, (_, row) in enumerate(tqdm(all_df.iterrows(), total=len(all_df))):
        try:
            # 오디오 로드
            y, _ = librosa.load(row['File_Path'], sr=SR)
            # 피크 정규화 — 녹음 환경별 음량 차이 제거
            peak = np.max(np.abs(y))
            if peak > 0:
                y = y / peak

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
                features.append(librosa.power_to_db(mel_spec, ref=FIXED_REF))
                labels.append(row['Label'])
                file_ids.append(file_idx)

        except Exception as e:
            continue

    # 데이터 저장
    X = np.expand_dims(np.array(features), axis=1)  # (Total_Segments, 1, 64, 79)
    y = np.array(labels)
    fids = np.array(file_ids)

    # 클래스별 세그먼트 수 출력
    unique, counts = np.unique(y, return_counts=True)
    label_names = {0: 'Normal', 1: 'Abnormal', 2: 'Unknown'}
    print(f"\n클래스별 세그먼트 수:")
    for u, c in zip(unique, counts):
        print(f"  {label_names.get(int(u), u)}: {c}개")

    np.save(os.path.join(CACHE_DIR, 'x_data.npy'), X)
    np.save(os.path.join(CACHE_DIR, 'y_data.npy'), y)
    np.save(os.path.join(CACHE_DIR, 'file_ids.npy'), fids)

    n_files = len(np.unique(fids))
    print(f"\n성공적으로 완료되었습니다!")
    print(f"생성된 총 세그먼트 수: {len(X)} (출처 파일 {n_files}개)")
    print(f"최종 데이터 텐서 모양: {X.shape}")
    print(f"파일 인덱스 저장: file_ids.npy (파일 단위 분할용)")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\n오류 발생: {e}")