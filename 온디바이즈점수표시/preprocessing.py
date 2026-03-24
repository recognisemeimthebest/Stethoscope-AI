import os
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm

# 1. 경로 설정
adult_base = r"G:\stetho_ai\HeartSound For Adult\classification-of-heart-sound-recordings\classification-of-heart-sound-recordings"
ped_base = r"G:\stetho_ai\HeartSound For Pediatric"
ped_csv = os.path.join(ped_base, "labeled_dataset.csv")


def get_segments(file_path, target_sr=4000, duration=2, hop=1):
    if not os.path.exists(file_path):
        return []
    try:
        # 4000Hz 리샘플링 로드
        y, sr = librosa.load(file_path, sr=target_sr)
        samples_per_win = int(target_sr * duration)
        hop_samples = int(target_sr * hop)

        if len(y) < samples_per_win:
            return []

        # 2초 윈도우, 1초 슬라이딩 추출
        return [y[i: i + samples_per_win] for i in range(0, len(y) - samples_per_win + 1, hop_samples)]
    except:
        return []


# 데이터 저장소
storage = {'adult_norm': [], 'adult_abnorm': [], 'ped_norm': [], 'ped_abnorm': []}

# 2. 성인 데이터 처리 (각 폴더별 순회)
print("--- 성인 데이터 처리 중 (a~f 폴더 순회) ---")
for letter in ['a', 'b', 'c', 'd', 'e', 'f']:
    folder_name = f"training-{letter}"
    folder_path = os.path.join(adult_base, folder_name)
    csv_path = os.path.join(folder_path, "REFERENCE.csv")

    if not os.path.exists(csv_path):
        print(f"경고: {folder_name}에 REFERENCE.csv가 없습니다. 건너뜁니다.")
        continue

    print(f"[{folder_name}] 읽는 중...")
    df_adult = pd.read_csv(csv_path, header=None, names=['File', 'Label'])

    for _, row in tqdm(df_adult.iterrows(), total=len(df_adult), desc=folder_name):
        f_id = str(row['File'])
        f_path = os.path.join(folder_path, f"{f_id}.wav")

        segs = get_segments(f_path)
        if segs:
            # PhysioNet 라벨: -1(정상), 1(비정상)
            if int(row['Label']) == -1:
                storage['adult_norm'].extend(segs)
            else:
                storage['adult_abnorm'].extend(segs)

# 3. 소아 데이터 처리 (Absent/Present 대소문자 대응)
print("\n--- 소아 데이터 처리 중 ---")
if os.path.exists(ped_csv):
    df_ped = pd.read_csv(ped_csv)
    for _, row in tqdm(df_ped.iterrows(), total=len(df_ped), desc="Pediatric"):
        f_name = os.path.basename(row['File_Path'])
        f_path = os.path.join(ped_base, "cleaned_data", f_name)

        segs = get_segments(f_path)
        if segs:
            # 라벨 공백 제거 및 소문자 변환 후 비교
            label_clean = str(row['Murmur']).strip().lower()
            if label_clean == 'absent':
                storage['ped_norm'].extend(segs)
            elif label_clean == 'present':
                storage['ped_abnorm'].extend(segs)
else:
    print(f"오류: 소아용 CSV 파일을 찾을 수 없습니다: {ped_csv}")

# 4. 1:1:1:1 밸런싱
counts = {k: len(v) for k, v in storage.items()}
print(f"\n수집 결과 요약: {counts}")

# 모든 그룹의 데이터가 최소 1개 이상인지 확인
if any(v == 0 for v in counts.values()):
    print("경고: 일부 그룹에 데이터가 없습니다. 수집된 그룹만으로 밸런싱을 진행합니다.")
    valid_storage = {k: v for k, v in storage.items() if len(v) > 0}
else:
    valid_storage = storage

if not valid_storage:
    print("최종 수집된 데이터가 없습니다.")
else:
    # 가장 적은 그룹의 개수 추출
    min_samples = min(len(v) for v in valid_storage.values())
    print(f"그룹별 타겟 샘플 수: {min_samples} (총 {min_samples * len(valid_storage)}개)")

    final_x, final_y = [], []
    for k, v in valid_storage.items():
        # 랜덤하게 개수 맞추기
        idx = np.random.choice(len(v), min_samples, replace=False)
        for i in idx:
            final_x.append(v[i])
            # abnormal(1), normal(0)
            final_y.append(1 if 'abnorm' in k else 0)

    final_x = np.array(final_x)
    final_y = np.array(final_y)

    # 전체 데이터 셔플링 (학습 효율을 위해 중요)
    indices = np.arange(len(final_x))
    np.random.shuffle(indices)
    final_x = final_x[indices]
    final_y = final_y[indices]

    # 저장
    np.save('x_data.npy', final_x)
    np.save('y_data.npy', final_y)
    print(f"\n최종 데이터셋 저장 완료!")
    print(f"X shape: {final_x.shape} (2초 길이의 4000Hz 데이터)")
    print(f"Y shape: {final_y.shape}")