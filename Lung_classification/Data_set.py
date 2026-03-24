import os
import random
import shutil
from collections import defaultdict

# 1. 경로 설정
src_dir = r"G:\stetho_ai\LUNG SOUND\Child_Merged"
base_out_dir = r"G:\stetho_ai\LUNG SOUND\Dataset_Split"

# Train, Val, Test 비율 설정 (8:1:1)
train_ratio = 0.8
val_ratio = 0.1
# test_ratio는 나머지 0.1

# 2. Train, Val, Test 폴더 생성
splits = ['train', 'val', 'test']
for split in splits:
    split_dir = os.path.join(base_out_dir, split)
    os.makedirs(split_dir, exist_ok=True)

# src_dir 안의 7개 클래스 폴더 목록 가져오기
classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

for cls in classes:
    cls_dir = os.path.join(src_dir, cls)
    files = [f for f in os.listdir(cls_dir) if f.endswith('.wav')]

    # 각 split 폴더 안에 클래스 폴더 생성
    for split in splits:
        os.makedirs(os.path.join(base_out_dir, split, cls), exist_ok=True)

    # 3. 데이터 누수 방지: 원본 파일명(환자/녹음 ID) 기준으로 그룹화
    records = defaultdict(list)
    for f in files:
        # 파일명이 "원본이름_순서_클래스.wav" 형식이므로, 뒤에서 두 번 언더바를 기준으로 자름
        # 예: "101_1b1_Al_sc_Meditron_0_Normal.wav" -> 원본이름 추출
        parts = f.rsplit('_', 2)
        base_name = parts[0]
        records[base_name].append(f)

    # 원본 파일명 리스트를 랜덤하게 섞기
    record_keys = list(records.keys())
    random.shuffle(record_keys)

    # 4. 8:1:1 비율로 분할 인덱스 계산
    total_records = len(record_keys)
    train_idx = int(total_records * train_ratio)
    val_idx = train_idx + int(total_records * val_ratio)

    train_keys = record_keys[:train_idx]
    val_keys = record_keys[train_idx:val_idx]
    test_keys = record_keys[val_idx:]


    # 5. 파일 복사 함수
    def copy_files(keys, split_name):
        count = 0
        for key in keys:
            for f in records[key]:
                src_path = os.path.join(cls_dir, f)
                dst_path = os.path.join(base_out_dir, split_name, cls, f)
                shutil.copy2(src_path, dst_path)
                count += 1
        return count


    # 실제 파일 복사 및 결과 출력
    train_cnt = copy_files(train_keys, 'train')
    val_cnt = copy_files(val_keys, 'val')
    test_cnt = copy_files(test_keys, 'test')

    print(f"[{cls}] 분할 완료 - Train: {train_cnt}개, Val: {val_cnt}개, Test: {test_cnt}개 (그룹 수: {total_records})")

print(f"\n모든 클래스의 8:1:1 분할이 완료되었습니다! 확인 경로: {base_out_dir}")