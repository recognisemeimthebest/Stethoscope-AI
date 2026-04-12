import os
import random
import librosa
import numpy as np
import matplotlib.pyplot as plt
import warnings

# librosa 경고 메시지 숨기기 (짧은 오디오 처리 시 뜨는 경고 방지)
warnings.filterwarnings('ignore')

# 1. 경로 설정
src_base = r"G:\stetho_ai\_misc\lung\classification\LUNG_SOUND\Dataset_Split"
out_base = r"G:\stetho_ai\_misc\lung\classification\LUNG_SOUND\Spectrograms_Balanced"
unknown_audio_dir = r"G:\stetho_ai\_misc\datasets\ESC-50-master\ESC-50-master\audio"

# 2. 클래스 병합 맵 (7개 -> 5개)
class_map = {
    'Normal': 'Normal',
    'Fine Crackle': 'Crackle',
    'Coarse Crackle': 'Crackle',
    'Wheeze': 'Wheeze',
    'Rhonchi': 'Wheeze',
    'Wheeze+Crackle': 'Complex',
    'Stridor': 'Stridor'
}


# 3. 오디오 -> 멜-스펙트로그램 변환 함수 (핵심 전처리)
def save_melspectrogram(y, sr, out_path):
    # 길이를 2초(2 * sr)로 고정 (짧으면 0으로 채우고, 길면 자름)
    y_fixed = librosa.util.fix_length(y, size=2 * sr)

    # 멜-스펙트로그램 추출 (간호학적 주요 주파수 대역에 집중)
    S = librosa.feature.melspectrogram(y=y_fixed, sr=sr, n_mels=128, fmin=50, fmax=2000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # 이미지로 저장 (여백과 축 제거하여 순수 데이터만 저장)
    plt.figure(figsize=(3, 3))
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    plt.imshow(S_dB, aspect='auto', origin='lower', cmap='magma')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


# 4. 데이터 증폭 (Augmentation) 함수 - 피치 조절 및 노이즈 추가
def augment_audio(y, sr):
    aug_list = []
    # 1. 노이즈 추가
    noise = np.random.randn(len(y))
    y_noise = y + 0.005 * noise
    aug_list.append(y_noise)
    # 2. 피치 낮추기 (-2 반음)
    y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
    aug_list.append(y_pitch_down)
    # 3. 피치 높이기 (+2 반음)
    y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    aug_list.append(y_pitch_up)
    return aug_list


# 5. 메인 실행 루프
splits = ['train', 'val', 'test']

for split in splits:
    print(f"\n[{split.upper()} 데이터 변환 시작]")
    split_src = os.path.join(src_base, split)
    split_out = os.path.join(out_base, split)

    # 출력 폴더 생성
    for new_cls in set(class_map.values()):
        os.makedirs(os.path.join(split_out, new_cls), exist_ok=True)

    for old_cls in os.listdir(split_src):
        if old_cls not in class_map:
            continue

        new_cls = class_map[old_cls]
        cls_dir = os.path.join(split_src, old_cls)
        files = [f for f in os.listdir(cls_dir) if f.endswith('.wav')]

        # [데이터 밸런싱 로직 (Train 세트에서만 적용)]
        if split == 'train':
            # Normal은 너무 많으니 무작위로 1,000개만 사용
            if old_cls == 'Normal' and len(files) > 1000:
                files = random.sample(files, 1000)

        count = 0
        for f in files:
            wav_path = os.path.join(cls_dir, f)
            base_name = f.replace('.wav', '')

            try:
                y, sr = librosa.load(wav_path, sr=16000)

                # 원본 저장
                save_melspectrogram(y, sr, os.path.join(split_out, new_cls, f"{base_name}.png"))
                count += 1

                # [소수 클래스 데이터 증폭 (Train 세트에서만 적용)]
                if split == 'train' and old_cls in ['Wheeze+Crackle', 'Stridor', 'Rhonchi']:
                    augmented_audios = augment_audio(y, sr)
                    for i, aug_y in enumerate(augmented_audios):
                        aug_name = f"{base_name}_aug{i}.png"
                        save_melspectrogram(aug_y, sr, os.path.join(split_out, new_cls, aug_name))
                        count += 1
            except Exception as e:
                print(f"에러 발생 ({f}): {e}")

        print(f"  - {old_cls} -> {new_cls} 병합 및 이미지 변환 완료 (총 {count}장 생성)")

# 6. Unknown 클래스 생성 (ESC-50 환경음 → 스펙트로그램)
print("\n[Unknown 클래스 생성 (ESC-50 환경음)]")
unknown_files = [f for f in os.listdir(unknown_audio_dir) if f.endswith('.wav')]
random.shuffle(unknown_files)

# train:val:test = 8:1:1 비율로 분배
n_total = len(unknown_files)
n_train = int(n_total * 0.8)
n_val = int(n_total * 0.1)

split_map = {
    'train': unknown_files[:n_train],
    'val': unknown_files[n_train:n_train + n_val],
    'test': unknown_files[n_train + n_val:]
}

for split, files in split_map.items():
    unknown_out = os.path.join(out_base, split, 'Unknown')
    os.makedirs(unknown_out, exist_ok=True)
    count = 0
    for f in files:
        wav_path = os.path.join(unknown_audio_dir, f)
        base_name = f.replace('.wav', '')
        try:
            y, sr = librosa.load(wav_path, sr=16000)
            save_melspectrogram(y, sr, os.path.join(unknown_out, f"{base_name}.png"))
            count += 1
        except Exception as e:
            print(f"  에러 ({f}): {e}")
    print(f"  - {split}/Unknown: {count}장 생성")

print(f"\n모든 작업이 완료되었습니다! 최종 이미지 폴더: {out_base}")