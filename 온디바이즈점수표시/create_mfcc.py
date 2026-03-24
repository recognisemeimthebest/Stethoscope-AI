import numpy as np
import librosa
from tqdm import tqdm

# 1. 데이터 로드
print("데이터 로드 중...")
x_data = np.load('x_data.npy')
y_data = np.load('y_data.npy')


# 2. MFCC 추출 함수 수정
def extract_mfcc(wav_data, sr=4000, n_mfcc=13):
    # n_fft와 hop_length를 조절하여 이미지의 가로 크기를 결정 (약 2초 데이터 -> 13x32 크기)
    # fmin=20을 추가하여 심음의 저역대 특징을 더 잘 잡도록 함
    mfcc = librosa.feature.mfcc(y=wav_data, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=256, fmin=20)

    # [핵심 수정] 샘플 단위 정규화 (Standardization)
    # 각 샘플별로 평균을 빼고 표준편차로 나눠서 특징을 극대화함
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-7)
    return mfcc


# 3. 모든 데이터 MFCC 변환
print("MFCC 변환 시작 (ESP32 최적화 및 특징 극대화)...")
x_mfcc = []
for i in tqdm(range(len(x_data))):
    mfcc_feat = extract_mfcc(x_data[i])
    x_mfcc.append(mfcc_feat)

x_mfcc = np.array(x_mfcc)

# 4. 차원 확장 (CNN 입력 형태: [개수, 높이, 너비, 채널])
x_mfcc_final = x_mfcc[..., np.newaxis]

# 5. 저장
np.save('x_mfcc.npy', x_mfcc_final)
np.save('y_label.npy', y_data)

print(f"\n변환 완료!")
print(f"MFCC 데이터 형태: {x_mfcc_final.shape}")
# 데이터 분포 확인 (학습 전 디버깅용)
print(f"데이터 평균: {np.mean(x_mfcc_final):.4f}, 표준편차: {np.std(x_mfcc_final):.4f}")