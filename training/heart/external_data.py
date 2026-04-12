import os
import librosa
import numpy as np
import torch
from tqdm import tqdm

# 1. 외부 데이터 파일 경로 설정 (파일 확장자까지 정확히 기입)
RAW_FILE = r"G:\stetho_ai\raw_heart_sound.wav"
MATCHED_FILE = r"G:\stetho_ai\matched_heart_sound.wav"

# 전처리 파라미터 (이전과 동일)
SR = 4000
WINDOW_DURATION = 5
SAMPLES_PER_WINDOW = SR * WINDOW_DURATION
HOP_LENGTH = 256
N_MELS = 64


# 2. 개별 파일 추론 및 상세 결과 출력 함수
def single_file_inference(file_path, model, xgb, name):
    if not os.path.exists(file_path):
        print(f"[{name}] 파일을 찾을 수 없습니다: {file_path}")
        return

    print(f"\n--- [{name}] 분석 시작 ---")
    model.eval()

    # 특징 추출 (슬라이딩 윈도우 적용)
    mels = process_new_audio(file_path)  # 이전 코드의 함수 그대로 사용

    if mels is None or len(mels) == 0:
        print(f"[{name}] 오디오 처리 실패")
        return

    # CNN 특징 추출
    mels_tensor = torch.FloatTensor(mels).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        _, features = model(mels_tensor)
        features_np = features.cpu().numpy()

    # XGBoost 예측 (0: 정상, 1: 비정상)
    seg_preds = xgb.predict(features_np)

    # 결과 요약
    normal_count = np.sum(seg_preds == 0)
    abnormal_count = np.sum(seg_preds == 1)

    # 최종 판정 (과반수 투표)
    final_label = "비정상 (Abnormal)" if np.mean(seg_preds) >= 0.5 else "정상 (Normal)"

    print(f"총 세그먼트 수: {len(seg_preds)}")
    print(f"구간별 결과: [정상: {normal_count} / 비정상: {abnormal_count}]")
    print(f"최종 판정: {final_label}")
    print(f"세부 판정값: {seg_preds}")


# 3. 실행
print("\n--- Phase 4: 직접 녹음한 심음 외부 검증 ---")
single_file_inference(RAW_FILE, model, xgb, "RAW_HEART_SOUND")
single_file_inference(MATCHED_FILE, model, xgb, "MATCHED_HEART_SOUND")