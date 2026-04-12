import os
import json
import librosa
import numpy as np

# 사용자 데이터 경로 설정 (Windows 경로 포맷에 맞게 수정)
DATA_PATHS = {
    "Adult_HeartSound": r"G:\stetho_ai\HeartSound For Adult\classification-of-heart-sound-recordings\classification-of-heart-sound-recordings",
    "Pediatric_HeartSound": r"G:\stetho_ai\HeartSound For Pediatric\cleaned_data",
    "ESC50_Noise": r"G:\stetho_ai\ESC-50-master\ESC-50-master\audio",
    "Custom_Data": r"G:\stetho_ai\LSTM_first"
}


def analyze_audio_files(base_path):
    info = {
        "total_files": 0,
        "sample_rates": set(),
        "total_duration_sec": 0.0,
        "durations": []
    }

    for root, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                try:
                    # sr=None으로 설정하여 원본 샘플링 레이트 그대로 읽기
                    y, sr = librosa.load(file_path, sr=None)
                    duration = librosa.get_duration(y=y, sr=sr)

                    info["total_files"] += 1
                    info["sample_rates"].add(sr)
                    info["total_duration_sec"] += duration
                    info["durations"].append(duration)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    # 통계 요약 (JSON 직렬화를 위해 set과 numpy 타입을 변환)
    if info["durations"]:
        info["avg_duration"] = round(float(np.mean(info["durations"])), 2)
        info["max_duration"] = round(float(np.max(info["durations"])), 2)
        info["min_duration"] = round(float(np.min(info["durations"])), 2)
        info["total_duration_min"] = round(info["total_duration_sec"] / 60, 2)

    info["sample_rates"] = list(info["sample_rates"])
    del info["durations"]  # 파일 목록이 너무 길어지므로 개별 길이는 삭제
    del info["total_duration_sec"]

    return info


def main():
    print("오디오 데이터 분석을 시작합니다. 데이터가 많아 몇 분 정도 걸릴 수 있습니다...")
    dataset_summary = {}

    for category, path in DATA_PATHS.items():
        print(f"[{category}] 분석 중... ({path})")
        if os.path.exists(path):
            dataset_summary[category] = analyze_audio_files(path)
            print(f" -> 완료: {dataset_summary[category]['total_files']}개의 파일 찾음.")
        else:
            print(f" -> 경고: 경로를 찾을 수 없습니다.")
            dataset_summary[category] = {"error": "Path not found"}

    # JSON 파일로 저장
    output_filename = "dataset_info.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(dataset_summary, f, indent=4, ensure_ascii=False)

    print(f"\n분석이 완료되었습니다! '{output_filename}' 파일이 생성되었습니다.")


if __name__ == "__main__":
    main()