import os
import json
import librosa
import soundfile as sf

# 1. 경로 설정 (필요에 따라 BioCAS2022 등으로 변경해서 재실행 가능)
json_dir = r"G:\stetho_ai\LUNG SOUND\SPRSound-main\Detection\train_detection_json"
wav_dir = r"G:\stetho_ai\LUNG SOUND\SPRSound-main\Detection\train_detection_wav"
output_base_dir = r"G:\stetho_ai\LUNG SOUND\Child_Merged"

# 2. 파일 처리 루프
for json_name in os.listdir(json_dir):
    if not json_name.endswith('.json'):
        continue

    base_name = json_name.replace('.json', '')
    wav_name = base_name + '.wav'
    wav_path = os.path.join(wav_dir, wav_name)
    json_path = os.path.join(json_dir, json_name)

    # 짝꿍이 되는 wav 파일이 없으면 패스
    if not os.path.exists(wav_path):
        continue

    # JSON 파일 열기
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 3. 데이터 품질 필터링 ("Poor Quality"는 버림)
    if data.get('record_annotation') == 'Poor Quality':
        continue

    events = data.get('event_annotation', [])
    if not events:  # 추출할 이벤트가 없으면 패스
        continue

    # 4. 오디오 파일 로드 (AI 학습의 표준인 16,000Hz로 통일하여 로드)
    try:
        y, sr = librosa.load(wav_path, sr=16000)
    except Exception as e:
        print(f"오디오 로드 에러 {wav_path}: {e}")
        continue

    # 5. 시간 정보에 맞춰 오디오 자르기(Slicing)
    for idx, event in enumerate(events):
        start_ms = int(event['start'])
        end_ms = int(event['end'])
        event_type = event['type']  # 예: "Normal", "Wheeze", "Crackle"

        # 해당 타입의 폴더가 없으면 생성
        target_dir = os.path.join(output_base_dir, event_type)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # 밀리초(ms)를 샘플(Sample) 인덱스로 변환
        start_sample = int((start_ms / 1000.0) * sr)
        end_sample = int((end_ms / 1000.0) * sr)

        # 배열 슬라이싱 (실제 오디오 자르기)
        y_slice = y[start_sample:end_sample]

        # 너무 짧은 소리(예: 0.5초 미만)는 노이즈일 확률이 높으므로 제외
        if len(y_slice) < int(0.5 * sr):
            continue

        # 잘라낸 오디오 저장 (예: 101_0_Normal.wav)
        out_filename = f"{base_name}_{idx}_{event_type}.wav"
        out_filepath = os.path.join(target_dir, out_filename)

        # 파일이 이미 존재하지 않을 때만 저장
        if not os.path.exists(out_filepath):
            sf.write(out_filepath, y_slice, sr)

print(f"추출 완료! 결과 폴더를 확인하세요: {output_base_dir}")