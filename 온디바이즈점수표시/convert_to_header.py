import os

# 파일 경로 설정
input_file = 'heart_sound_quant.tflite'
output_file = 'model_data.h'

if not os.path.exists(input_file):
    print(f"오류: {input_file} 파일이 없습니다. TFLite 변환을 먼저 완료해주세요.")
else:
    with open(input_file, 'rb') as f:
        data = f.read()

    with open(output_file, 'w') as f:
        # 헤더 가드 및 배열 정의
        f.write('#ifndef HEART_SOUND_MODEL_DATA_H\n')
        f.write('#define HEART_SOUND_MODEL_DATA_H\n\n')

        # 배열 이름은 ESP32 코드에서 호출할 이름입니다.
        f.write(f'const unsigned char heart_sound_model_data[] = {{\n')

        # 데이터를 16진수 배열로 변환 (12개마다 줄바꿈)
        for i, byte in enumerate(data):
            f.write(f' 0x{byte:02x},')
            if (i + 1) % 12 == 0:
                f.write('\n')

        f.write('\n};\n')
        f.write(f'const int heart_sound_model_data_len = {len(data)};\n\n')
        f.write('#endif\n')

    print(f"성공: {output_file} 파일이 생성되었습니다! (크기: {len(data)} bytes)")