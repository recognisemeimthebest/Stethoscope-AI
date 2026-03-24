import numpy as np
import tensorflow as tf
import librosa

# 모델 로드
interpreter = tf.lite.Interpreter(model_path="heart_sound_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# [수정] 차원을 (1, 13, 32, 1)로 맞추는 전처리 함수
def extract_for_stetho(file_path):
    y, sr = librosa.load(file_path, sr=4000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # 시간축(32프레임) 맞추기
    if mfcc.shape[1] > 32:
        mfcc = mfcc[:, :32]
    else:
        mfcc = np.pad(mfcc, ((0, 0), (0, 32 - mfcc.shape[1])), mode='constant')

    # [중요] 모양 변경: (1, 13, 32, 1)
    mfcc = mfcc.reshape(1, 13, 32, 1)

    # int8 양자화 적용 (Scaling)
    scale, zero_point = input_details[0]['quantization']
    if scale != 0:
        mfcc = (mfcc / scale + zero_point)

    return mfcc.astype(np.int8)


# 실행
input_data = extract_for_stetho("test.wav")  # 실제 파일명 확인!
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(f"결과값: {output_data}")