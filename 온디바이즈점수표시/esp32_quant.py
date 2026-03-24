import tensorflow as tf
import numpy as np

# 1. 학습된 모델 로드
model = tf.keras.models.load_model('heart_sound_model_v1.h5')

# 2. 양자화를 위한 대표 데이터셋(Representative Dataset) 준비
# 모델이 실제 데이터의 범위를 알게 하여 오차를 줄입니다.
x_mfcc = np.load('x_mfcc.npy')
def representative_data_gen():
    # 전체 데이터 중 100개 정도만 무작위로 추출하여 샘플링
    for i in range(100):
        data = np.expand_dims(x_mfcc[i], axis=0).astype(np.float32)
        yield [data]

# 3. TFLite 변환 설정
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# 정수 연산만 사용하도록 강제 (ESP32 가속을 위해 필수)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# 4. 변환 실행
tflite_quant_model = converter.convert()

# 5. .tflite 파일로 저장
with open('heart_sound_quant.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print("양자화 모델 변환 완료: heart_sound_quant.tflite")