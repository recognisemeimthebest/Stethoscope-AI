"""
TUNet-v2 .h5 → .tflite 변환 스크립트
Windows에서 실행 후 .tflite 파일을 RPi로 복사
"""
import os
import tensorflow as tf
import tensorflow.keras.backend as K

# 커스텀 손실 함수 (모델 로드에 필요)
def si_snr_loss(y_true, y_pred):
    y_true = K.squeeze(y_true, axis=-1)
    y_pred = K.squeeze(y_pred, axis=-1)
    y_true = y_true - K.mean(y_true, axis=-1, keepdims=True)
    y_pred = y_pred - K.mean(y_pred, axis=-1, keepdims=True)
    dot = K.sum(y_pred * y_true, axis=-1, keepdims=True)
    s_target = dot * y_true / (K.sum(y_true ** 2, axis=-1, keepdims=True) + 1e-8)
    e_noise = y_pred - s_target
    si_snr = 10 * K.log(
        K.sum(s_target ** 2, axis=-1) / (K.sum(e_noise ** 2, axis=-1) + 1e-8)
    ) / K.log(10.0)
    return -K.mean(si_snr)

def combined_loss(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))
    si_snr = si_snr_loss(y_true, y_pred)
    return 0.9 * mse + 0.1 * (si_snr / 20.0)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
H5_PATH = os.path.join(BASE_DIR, "tunet_v2_best.h5")
TFLITE_PATH = os.path.join(BASE_DIR, "tunet_v2_best.tflite")

print("모델 로드 중...")
model = tf.keras.models.load_model(
    H5_PATH,
    custom_objects={"combined_loss": combined_loss, "si_snr_loss": si_snr_loss},
)
model.summary()

print("\nTFLite 변환 중...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(TFLITE_PATH) / 1024 / 1024
print(f"\n변환 완료: {TFLITE_PATH}")
print(f"파일 크기: {size_mb:.1f} MB")
