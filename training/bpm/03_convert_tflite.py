"""
ONNX -> TFLite INT8 변환
========================
학습된 beat detector를 ESP32용 TFLite INT8로 변환.
+ C 헤더 배열로 내보내기 (ESP32 flash에 임베딩용).

사용법:
  C:/Users/dwd00/anaconda3/envs/stetho_ai/python.exe 03_convert_tflite.py
"""

import os
import numpy as np

MODEL_DIR = "G:/stetho_ai/BPM_ondevice/models"
DATA_DIR = "G:/stetho_ai/BPM_ondevice/data"
ONNX_PATH = os.path.join(MODEL_DIR, "beat_cnn.onnx")
TFLITE_FP32_PATH = os.path.join(MODEL_DIR, "beat_cnn_fp32.tflite")
TFLITE_INT8_PATH = os.path.join(MODEL_DIR, "beat_cnn_int8.tflite")
HEADER_PATH = os.path.join(MODEL_DIR, "beat_cnn_model.h")

WINDOW_SAMPLES = 400


def onnx_to_tf_savedmodel():
    """ONNX -> TF SavedModel via onnx2tf or onnx-tf."""
    import onnx
    from onnx_tf.backend import prepare

    onnx_model = onnx.load(ONNX_PATH)
    tf_rep = prepare(onnx_model)
    savedmodel_dir = os.path.join(MODEL_DIR, "tf_savedmodel")
    tf_rep.export_graph(savedmodel_dir)
    print(f"SavedModel saved: {savedmodel_dir}")
    return savedmodel_dir


def pytorch_to_tflite_direct():
    """
    PyTorch -> TFLite 직접 변환 (ONNX 우회).
    PyTorch 모델을 numpy weight로 추출 -> TF Keras로 재구성 -> TFLite 변환.
    """
    import torch
    import tensorflow as tf

    # PyTorch 모델 로드
    import sys
    sys.path.insert(0, "G:/stetho_ai/BPM_ondevice")

    # 02_train_beat_detector.py에서 모델 클래스 임포트
    from importlib.machinery import SourceFileLoader
    mod = SourceFileLoader("trainer", "G:/stetho_ai/BPM_ondevice/02_train_beat_detector.py").load_module()
    TinyBeatCNN = mod.TinyBeatCNN

    pt_model = TinyBeatCNN()
    pt_model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "beat_cnn_best.pth"),
                    map_location="cpu", weights_only=True)
    )
    pt_model.eval()

    # 동등한 Keras 모델 구성
    # PyTorch padding=N 은 symmetric -> ZeroPadding1D + valid로 정확히 재현
    inp = tf.keras.Input(shape=(WINDOW_SAMPLES, 1))  # (batch, time, channels)
    x = inp

    # Block 1: Conv1D(1->16, k=7, s=2, padding=3)
    x = tf.keras.layers.ZeroPadding1D(padding=3)(x)
    x = tf.keras.layers.Conv1D(16, 7, strides=2, padding='valid', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Block 2: Conv1D(16->32, k=5, s=2, padding=2)
    x = tf.keras.layers.ZeroPadding1D(padding=2)(x)
    x = tf.keras.layers.Conv1D(32, 5, strides=2, padding='valid', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Block 3: Conv1D(32->32, k=3, s=2, padding=1)
    x = tf.keras.layers.ZeroPadding1D(padding=1)(x)
    x = tf.keras.layers.Conv1D(32, 3, strides=2, padding='valid', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # GAP -> Dense(1)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(1)(x)

    keras_model = tf.keras.Model(inp, x)

    # PyTorch weight -> Keras weight 복사
    sd = pt_model.state_dict()

    def set_conv_weights(keras_layer, pt_prefix):
        """PyTorch Conv1d (out, in, k) -> Keras Conv1D (k, in, out)"""
        w = sd[f"{pt_prefix}.weight"].numpy()  # (out, in, k)
        w = np.transpose(w, (2, 1, 0))         # (k, in, out)
        keras_layer.set_weights([w])

    def set_bn_weights(keras_layer, pt_prefix):
        """PyTorch BN -> Keras BN: [gamma, beta, running_mean, running_var]"""
        gamma = sd[f"{pt_prefix}.weight"].numpy()
        beta = sd[f"{pt_prefix}.bias"].numpy()
        mean = sd[f"{pt_prefix}.running_mean"].numpy()
        var = sd[f"{pt_prefix}.running_var"].numpy()
        keras_layer.set_weights([gamma, beta, mean, var])

    def set_dense_weights(keras_layer, pt_prefix):
        w = sd[f"{pt_prefix}.weight"].numpy().T  # (out, in) -> (in, out)
        b = sd[f"{pt_prefix}.bias"].numpy()
        keras_layer.set_weights([w, b])

    # 레이어 매핑 (ZeroPadding 레이어는 weight 없으므로 자동 스킵됨)
    keras_layers = [l for l in keras_model.layers if len(l.get_weights()) > 0]
    # Conv1, BN1, Conv2, BN2, Conv3, BN3, Dense
    pt_prefixes = [
        ("features.0", set_conv_weights),   # Conv1d
        ("features.1", set_bn_weights),     # BN1
        ("features.3", set_conv_weights),   # Conv1d
        ("features.4", set_bn_weights),     # BN2
        ("features.6", set_conv_weights),   # Conv1d
        ("features.7", set_bn_weights),     # BN3
        ("classifier.2", set_dense_weights), # Linear
    ]

    for layer, (prefix, setter) in zip(keras_layers, pt_prefixes):
        setter(layer, prefix)
        print(f"  Copied: {prefix} -> {layer.name}")

    # 검증: PyTorch vs Keras 출력 비교
    test_input = np.random.randn(1, WINDOW_SAMPLES).astype(np.float32)
    with torch.no_grad():
        pt_out = pt_model(torch.from_numpy(test_input).unsqueeze(1)).item()
    keras_out = keras_model.predict(test_input.reshape(1, WINDOW_SAMPLES, 1), verbose=0)[0, 0]
    print(f"\n  Verification - PT: {pt_out:.6f}, Keras: {keras_out:.6f}, diff: {abs(pt_out-keras_out):.6f}")

    return keras_model


def convert_to_tflite(keras_model):
    """Keras -> TFLite FP32 + INT8."""
    import tensorflow as tf

    # FP32
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_fp32 = converter.convert()
    with open(TFLITE_FP32_PATH, 'wb') as f:
        f.write(tflite_fp32)
    print(f"\nFP32 TFLite: {len(tflite_fp32)/1024:.1f} KB -> {TFLITE_FP32_PATH}")

    # INT8 양자화 — representative dataset 필요
    X_cal = np.load(os.path.join(DATA_DIR, "X_windows.npy"), mmap_mode='r')
    # 1000개 랜덤 샘플로 calibration
    cal_idx = np.random.choice(len(X_cal), size=min(1000, len(X_cal)), replace=False)

    def representative_dataset():
        for i in cal_idx:
            sample = X_cal[i].reshape(1, WINDOW_SAMPLES, 1).astype(np.float32)
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_int8 = converter.convert()

    with open(TFLITE_INT8_PATH, 'wb') as f:
        f.write(tflite_int8)
    print(f"INT8 TFLite: {len(tflite_int8)/1024:.1f} KB -> {TFLITE_INT8_PATH}")

    return tflite_int8


def generate_c_header(tflite_bytes):
    """TFLite 바이너리 -> C 헤더 배열 (ESP32 임베딩용)."""
    with open(HEADER_PATH, 'w') as f:
        f.write("// Auto-generated: TinyBeatCNN INT8 TFLite model\n")
        f.write(f"// Size: {len(tflite_bytes)} bytes ({len(tflite_bytes)/1024:.1f} KB)\n")
        f.write("// Input: int8[1][400][1] (200ms @ 2kHz)\n")
        f.write("// Output: int8[1][1] (beat logit)\n\n")
        f.write("#pragma once\n\n")
        f.write(f"const unsigned int beat_cnn_model_len = {len(tflite_bytes)};\n")
        f.write("alignas(16) const unsigned char beat_cnn_model[] = {\n")

        for i in range(0, len(tflite_bytes), 12):
            chunk = tflite_bytes[i:i+12]
            hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
            f.write(f"    {hex_str},\n")

        f.write("};\n")

    print(f"\nC header: {HEADER_PATH}")
    print(f"  -> ESP32 firmware에 #include 후 사용")


def main():
    print("=" * 50)
    print("ONNX -> TFLite INT8 변환")
    print("=" * 50)

    # PyTorch -> Keras -> TFLite 직접 변환
    print("\n[1/3] PyTorch -> Keras 변환...")
    keras_model = pytorch_to_tflite_direct()

    print("\n[2/3] Keras -> TFLite 변환...")
    tflite_int8 = convert_to_tflite(keras_model)

    print("\n[3/3] C 헤더 생성...")
    generate_c_header(tflite_int8)

    print(f"\n{'='*50}")
    print("완료! ESP32에 필요한 파일:")
    print(f"  1. {TFLITE_INT8_PATH} (TFLite 바이너리)")
    print(f"  2. {HEADER_PATH} (C 헤더 — 펌웨어 임베딩용)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
