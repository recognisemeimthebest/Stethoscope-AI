"""
TensorFlow 네이티브 1D-CNN Beat Detector 학습
==============================================
PyTorch->Keras 변환 시 padding 차이 문제를 피하기 위해
직접 TF/Keras로 동일 아키텍처 학습 후 TFLite INT8 변환.

사용법:
  C:/Users/dwd00/anaconda3/envs/stetho_ai/python.exe 02b_train_tf.py
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ---- Settings ----
DATA_DIR = "G:/stetho_ai/BPM_ondevice/data"
OUTPUT_DIR = "G:/stetho_ai/BPM_ondevice/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SAMPLES = 400
BATCH_SIZE = 256
EPOCHS = 30
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_model():
    """
    Tiny 1D-CNN: Conv1D(16,7,s2) -> Conv1D(32,5,s2) -> Conv1D(32,3,s2) -> GAP -> Dense(1)
    causal padding (valid + manual ZeroPad) to match ESP32 inference.
    """
    inp = tf.keras.Input(shape=(WINDOW_SAMPLES, 1), name="audio_window")
    x = inp

    # Block 1: pad 3 each side -> Conv1D(16, 7, s2) -> BN -> ReLU
    x = tf.keras.layers.ZeroPadding1D(padding=3)(x)
    x = tf.keras.layers.Conv1D(16, 7, strides=2, padding='valid', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Block 2: pad 2 each side -> Conv1D(32, 5, s2) -> BN -> ReLU
    x = tf.keras.layers.ZeroPadding1D(padding=2)(x)
    x = tf.keras.layers.Conv1D(32, 5, strides=2, padding='valid', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Block 3: pad 1 each side -> Conv1D(32, 3, s2) -> BN -> ReLU
    x = tf.keras.layers.ZeroPadding1D(padding=1)(x)
    x = tf.keras.layers.Conv1D(32, 3, strides=2, padding='valid', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # GAP -> Dense(1)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', name="beat_prob")(x)

    model = tf.keras.Model(inp, x)
    return model


def main():
    # 1) Load data
    print("Loading data...")
    X = np.load(os.path.join(DATA_DIR, "X_windows.npy"), mmap_mode='r')
    Y = np.load(os.path.join(DATA_DIR, "Y_labels.npy"), mmap_mode='r')
    print(f"X: {X.shape}, Y: {Y.shape}, beat ratio: {np.mean(Y):.3f}")

    # 2) Train/Val split
    idx = np.arange(len(X))
    train_idx, val_idx = train_test_split(idx, test_size=0.15, random_state=SEED, stratify=Y)

    X_train = X[train_idx].copy().reshape(-1, WINDOW_SAMPLES, 1)
    Y_train = Y[train_idx].copy().astype(np.float32)
    X_val = X[val_idx].copy().reshape(-1, WINDOW_SAMPLES, 1)
    Y_val = Y[val_idx].copy().astype(np.float32)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # 3) Class weight
    n_beat = np.sum(Y_train == 1)
    n_nobeat = np.sum(Y_train == 0)
    class_weight = {0: 1.0, 1: n_nobeat / n_beat}
    print(f"Class weight: beat={class_weight[1]:.2f}")

    # 4) Build model
    model = build_model()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 5) Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, "beat_cnn_tf_best.h5"),
            monitor='val_accuracy', save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10, restore_best_weights=True
        ),
    ]

    # 6) Train
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # 7) Evaluate
    val_pred_prob = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    val_pred = (val_pred_prob.flatten() > 0.5).astype(int)

    print(f"\n{'='*50}")
    print("Classification Report:")
    print(classification_report(Y_val, val_pred, target_names=["no-beat", "beat"]))
    print("Confusion Matrix:")
    print(confusion_matrix(Y_val, val_pred))

    # 8) TFLite FP32
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = converter.convert()
    fp32_path = os.path.join(OUTPUT_DIR, "beat_cnn_fp32.tflite")
    with open(fp32_path, 'wb') as f:
        f.write(tflite_fp32)
    print(f"\nFP32 TFLite: {len(tflite_fp32)/1024:.1f} KB")

    # 9) TFLite INT8
    def representative_dataset():
        cal_idx = np.random.choice(len(X_train), size=1000, replace=False)
        for i in cal_idx:
            yield [X_train[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_int8 = converter.convert()

    int8_path = os.path.join(OUTPUT_DIR, "beat_cnn_int8.tflite")
    with open(int8_path, 'wb') as f:
        f.write(tflite_int8)
    print(f"INT8 TFLite: {len(tflite_int8)/1024:.1f} KB")

    # 10) C header
    header_path = os.path.join(OUTPUT_DIR, "beat_cnn_model.h")
    with open(header_path, 'w') as f:
        f.write("// Auto-generated: TinyBeatCNN INT8 TFLite model\n")
        f.write(f"// Size: {len(tflite_int8)} bytes ({len(tflite_int8)/1024:.1f} KB)\n")
        f.write("// Input: int8[1][400][1] (200ms @ 2kHz, resample to match device SR)\n")
        f.write("// Output: int8[1][1] -> sigmoid -> P(beat)\n\n")
        f.write("#pragma once\n\n")
        f.write(f"const unsigned int beat_cnn_model_len = {len(tflite_int8)};\n")
        f.write("alignas(16) const unsigned char beat_cnn_model[] = {\n")
        for i in range(0, len(tflite_int8), 12):
            chunk = tflite_int8[i:i+12]
            hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
            f.write(f"    {hex_str},\n")
        f.write("};\n")
    print(f"C header: {header_path}")

    # 11) INT8 accuracy verification
    interp = tf.lite.Interpreter(model_path=int8_path)
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    i_scale = inp_det['quantization_parameters']['scales'][0]
    i_zp = inp_det['quantization_parameters']['zero_points'][0]
    o_scale = out_det['quantization_parameters']['scales'][0]
    o_zp = out_det['quantization_parameters']['zero_points'][0]

    test_idx = np.random.choice(len(X_val), size=500, replace=False)
    int8_correct = 0
    for i in test_idx:
        sample = X_val[i:i+1].astype(np.float32)
        sample_q = np.clip(np.round(sample / i_scale + i_zp), -128, 127).astype(np.int8)
        interp.set_tensor(inp_det['index'], sample_q)
        interp.invoke()
        raw_out = interp.get_tensor(out_det['index'])[0, 0]
        prob = (raw_out - o_zp) * o_scale
        pred = 1 if prob > 0.5 else 0
        if pred == int(Y_val[i]):
            int8_correct += 1

    print(f"\nINT8 accuracy (500 val samples): {int8_correct/500:.2%}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
