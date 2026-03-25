"""
================================================================
LU-Net 심음 노이즈 제거 - Step 2: 모델 학습 (2초 버전)
================================================================
5초 버전 대비 변경:
  - input_length: 20000 → 8000
  - DATA_DIR: LUNet_Dataset_2sec
  - 에포크당 시간 대폭 단축 예상 (10분 → 2-3분)
================================================================
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, UpSampling1D, Bidirectional, LSTM,
    Concatenate, ReLU, BatchNormalization, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from sklearn.model_selection import train_test_split

# ============================================================
# 설정
# ============================================================
DATA_DIR = r"G:\stetho_ai\LUNet_Dataset_2sec"    # ★ 2초 데이터셋
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001


# ============================================================
# 1. LU-Net 아키텍처 (5초 버전과 동일 구조)
# ============================================================
def build_lunet(input_length, n_filters_base=32):
    """
    LU-Net: 1D Conv U-Net with Bi-LSTM skip connections

    2초 버전 해상도 변화:
      Enc1: 8000  (원본)
      Enc2: 4000  (↓2)
      Enc3: 2000  (↓2)
      Enc4: 1000  (↓2)
      Enc5: 500   (↓2) ← Bottleneck

    5초 대비 skip1 Bi-LSTM이 8000 스텝 처리 (vs 20000)
    → 학습 속도 대폭 향상
    """
    nf = n_filters_base  # 32

    # ----- Input -----
    inputs = Input(shape=(input_length, 1), name="noisy_input")

    # ----- Encoder -----
    # Enc1: 원본 해상도 (8000)
    enc1 = Conv1D(nf, kernel_size=7, padding="same")(inputs)
    enc1 = BatchNormalization()(enc1)
    enc1 = ReLU()(enc1)
    enc1 = Conv1D(nf, kernel_size=7, padding="same")(enc1)
    enc1 = BatchNormalization()(enc1)
    enc1 = ReLU()(enc1)

    # Enc2: stride 2 (4000)
    enc2 = Conv1D(nf * 2, kernel_size=7, strides=2, padding="same")(enc1)
    enc2 = BatchNormalization()(enc2)
    enc2 = ReLU()(enc2)
    enc2 = Conv1D(nf * 2, kernel_size=7, padding="same")(enc2)
    enc2 = BatchNormalization()(enc2)
    enc2 = ReLU()(enc2)

    # Enc3: (2000)
    enc3 = Conv1D(nf * 4, kernel_size=5, strides=2, padding="same")(enc2)
    enc3 = BatchNormalization()(enc3)
    enc3 = ReLU()(enc3)
    enc3 = Conv1D(nf * 4, kernel_size=5, padding="same")(enc3)
    enc3 = BatchNormalization()(enc3)
    enc3 = ReLU()(enc3)

    # Enc4: (1000)
    enc4 = Conv1D(nf * 8, kernel_size=3, strides=2, padding="same")(enc3)
    enc4 = BatchNormalization()(enc4)
    enc4 = ReLU()(enc4)
    enc4 = Conv1D(nf * 8, kernel_size=3, padding="same")(enc4)
    enc4 = BatchNormalization()(enc4)
    enc4 = ReLU()(enc4)

    # Enc5 (Bottleneck): (500)
    enc5 = Conv1D(nf * 16, kernel_size=3, strides=2, padding="same")(enc4)
    enc5 = BatchNormalization()(enc5)
    enc5 = ReLU()(enc5)
    enc5 = Conv1D(nf * 16, kernel_size=3, padding="same")(enc5)
    enc5 = BatchNormalization()(enc5)
    enc5 = ReLU()(enc5)

    # ----- Skip Connections with Bi-LSTM -----
    skip1 = Bidirectional(
        LSTM(32, return_sequences=True), name="bilstm_skip1"
    )(enc1)  # (8000, 64)

    skip2 = Bidirectional(
        LSTM(32, return_sequences=True), name="bilstm_skip2"
    )(enc2)  # (4000, 64)

    skip3 = Bidirectional(
        LSTM(64, return_sequences=True), name="bilstm_skip3"
    )(enc3)  # (2000, 128)

    skip4 = Bidirectional(
        LSTM(64, return_sequences=True), name="bilstm_skip4"
    )(enc4)  # (1000, 128)

    # ----- Decoder -----
    # Dec4: (500) → (1000)
    dec4 = UpSampling1D(size=2)(enc5)
    dec4 = Concatenate()([dec4, skip4])
    dec4 = Conv1D(nf * 8, kernel_size=3, padding="same")(dec4)
    dec4 = BatchNormalization()(dec4)
    dec4 = ReLU()(dec4)
    dec4 = Dropout(0.2)(dec4)

    # Dec3: (1000) → (2000)
    dec3 = UpSampling1D(size=2)(dec4)
    dec3 = Concatenate()([dec3, skip3])
    dec3 = Conv1D(nf * 4, kernel_size=5, padding="same")(dec3)
    dec3 = BatchNormalization()(dec3)
    dec3 = ReLU()(dec3)
    dec3 = Dropout(0.2)(dec3)

    # Dec2: (2000) → (4000)
    dec2 = UpSampling1D(size=2)(dec3)
    dec2 = Concatenate()([dec2, skip2])
    dec2 = Conv1D(nf * 2, kernel_size=7, padding="same")(dec2)
    dec2 = BatchNormalization()(dec2)
    dec2 = ReLU()(dec2)
    dec2 = Dropout(0.2)(dec2)

    # Dec1: (4000) → (8000)
    dec1 = UpSampling1D(size=2)(dec2)
    dec1 = Concatenate()([dec1, skip1])
    dec1 = Conv1D(nf, kernel_size=7, padding="same")(dec1)
    dec1 = BatchNormalization()(dec1)
    dec1 = ReLU()(dec1)

    # ----- Output -----
    outputs = Conv1D(1, kernel_size=1, padding="same",
                     activation="linear", name="clean_output")(dec1)

    model = Model(inputs=inputs, outputs=outputs, name="LU-Net-2sec")
    return model


# ============================================================
# 2. 학습
# ============================================================
def main():
    print("=" * 60)
    print(" LU-Net 학습 시작 (2초 버전)")
    print("=" * 60)

    # ----- 데이터 로드 -----
    print("\n[1] 데이터 로드 중...")
    X = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    Y = np.load(os.path.join(DATA_DIR, "Y_train.npy"))

    X = X[..., np.newaxis]
    Y = Y[..., np.newaxis]

    print(f"    X shape: {X.shape}")
    print(f"    Y shape: {Y.shape}")
    print(f"    X range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"    Y range: [{Y.min():.3f}, {Y.max():.3f}]")

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    print(f"    Train: {X_train.shape[0]}개, Val: {X_val.shape[0]}개")

    # ----- 모델 빌드 -----
    print("\n[2] LU-Net 모델 빌드 중 (2초 = 8000 samples)...")
    input_length = X_train.shape[1]
    model = build_lunet(input_length, n_filters_base=32)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )

    # ----- 콜백 -----
    callbacks = [
        ModelCheckpoint(
            os.path.join(DATA_DIR, "lunet_best_2sec.h5"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ----- 학습 -----
    print(f"\n[3] 학습 시작 (epochs={EPOCHS}, batch={BATCH_SIZE})")
    print("    2초 버전 → 에포크당 2-3분 예상\n")

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    # ----- 히스토리 저장 -----
    np.savez(
        os.path.join(DATA_DIR, "training_history_2sec.npz"),
        loss=history.history["loss"],
        val_loss=history.history["val_loss"],
        mae=history.history["mae"],
        val_mae=history.history["val_mae"],
    )

    print("\n" + "=" * 60)
    print(" 학습 완료! (2초 버전)")
    print(f"  모델: {os.path.join(DATA_DIR, 'lunet_best_2sec.h5')}")
    print(f"  최종 val_loss: {min(history.history['val_loss']):.6f}")
    print(f"  최종 val_mae:  {min(history.history['val_mae']):.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()