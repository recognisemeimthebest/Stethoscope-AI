"""
================================================================
TU-Net v2 - TCN Skip + Attention + Residual + SI-SNR 90:10
================================================================
LU-Net v2 (90:10)과 동일하되 Bi-LSTM → TCN 교체

비교 대상: LU-Net v2 (ΔSNR +5.59, SI-SNR 12.76)
================================================================
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, UpSampling1D,
    Concatenate, ReLU, BatchNormalization, Dropout,
    GlobalAveragePooling1D, Dense, Reshape, Multiply, Subtract,
    Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K

DATA_DIR = r"G:\stetho_ai\LUNet_Dataset"
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001


# ============================================================
# Loss (90:10)
# ============================================================
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


# ============================================================
# TCN Block
# ============================================================
def tcn_block(x, filters, kernel_size=3, dilations=[1, 2, 4, 8],
              name_prefix="tcn"):
    skip_connections = []

    for i, d in enumerate(dilations):
        conv = Conv1D(filters, kernel_size, dilation_rate=d,
                      padding="causal",
                      name=f"{name_prefix}_dconv_d{d}")(x)
        conv = BatchNormalization(name=f"{name_prefix}_bn_{i}")(conv)
        conv = ReLU(name=f"{name_prefix}_relu_{i}")(conv)
        conv = Dropout(0.1, name=f"{name_prefix}_drop_{i}")(conv)

        res = Conv1D(filters, 1, padding="same",
                     name=f"{name_prefix}_res_{i}")(conv)

        if x.shape[-1] != filters:
            x = Conv1D(filters, 1, padding="same",
                       name=f"{name_prefix}_match_{i}")(x)

        x = Add(name=f"{name_prefix}_add_{i}")([x, res])
        skip_connections.append(res)

    if len(skip_connections) > 1:
        out = Add(name=f"{name_prefix}_skip_sum")(skip_connections)
    else:
        out = skip_connections[0]

    out = ReLU(name=f"{name_prefix}_out_relu")(out)
    return out


# ============================================================
# Channel Attention
# ============================================================
def channel_attention(x, ratio=4, name_prefix="ca"):
    channels = x.shape[-1]
    squeeze = GlobalAveragePooling1D(name=f"{name_prefix}_gap")(x)
    excitation = Dense(channels // ratio, activation="relu",
                       name=f"{name_prefix}_fc1")(squeeze)
    excitation = Dense(channels, activation="sigmoid",
                       name=f"{name_prefix}_fc2")(excitation)
    excitation = Reshape((1, channels),
                         name=f"{name_prefix}_reshape")(excitation)
    return Multiply(name=f"{name_prefix}_mul")([x, excitation])


# ============================================================
# TU-Net v2 아키텍처
# ============================================================
def build_tunet_v2(input_length, n_filters_base=32):
    """
    TU-Net v2: TCN + Attention + Residual Learning
    LU-Net v2와 동일 구조, Bi-LSTM → TCN 교체
    """
    nf = n_filters_base

    inputs = Input(shape=(input_length, 1), name="noisy_input")

    # ----- Encoder -----
    enc1 = Conv1D(nf, kernel_size=7, padding="same")(inputs)
    enc1 = BatchNormalization()(enc1)
    enc1 = ReLU()(enc1)
    enc1 = Conv1D(nf, kernel_size=7, padding="same")(enc1)
    enc1 = BatchNormalization()(enc1)
    enc1 = ReLU()(enc1)

    enc2 = Conv1D(nf * 2, kernel_size=7, strides=2, padding="same")(enc1)
    enc2 = BatchNormalization()(enc2)
    enc2 = ReLU()(enc2)
    enc2 = Conv1D(nf * 2, kernel_size=7, padding="same")(enc2)
    enc2 = BatchNormalization()(enc2)
    enc2 = ReLU()(enc2)

    enc3 = Conv1D(nf * 4, kernel_size=5, strides=2, padding="same")(enc2)
    enc3 = BatchNormalization()(enc3)
    enc3 = ReLU()(enc3)
    enc3 = Conv1D(nf * 4, kernel_size=5, padding="same")(enc3)
    enc3 = BatchNormalization()(enc3)
    enc3 = ReLU()(enc3)

    enc4 = Conv1D(nf * 8, kernel_size=3, strides=2, padding="same")(enc3)
    enc4 = BatchNormalization()(enc4)
    enc4 = ReLU()(enc4)
    enc4 = Conv1D(nf * 8, kernel_size=3, padding="same")(enc4)
    enc4 = BatchNormalization()(enc4)
    enc4 = ReLU()(enc4)

    enc5 = Conv1D(nf * 16, kernel_size=3, strides=2, padding="same")(enc4)
    enc5 = BatchNormalization()(enc5)
    enc5 = ReLU()(enc5)
    enc5 = Conv1D(nf * 16, kernel_size=3, padding="same")(enc5)
    enc5 = BatchNormalization()(enc5)
    enc5 = ReLU()(enc5)

    # ----- Skip: TCN + Attention -----
    skip1 = tcn_block(enc1, filters=64, kernel_size=3,
                      dilations=[1, 2, 4, 8, 16, 32],
                      name_prefix="tcn_skip1")
    skip1 = channel_attention(skip1, ratio=4, name_prefix="ca_skip1")

    skip2 = tcn_block(enc2, filters=64, kernel_size=3,
                      dilations=[1, 2, 4, 8, 16],
                      name_prefix="tcn_skip2")
    skip2 = channel_attention(skip2, ratio=4, name_prefix="ca_skip2")

    skip3 = tcn_block(enc3, filters=128, kernel_size=3,
                      dilations=[1, 2, 4, 8],
                      name_prefix="tcn_skip3")
    skip3 = channel_attention(skip3, ratio=4, name_prefix="ca_skip3")

    skip4 = tcn_block(enc4, filters=128, kernel_size=3,
                      dilations=[1, 2, 4],
                      name_prefix="tcn_skip4")
    skip4 = channel_attention(skip4, ratio=4, name_prefix="ca_skip4")

    # ----- Decoder -----
    dec4 = UpSampling1D(size=2)(enc5)
    dec4 = Concatenate()([dec4, skip4])
    dec4 = Conv1D(nf * 8, kernel_size=3, padding="same")(dec4)
    dec4 = BatchNormalization()(dec4)
    dec4 = ReLU()(dec4)
    dec4 = Dropout(0.2)(dec4)

    dec3 = UpSampling1D(size=2)(dec4)
    dec3 = Concatenate()([dec3, skip3])
    dec3 = Conv1D(nf * 4, kernel_size=5, padding="same")(dec3)
    dec3 = BatchNormalization()(dec3)
    dec3 = ReLU()(dec3)
    dec3 = Dropout(0.2)(dec3)

    dec2 = UpSampling1D(size=2)(dec3)
    dec2 = Concatenate()([dec2, skip2])
    dec2 = Conv1D(nf * 2, kernel_size=7, padding="same")(dec2)
    dec2 = BatchNormalization()(dec2)
    dec2 = ReLU()(dec2)
    dec2 = Dropout(0.2)(dec2)

    dec1 = UpSampling1D(size=2)(dec2)
    dec1 = Concatenate()([dec1, skip1])
    dec1 = Conv1D(nf, kernel_size=7, padding="same")(dec1)
    dec1 = BatchNormalization()(dec1)
    dec1 = ReLU()(dec1)

    # ----- Residual Learning: noise estimation -----
    noise_estimate = Conv1D(1, kernel_size=1, padding="same",
                            activation="linear",
                            name="noise_estimate")(dec1)
    outputs = Subtract(name="clean_output")([inputs, noise_estimate])

    model = Model(inputs=inputs, outputs=outputs, name="TU-Net-v2")
    return model


# ============================================================
# 학습
# ============================================================
def main():
    print("=" * 60)
    print(" TU-Net v2 학습 시작")
    print(" (TCN + Attention + Residual + SI-SNR 90:10)")
    print("=" * 60)

    print("\n[1] 데이터 로드 중...")
    X = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    Y = np.load(os.path.join(DATA_DIR, "Y_train.npy"))
    X = X[..., np.newaxis]
    Y = Y[..., np.newaxis]

    print(f"    X shape: {X.shape}")
    print(f"    Y shape: {Y.shape}")

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    print(f"    Train: {X_train.shape[0]}개, Val: {X_val.shape[0]}개")

    print("\n[2] TU-Net v2 모델 빌드 중...")
    input_length = X_train.shape[1]
    model = build_tunet_v2(input_length, n_filters_base=32)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=combined_loss,
        metrics=["mae"],
    )

    callbacks = [
        ModelCheckpoint(
            os.path.join(DATA_DIR, "tunet_v2_best.h5"),
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=12,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4,
            min_lr=1e-6, verbose=1,
        ),
    ]

    print(f"\n[3] 학습 시작 (epochs={EPOCHS}, batch={BATCH_SIZE})")
    print("    TCN → LSTM 대비 2-3배 빠를 예상\n")

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    np.savez(
        os.path.join(DATA_DIR, "training_history_tunet_v2.npz"),
        loss=history.history["loss"],
        val_loss=history.history["val_loss"],
        mae=history.history["mae"],
        val_mae=history.history["val_mae"],
    )

    print("\n" + "=" * 60)
    print(" TU-Net v2 학습 완료!")
    print(f"  모델: tunet_v2_best.h5")
    print(f"  최종 val_loss: {min(history.history['val_loss']):.6f}")
    print(f"  최종 val_mae:  {min(history.history['val_mae']):.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()