"""
================================================================
LU-Net v2 - Attention + Residual Learning + SI-SNR Loss
================================================================
5초 LU-Net baseline (ΔSNR +5.94 dB) 위에 세 가지 개선 적용:

1. Channel Attention (SE Block) on skip connections
   - Bi-LSTM 출력에서 "어떤 채널이 심음에 중요한지" 가중치 학습
   - 논문 대비 아키텍처 차별점

2. Residual Learning
   - clean = input - noise 대신, noise를 예측하고 input에서 빼기
   - 모델이 "빼야 할 것만" 집중 → sparse한 문제로 단순화

3. SI-SNR Loss
   - MSE는 숫자 차이만 봄 → SI-SNR은 파형 형태 일치도를 직접 최적화
   - 음성 분리 분야 표준 손실 함수

사용법:
  기존 5초 데이터셋(G:\stetho_ai\LUNet_Dataset) 그대로 사용
  python 02_train_lunet_v2.py

필요 라이브러리:
  pip install tensorflow numpy scikit-learn
================================================================
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, UpSampling1D, Bidirectional, LSTM,
    Concatenate, ReLU, BatchNormalization, Dropout,
    GlobalAveragePooling1D, Dense, Reshape, Multiply, Subtract
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K

# ============================================================
# 설정
# ============================================================
DATA_DIR = r"G:\stetho_ai\LUNet_Dataset"       # 기존 5초 데이터셋
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001


# ============================================================
# 1. SI-SNR Loss (Scale-Invariant Signal-to-Noise Ratio)
# ============================================================
def si_snr_loss(y_true, y_pred):
    """
    SI-SNR Loss: 파형의 형태 일치도를 직접 최적화
    - MSE는 각 시점의 값 차이만 봄
    - SI-SNR은 전체 파형의 shape이 얼마나 일치하는지 측정
    - loss = -SI-SNR (최대화해야 하니까 음수)

    참고: Conv-TasNet (Luo & Mesgarani, 2019) 이후 표준
    """
    # (batch, time, 1) → (batch, time)
    y_true = K.squeeze(y_true, axis=-1)
    y_pred = K.squeeze(y_pred, axis=-1)

    # zero-mean normalization
    y_true = y_true - K.mean(y_true, axis=-1, keepdims=True)
    y_pred = y_pred - K.mean(y_pred, axis=-1, keepdims=True)

    # s_target = <y_pred, y_true> * y_true / ||y_true||^2
    dot = K.sum(y_pred * y_true, axis=-1, keepdims=True)
    s_target = dot * y_true / (K.sum(y_true ** 2, axis=-1, keepdims=True) + 1e-8)

    # e_noise = y_pred - s_target
    e_noise = y_pred - s_target

    # SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    si_snr = 10 * K.log(
        K.sum(s_target ** 2, axis=-1) / (K.sum(e_noise ** 2, axis=-1) + 1e-8)
    ) / K.log(10.0)

    # loss = -SI-SNR (음수를 최소화 = SI-SNR 최대화)
    return -K.mean(si_snr)


def combined_loss(y_true, y_pred):
    """
    복합 손실: MSE + SI-SNR Loss
    - MSE가 빠른 수렴을 돕고
    - SI-SNR이 음질 최적화를 담당
    """
    mse = K.mean(K.square(y_true - y_pred))
    si_snr = si_snr_loss(y_true, y_pred)
    return 0.9 * mse + 0.1 * (si_snr / 20.0)


# ============================================================
# 2. Channel Attention (SE Block)
# ============================================================
def channel_attention(x, ratio=4, name_prefix="ca"):
    """
    Squeeze-and-Excitation (SE) Block for 1D signals

    동작 원리:
    1. Global Average Pooling으로 각 채널의 "중요도" 요약
    2. FC → ReLU → FC → Sigmoid로 채널별 가중치(0~1) 생성
    3. 원본 feature에 곱해서 중요한 채널 강조, 불필요한 채널 억제

    심음에서의 역할:
    - S1/S2 관련 채널 → 가중치 높음 (보존)
    - 노이즈 관련 채널 → 가중치 낮음 (억제)
    """
    channels = x.shape[-1]
    squeeze = GlobalAveragePooling1D(name=f"{name_prefix}_gap")(x)

    excitation = Dense(
        channels // ratio, activation="relu",
        name=f"{name_prefix}_fc1"
    )(squeeze)
    excitation = Dense(
        channels, activation="sigmoid",
        name=f"{name_prefix}_fc2"
    )(excitation)

    # (batch, channels) → (batch, 1, channels) for broadcasting
    excitation = Reshape((1, channels), name=f"{name_prefix}_reshape")(excitation)

    return Multiply(name=f"{name_prefix}_mul")([x, excitation])


# ============================================================
# 3. LU-Net v2 아키텍처
# ============================================================
def build_lunet_v2(input_length, n_filters_base=32):
    """
    LU-Net v2: Attention + Residual Learning

    vs v1 (baseline):
      + Channel Attention on each skip connection
      + Residual learning: output = input - predicted_noise
      + 동일 파라미터 수 대비 더 효과적인 feature selection
    """
    nf = n_filters_base  # 32

    # ----- Input -----
    inputs = Input(shape=(input_length, 1), name="noisy_input")

    # ----- Encoder -----
    # Enc1: 원본 해상도 (20000)
    enc1 = Conv1D(nf, kernel_size=7, padding="same")(inputs)
    enc1 = BatchNormalization()(enc1)
    enc1 = ReLU()(enc1)
    enc1 = Conv1D(nf, kernel_size=7, padding="same")(enc1)
    enc1 = BatchNormalization()(enc1)
    enc1 = ReLU()(enc1)

    # Enc2: (10000)
    enc2 = Conv1D(nf * 2, kernel_size=7, strides=2, padding="same")(enc1)
    enc2 = BatchNormalization()(enc2)
    enc2 = ReLU()(enc2)
    enc2 = Conv1D(nf * 2, kernel_size=7, padding="same")(enc2)
    enc2 = BatchNormalization()(enc2)
    enc2 = ReLU()(enc2)

    # Enc3: (5000)
    enc3 = Conv1D(nf * 4, kernel_size=5, strides=2, padding="same")(enc2)
    enc3 = BatchNormalization()(enc3)
    enc3 = ReLU()(enc3)
    enc3 = Conv1D(nf * 4, kernel_size=5, padding="same")(enc3)
    enc3 = BatchNormalization()(enc3)
    enc3 = ReLU()(enc3)

    # Enc4: (2500)
    enc4 = Conv1D(nf * 8, kernel_size=3, strides=2, padding="same")(enc3)
    enc4 = BatchNormalization()(enc4)
    enc4 = ReLU()(enc4)
    enc4 = Conv1D(nf * 8, kernel_size=3, padding="same")(enc4)
    enc4 = BatchNormalization()(enc4)
    enc4 = ReLU()(enc4)

    # Enc5 (Bottleneck): (1250)
    enc5 = Conv1D(nf * 16, kernel_size=3, strides=2, padding="same")(enc4)
    enc5 = BatchNormalization()(enc5)
    enc5 = ReLU()(enc5)
    enc5 = Conv1D(nf * 16, kernel_size=3, padding="same")(enc5)
    enc5 = BatchNormalization()(enc5)
    enc5 = ReLU()(enc5)

    # ----- Skip Connections: Bi-LSTM + Channel Attention -----
    # [개선점] Bi-LSTM 출력에 SE Block 추가
    skip1 = Bidirectional(
        LSTM(32, return_sequences=True), name="bilstm_skip1"
    )(enc1)
    skip1 = channel_attention(skip1, ratio=4, name_prefix="ca_skip1")

    skip2 = Bidirectional(
        LSTM(32, return_sequences=True), name="bilstm_skip2"
    )(enc2)
    skip2 = channel_attention(skip2, ratio=4, name_prefix="ca_skip2")

    skip3 = Bidirectional(
        LSTM(64, return_sequences=True), name="bilstm_skip3"
    )(enc3)
    skip3 = channel_attention(skip3, ratio=4, name_prefix="ca_skip3")

    skip4 = Bidirectional(
        LSTM(64, return_sequences=True), name="bilstm_skip4"
    )(enc4)
    skip4 = channel_attention(skip4, ratio=4, name_prefix="ca_skip4")

    # ----- Decoder -----
    # Dec4: (1250) → (2500)
    dec4 = UpSampling1D(size=2)(enc5)
    dec4 = Concatenate()([dec4, skip4])
    dec4 = Conv1D(nf * 8, kernel_size=3, padding="same")(dec4)
    dec4 = BatchNormalization()(dec4)
    dec4 = ReLU()(dec4)
    dec4 = Dropout(0.2)(dec4)

    # Dec3: (2500) → (5000)
    dec3 = UpSampling1D(size=2)(dec4)
    dec3 = Concatenate()([dec3, skip3])
    dec3 = Conv1D(nf * 4, kernel_size=5, padding="same")(dec3)
    dec3 = BatchNormalization()(dec3)
    dec3 = ReLU()(dec3)
    dec3 = Dropout(0.2)(dec3)

    # Dec2: (5000) → (10000)
    dec2 = UpSampling1D(size=2)(dec3)
    dec2 = Concatenate()([dec2, skip2])
    dec2 = Conv1D(nf * 2, kernel_size=7, padding="same")(dec2)
    dec2 = BatchNormalization()(dec2)
    dec2 = ReLU()(dec2)
    dec2 = Dropout(0.2)(dec2)

    # Dec1: (10000) → (20000)
    dec1 = UpSampling1D(size=2)(dec2)
    dec1 = Concatenate()([dec1, skip1])
    dec1 = Conv1D(nf, kernel_size=7, padding="same")(dec1)
    dec1 = BatchNormalization()(dec1)
    dec1 = ReLU()(dec1)

    # ----- [Residual Learning] 노이즈 예측 → 입력에서 빼기 -----
    # 모델이 clean 전체를 복원하는 대신, noise만 예측
    # clean = input - noise_estimate
    noise_estimate = Conv1D(
        1, kernel_size=1, padding="same",
        activation="linear", name="noise_estimate"
    )(dec1)

    # output = input - predicted_noise
    outputs = Subtract(name="clean_output")([inputs, noise_estimate])

    model = Model(inputs=inputs, outputs=outputs, name="LU-Net-v2")
    return model


# ============================================================
# 4. 학습
# ============================================================
def main():
    print("=" * 60)
    print(" LU-Net v2 학습 시작")
    print(" (Attention + Residual + SI-SNR Loss)")
    print("=" * 60)

    # ----- 데이터 로드 -----
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

    # ----- 모델 빌드 -----
    print("\n[2] LU-Net v2 모델 빌드 중...")
    input_length = X_train.shape[1]
    model = build_lunet_v2(input_length, n_filters_base=32)
    model.summary()

    # [개선점] SI-SNR + MSE 복합 손실 함수
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=combined_loss,
        metrics=["mae"],
    )

    # ----- 콜백 -----
    callbacks = [
        ModelCheckpoint(
            os.path.join(DATA_DIR, "lunet_v2_90_10.h5"),
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
    print("    v1 대비 변경: Attention + Residual + SI-SNR Loss")
    print("    에포크당 ~10-12분 예상 (5초 데이터)\n")

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    # ----- 히스토리 저장 -----
    np.savez(
        os.path.join(DATA_DIR, "training_history_v2.npz"),
        loss=history.history["loss"],
        val_loss=history.history["val_loss"],
        mae=history.history["mae"],
        val_mae=history.history["val_mae"],
    )

    print("\n" + "=" * 60)
    print(" LU-Net v2 학습 완료!")
    print(f"  모델: {os.path.join(DATA_DIR, 'lunet_v2_90_10.h5')}")
    print(f"  최종 val_loss: {min(history.history['val_loss']):.6f}")
    print(f"  최종 val_mae:  {min(history.history['val_mae']):.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()