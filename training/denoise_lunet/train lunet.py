"""
================================================================
LU-Net 심음 노이즈 제거 - Step 2: 모델 구축 + 학습
================================================================
아키텍처: LU-Net (Ali et al., 2023 IEEE Access)
  - Encoder: 5-level 1D Conv (stride 2 다운샘플링)
  - Skip connections: 각 레벨에 Bi-LSTM
  - Decoder: 4-level 1D Conv + UpSampling1D
  - 입출력: raw waveform (1D time-domain)

Gemini 파이프라인 대비 핵심 차이:
  1. Time-domain 직접 처리 (스펙트로그램 변환/역변환 없음)
  2. 다중 스케일 Bi-LSTM (단일 LSTM이 아님)
  3. U-Net skip connection으로 세밀한 파형 보존

사용법:
  python 02_train_lunet.py

필요 라이브러리:
  pip install tensorflow numpy scikit-learn
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
DATA_DIR = r"G:\stetho_ai\LUNet_Dataset"
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001


# ============================================================
# 1. LU-Net 아키텍처
# ============================================================
def build_lunet(input_length, n_filters_base=32):
    """
    LU-Net: 1D Conv U-Net with Bi-LSTM skip connections

    구조:
      Encoder (5단):
        Enc1: Conv1D(32)           → 원본 해상도 (20000)
        Enc2: Conv1D(64, stride=2) → ½ (10000)
        Enc3: Conv1D(128, stride=2)→ ¼ (5000)
        Enc4: Conv1D(256, stride=2)→ ⅛ (2500)
        Enc5: Conv1D(512, stride=2)→ 1/16 (1250) ← Bottleneck

      Skip connections (4개):
        각 Enc 출력 → Bi-LSTM → Decoder에 Concatenate

      Decoder (4단):
        Dec4: UpSample + Concat + Conv1D(256)
        Dec3: UpSample + Concat + Conv1D(128)
        Dec2: UpSample + Concat + Conv1D(64)
        Dec1: Concat + Conv1D(32)
        Final: Conv1D(1) → denoised waveform
    """
    nf = n_filters_base  # 32

    # ----- Input -----
    # (batch, 20000, 1) - mono waveform
    inputs = Input(shape=(input_length, 1), name="noisy_input")

    # ----- Encoder -----
    # Enc1: 원본 해상도
    enc1 = Conv1D(nf, kernel_size=7, padding="same")(inputs)
    enc1 = BatchNormalization()(enc1)
    enc1 = ReLU()(enc1)
    enc1 = Conv1D(nf, kernel_size=7, padding="same")(enc1)
    enc1 = BatchNormalization()(enc1)
    enc1 = ReLU()(enc1)
    # shape: (20000, 32)

    # Enc2: stride 2 다운샘플링
    enc2 = Conv1D(nf * 2, kernel_size=7, strides=2, padding="same")(enc1)
    enc2 = BatchNormalization()(enc2)
    enc2 = ReLU()(enc2)
    enc2 = Conv1D(nf * 2, kernel_size=7, padding="same")(enc2)
    enc2 = BatchNormalization()(enc2)
    enc2 = ReLU()(enc2)
    # shape: (10000, 64)

    # Enc3
    enc3 = Conv1D(nf * 4, kernel_size=5, strides=2, padding="same")(enc2)
    enc3 = BatchNormalization()(enc3)
    enc3 = ReLU()(enc3)
    enc3 = Conv1D(nf * 4, kernel_size=5, padding="same")(enc3)
    enc3 = BatchNormalization()(enc3)
    enc3 = ReLU()(enc3)
    # shape: (5000, 128)

    # Enc4
    enc4 = Conv1D(nf * 8, kernel_size=3, strides=2, padding="same")(enc3)
    enc4 = BatchNormalization()(enc4)
    enc4 = ReLU()(enc4)
    enc4 = Conv1D(nf * 8, kernel_size=3, padding="same")(enc4)
    enc4 = BatchNormalization()(enc4)
    enc4 = ReLU()(enc4)
    # shape: (2500, 256)

    # Enc5 (Bottleneck)
    enc5 = Conv1D(nf * 16, kernel_size=3, strides=2, padding="same")(enc4)
    enc5 = BatchNormalization()(enc5)
    enc5 = ReLU()(enc5)
    enc5 = Conv1D(nf * 16, kernel_size=3, padding="same")(enc5)
    enc5 = BatchNormalization()(enc5)
    enc5 = ReLU()(enc5)
    # shape: (1250, 512)

    # ----- Skip Connections with Bi-LSTM -----
    # 각 스케일에서 시간적 문맥을 학습
    # LSTM units를 스케일에 따라 조절 (낮은 해상도 = 적은 유닛)
    skip1 = Bidirectional(
        LSTM(32, return_sequences=True), name="bilstm_skip1"
    )(enc1)  # (20000, 64)

    skip2 = Bidirectional(
        LSTM(32, return_sequences=True), name="bilstm_skip2"
    )(enc2)  # (10000, 64)

    skip3 = Bidirectional(
        LSTM(64, return_sequences=True), name="bilstm_skip3"
    )(enc3)  # (5000, 128)

    skip4 = Bidirectional(
        LSTM(64, return_sequences=True), name="bilstm_skip4"
    )(enc4)  # (2500, 128)

    # ----- Decoder -----
    # Dec4: (1250) → UpSample → (2500) + skip4
    dec4 = UpSampling1D(size=2)(enc5)       # (2500, 512)
    dec4 = Concatenate()([dec4, skip4])      # (2500, 512+128=640)
    dec4 = Conv1D(nf * 8, kernel_size=3, padding="same")(dec4)
    dec4 = BatchNormalization()(dec4)
    dec4 = ReLU()(dec4)
    dec4 = Dropout(0.2)(dec4)
    # (2500, 256)

    # Dec3: (2500) → UpSample → (5000) + skip3
    dec3 = UpSampling1D(size=2)(dec4)       # (5000, 256)
    dec3 = Concatenate()([dec3, skip3])      # (5000, 256+128=384)
    dec3 = Conv1D(nf * 4, kernel_size=5, padding="same")(dec3)
    dec3 = BatchNormalization()(dec3)
    dec3 = ReLU()(dec3)
    dec3 = Dropout(0.2)(dec3)
    # (5000, 128)

    # Dec2: (5000) → UpSample → (10000) + skip2
    dec2 = UpSampling1D(size=2)(dec3)       # (10000, 128)
    dec2 = Concatenate()([dec2, skip2])      # (10000, 128+64=192)
    dec2 = Conv1D(nf * 2, kernel_size=7, padding="same")(dec2)
    dec2 = BatchNormalization()(dec2)
    dec2 = ReLU()(dec2)
    dec2 = Dropout(0.2)(dec2)
    # (10000, 64)

    # Dec1: (10000) → UpSample → (20000) + skip1
    dec1 = UpSampling1D(size=2)(dec2)       # (20000, 64)
    dec1 = Concatenate()([dec1, skip1])      # (20000, 64+64=128)
    dec1 = Conv1D(nf, kernel_size=7, padding="same")(dec1)
    dec1 = BatchNormalization()(dec1)
    dec1 = ReLU()(dec1)
    # (20000, 32)

    # ----- Output layer -----
    # Linear activation: waveform은 -1 ~ +1 범위
    outputs = Conv1D(1, kernel_size=1, padding="same",
                     activation="linear", name="clean_output")(dec1)
    # (20000, 1)

    model = Model(inputs=inputs, outputs=outputs, name="LU-Net")
    return model


# ============================================================
# 2. 학습
# ============================================================
def main():
    print("=" * 60)
    print(" LU-Net 학습 시작")
    print("=" * 60)

    # ----- 데이터 로드 -----
    print("\n[1] 데이터 로드 중...")
    X = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    Y = np.load(os.path.join(DATA_DIR, "Y_train.npy"))

    # (samples, length) → (samples, length, 1) 채널 차원 추가
    X = X[..., np.newaxis]
    Y = Y[..., np.newaxis]

    print(f"    X shape: {X.shape}")
    print(f"    Y shape: {Y.shape}")
    print(f"    X range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"    Y range: [{Y.min():.3f}, {Y.max():.3f}]")

    # Train/Val 분리 (80:20)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    print(f"    Train: {X_train.shape[0]}개, Val: {X_val.shape[0]}개")

    # ----- 모델 빌드 -----
    print("\n[2] LU-Net 모델 빌드 중...")
    input_length = X_train.shape[1]  # 20000
    model = build_lunet(input_length, n_filters_base=32)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )

    # ----- 콜백 설정 -----
    callbacks = [
        ModelCheckpoint(
            os.path.join(DATA_DIR, "lunet_best.h5"),
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

    # ----- 학습 시작 -----
    print(f"\n[3] 학습 시작 (epochs={EPOCHS}, batch={BATCH_SIZE})")
    print("    GPU가 있으면 자동으로 사용됩니다.")
    print("    RTX 4070 Ti Super → 에포크당 약 1-3분 예상\n")

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    # ----- 학습 히스토리 저장 -----
    np.savez(
        os.path.join(DATA_DIR, "training_history.npz"),
        loss=history.history["loss"],
        val_loss=history.history["val_loss"],
        mae=history.history["mae"],
        val_mae=history.history["val_mae"],
    )

    print("\n" + "=" * 60)
    print(" 학습 완료!")
    print(f"  최고 성능 모델: {os.path.join(DATA_DIR, 'lunet_best.h5')}")
    print(f"  최종 val_loss: {min(history.history['val_loss']):.6f}")
    print(f"  최종 val_mae:  {min(history.history['val_mae']):.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()