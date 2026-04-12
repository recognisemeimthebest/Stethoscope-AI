"""
On-Device BPM Noise Gate Model Architecture
============================================
논문 기반:
  - Lightweight 1D CNN (IEEE 9921376, PhysioNet 2016 Challenge 접근법)
  - TFLite Micro 호환 (표준 Conv1D, MaxPool1D, Dense 연산만 사용)

아키텍처:
  Input: (100, 1) - 100프레임 Shannon Energy Envelope (1초 @ 10ms/frame)
  ↓ Conv1D(16, 7) → (94, 16)
  ↓ MaxPool1D(2)  → (47, 16)
  ↓ Conv1D(32, 5) → (43, 32)
  ↓ MaxPool1D(2)  → (21, 32)
  ↓ Conv1D(32, 3) → (19, 32)
  ↓ GlobalAvgPool → (32,)
  ↓ Dense(16)     → (16,)
  ↓ Dense(1, sig) → (1,)  # 심장음 확률

총 파라미터: ~6,400개
INT8 양자화 후 크기: ~7KB
ESP32 추론 시간: ~15ms
"""

import tensorflow as tf


def build_noise_gate(input_length=100, name="noise_gate"):
    """
    심장음 존재 여부를 판별하는 Tiny 1D CNN (Noise Gate).

    Args:
        input_length: 입력 창 크기 (프레임 수, default=100=1초)
        name: 모델 이름

    Returns:
        tf.keras.Model: 컴파일된 모델
    """
    inputs = tf.keras.Input(shape=(input_length, 1), name="envelope_input")

    # Block 1: 저주파 패턴 감지 (심음의 주기적 에너지 변화)
    x = tf.keras.layers.Conv1D(
        filters=16, kernel_size=7, activation="relu",
        padding="valid", name="conv1"
    )(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=2, name="pool1")(x)

    # Block 2: 중간 주파수 특징
    x = tf.keras.layers.Conv1D(
        filters=32, kernel_size=5, activation="relu",
        padding="valid", name="conv2"
    )(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2, name="pool2")(x)

    # Block 3: 고수준 특징 추출
    x = tf.keras.layers.Conv1D(
        filters=32, kernel_size=3, activation="relu",
        padding="valid", name="conv3"
    )(x)

    # 전역 평균 풀링 (공간 정보 집약, 크기 불변)
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)

    # 분류기
    x = tf.keras.layers.Dense(16, activation="relu", name="fc1")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def compile_model(model, learning_rate=1e-3):
    """모델 컴파일."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model


if __name__ == "__main__":
    model = build_noise_gate()
    compile_model(model)
    model.summary()

    # 파라미터 수 출력
    total_params = model.count_params()
    print(f"\n총 파라미터: {total_params:,}")
    print(f"INT8 양자화 후 예상 크기: ~{total_params // 1024 + 5} KB")
