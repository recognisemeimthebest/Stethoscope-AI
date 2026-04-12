import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Conv1D, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

DATA_DIR = r"G:\stetho_ai\LSTM_first\Dataset_10000"

print("1. 저장된 데이터셋을 불러오는 중...")
X = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
Y = np.load(os.path.join(DATA_DIR, 'Y_train.npy'))

# 🌟 [업그레이드 1] 데이터 스케일링 (Min-Max Normalization)
# 나중에 예측된 값을 다시 데시벨(dB)로 복원하려면 이 min, max 값을 꼭 기억해야 합니다!
X_min, X_max = np.min(X), np.max(X)
Y_min, Y_max = np.min(Y), np.max(Y)

print(f" -> 원본 스케일 범위: {X_min:.2f}dB ~ {X_max:.2f}dB")

X_scaled = (X - X_min) / (X_max - X_min + 1e-8)  # 0분모 방지용 1e-8
Y_scaled = (Y - Y_min) / (Y_max - Y_min + 1e-8)

# 훈련/검증 데이터 분리 (8:2)
X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)
print(f" -> 훈련 데이터: {X_train.shape[0]}개, 검증 데이터: {X_val.shape[0]}개\n")

# ==========================================
# 🌟 [업그레이드 2] CRNN 아키텍처 설계 (Conv1D + Bi-LSTM)
# ==========================================
input_shape = (X_train.shape[1], X_train.shape[2])  # (79, 64)

model = Sequential([
    # 1. 시각적 특징 추출기 (Conv1D)
    Conv1D(filters=64, kernel_size=3, padding='same', input_shape=input_shape),
    BatchNormalization(),
    Activation('relu'),

    Conv1D(filters=64, kernel_size=3, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.2),

    # 2. 시간적 흐름 파악기 (Bi-LSTM)
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),

    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),

    # 3. 출력층: 스케일링된 0~1 사이의 값으로 출력하므로 sigmoid 사용
    Dense(64, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()

# ==========================================
# 3. 모델 학습 (Training) 설정
# ==========================================
print("\n3. 본격적인 학습을 시작합니다! (장기전 돌입)")

checkpoint = ModelCheckpoint(
    os.path.join(DATA_DIR, 'pro_denoising_crnn.h5'),
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# 🌟 [업그레이드 3] 학습률 스케줄러 (진전이 없으면 보폭을 반으로 줄임)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # 학습률을 50%로 감소
    patience=3,  # 3번의 에포크 동안 개선이 없으면 발동
    min_lr=1e-6,  # 최소 하한선
    verbose=1
)

# 훈련 시작!
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=100,  # Early Stopping이 있으니 넉넉하게 100
    batch_size=64,  # GPU 효율을 위해 배치 사이즈업
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# 나중에 스케일 복원을 위해 min/max 값도 저장해둡니다.
np.save(os.path.join(DATA_DIR, 'scaling_params.npy'), np.array([Y_min, Y_max]))

print("\n🎉 완벽하게 학습되었습니다! 최고 성능 모델과 스케일링 파라미터가 저장되었습니다.")