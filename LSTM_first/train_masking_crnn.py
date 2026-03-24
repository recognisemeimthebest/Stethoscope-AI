import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Conv1D, BatchNormalization, Activation, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

DATA_DIR = r"G:\stetho_ai\LSTM_first\Dataset_10000"

print("1. 데이터셋 로딩 및 스케일링 중...")
X = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
Y = np.load(os.path.join(DATA_DIR, 'Y_train.npy'))

X_min, X_max = np.min(X), np.max(X)
Y_min, Y_max = np.min(Y), np.max(Y)

X_scaled = (X - X_min) / (X_max - X_min + 1e-8)
Y_scaled = (Y - Y_min) / (Y_max - Y_min + 1e-8)

X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)
print(f" -> 훈련 데이터: {X_train.shape[0]}개, 검증 데이터: {X_val.shape[0]}개\n")

# ==========================================
# 🌟 [업그레이드] 마스킹(Masking) CRNN 아키텍처
# ==========================================
input_shape = (X_train.shape[1], X_train.shape[2]) # (79, 64)

# 1. 입력층 (원본 노이즈 스펙트로그램)
inputs = Input(shape=input_shape, name='noisy_input')

# 2. 특징 추출기 (Conv1D)
x = Conv1D(filters=64, kernel_size=3, padding='same')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv1D(filters=64, kernel_size=3, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)

# 3. 시간적 흐름 파악기 (Bi-LSTM)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Dropout(0.3)(x)

x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Dropout(0.3)(x)

# 4. 🔥 마스크 생성층 (0~1 사이의 통과 비율을 결정)
mask = Dense(64, activation='sigmoid', name='mask_layer')(x)

# 5. 🔥 핵심 곱셈 연산! (원본 Input * Mask)
# 노이즈는 0에 가깝게 곱해져 사라지고, 심음은 1에 가깝게 곱해져 살아남습니다.
outputs = Multiply(name='masked_output')([inputs, mask])

# 모델 조립
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()

# ==========================================
# 모델 학습 (Training) 설정
# ==========================================
print("\n3. 극한의 성능을 향해 학습을 시작합니다!")

checkpoint = ModelCheckpoint(
    os.path.join(DATA_DIR, 'masking_denoising_crnn.h5'), # 파일 이름 변경
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=100,
    batch_size=64,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

print("\n🎉 마스킹 모델 학습이 완료되었습니다! 'masking_denoising_crnn.h5'로 저장되었습니다.")