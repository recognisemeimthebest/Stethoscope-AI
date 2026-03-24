import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
import numpy as np

# 1. 데이터 로드
print("데이터를 불러오는 중...")
x_mfcc = np.load('x_mfcc.npy')
y_label = np.load('y_label.npy')

# 데이터가 잘 섞여있는지 확인 (학습 전 마지막 셔플)
indices = np.arange(len(x_mfcc))
np.random.shuffle(indices)
x_mfcc = x_mfcc[indices]
y_label = y_label[indices]

# 학습용(80%) / 검증용(20%) 분리
x_train, x_val, y_train, y_val = train_test_split(x_mfcc, y_label, test_size=0.2, random_state=42)

# 2. 모델 설계 (ESP32 맞춤형 Tiny-CNN)
model = models.Sequential([
    # 1st Conv: 심음의 기본적인 주파수 패턴 인식
    layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(13, 32, 1)),
    layers.MaxPooling2D((2, 2)),

    # 2nd Conv: 더 복잡한 리듬 패턴 인식
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    # 과적합 방지를 위한 Dropout (종완님 의견 반영 0.3)
    layers.Dropout(0.3),

    # Dense Layer: 특징 결합
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),

    # 최종 출력: 0.5보다 크면 심음(1), 작으면 노이즈(0)
    layers.Dense(1, activation='sigmoid')
])

# 3. 컴파일 (Learning Rate를 0.0005로 낮춰 정밀하게 학습)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. EarlyStopping & ModelCheckpoint
# val_loss가 8번의 에포크 동안 개선되지 않으면 조기 종료
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

# 5. 학습 진행
print(f"모델 학습 시작... (총 샘플 수: {len(x_train)})")
history = model.fit(
    x_train, y_train,
    epochs=100,  # 최대 100회 (EarlyStopping이 있으니 넉넉하게)
    batch_size=64,  # RTX 4070 Ti SUPER의 메모리를 고려해 배치 사이즈 확대
    validation_data=(x_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# 6. 모델 저장 및 결과 출력
model.save('heart_sound_model_v1.h5')
print("\n[학습 완료] heart_sound_model_v1.h5로 저장되었습니다.")

final_loss, final_acc = model.evaluate(x_val, y_val, verbose=0)
print(f"검증 데이터 최종 정확도: {final_acc * 100:.2f}%")