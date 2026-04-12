"""
LU-Net v2 (LSTM) vs TU-Net v2 (TCN) 추론 속도 벤치마크
"""
import os
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

# GPU 비활성화 (CPU 기준 공정 비교)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

MODEL_DIR = r"G:\stetho_ai\_misc\datasets\LUNet_Dataset"
CHUNK = 20000  # 5초 @ 4kHz

# ============================================================
# Custom loss functions (모델 로드용)
# ============================================================
def si_snr_loss(y_true, y_pred):
    y_true = y_true - K.mean(y_true, axis=1, keepdims=True)
    y_pred = y_pred - K.mean(y_pred, axis=1, keepdims=True)
    dot = K.sum(y_true * y_pred, axis=1, keepdims=True)
    s_target_energy = K.sum(y_true ** 2, axis=1, keepdims=True) + 1e-8
    proj = dot * y_true / s_target_energy
    noise = y_pred - proj
    si_snr = 10 * K.log(K.sum(proj ** 2, axis=1) / (K.sum(noise ** 2, axis=1) + 1e-8)) / K.log(10.0)
    return -K.mean(si_snr)

def combined_loss(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))
    si = si_snr_loss(y_true, y_pred)
    return 0.9 * mse + 0.1 * (si / 20.0)

custom_objects = {
    'si_snr_loss': si_snr_loss,
    'combined_loss': combined_loss,
}

# ============================================================
# 모델 로드
# ============================================================
print("Loading models...")
lunet_v2 = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "lunet_v2_90_10.h5"),
    custom_objects=custom_objects
)
tunet_v2 = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "tunet_v2_best.h5"),
    custom_objects=custom_objects
)
print(f"LU-Net v2 params: {lunet_v2.count_params():,}")
print(f"TU-Net v2 params: {tunet_v2.count_params():,}")

# ============================================================
# 더미 입력 생성
# ============================================================
dummy = np.random.randn(1, CHUNK, 1).astype(np.float32)

# 워밍업 (JIT 컴파일 등)
for _ in range(3):
    lunet_v2.predict(dummy, verbose=0)
    tunet_v2.predict(dummy, verbose=0)

# ============================================================
# 벤치마크
# ============================================================
N_RUNS = 50

# LU-Net v2 (LSTM)
times_lu = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    lunet_v2.predict(dummy, verbose=0)
    times_lu.append(time.perf_counter() - t0)

# TU-Net v2 (TCN)
times_tu = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    tunet_v2.predict(dummy, verbose=0)
    times_tu.append(time.perf_counter() - t0)

# 배치 벤치마크 (10개 동시)
dummy_batch = np.random.randn(10, CHUNK, 1).astype(np.float32)

times_lu_batch = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    lunet_v2.predict(dummy_batch, verbose=0)
    times_lu_batch.append(time.perf_counter() - t0)

times_tu_batch = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    tunet_v2.predict(dummy_batch, verbose=0)
    times_tu_batch.append(time.perf_counter() - t0)

# ============================================================
# 결과 출력
# ============================================================
lu_avg = np.mean(times_lu) * 1000
lu_std = np.std(times_lu) * 1000
tu_avg = np.mean(times_tu) * 1000
tu_std = np.std(times_tu) * 1000

lu_b_avg = np.mean(times_lu_batch) * 1000
tu_b_avg = np.mean(times_tu_batch) * 1000

print("\n" + "=" * 60)
print("  추론 속도 벤치마크 결과 (CPU, 5초 오디오 1개)")
print("=" * 60)
print(f"  LU-Net v2 (LSTM): {lu_avg:.1f} ± {lu_std:.1f} ms")
print(f"  TU-Net v2 (TCN):  {tu_avg:.1f} ± {tu_std:.1f} ms")
print(f"  속도 비율:         {lu_avg / tu_avg:.2f}x (TCN이 {lu_avg / tu_avg:.1f}배 빠름)")
print()
print(f"  배치 10개 기준:")
print(f"  LU-Net v2 (LSTM): {lu_b_avg:.1f} ms")
print(f"  TU-Net v2 (TCN):  {tu_b_avg:.1f} ms")
print(f"  속도 비율:         {lu_b_avg / tu_b_avg:.2f}x")
print("=" * 60)
