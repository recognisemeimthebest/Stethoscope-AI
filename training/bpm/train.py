"""
Step 2: Noise Gate Model Training
===================================
Shannon Energy Envelope 기반 Tiny 1D CNN 학습.
조기 종료 + 최적 모델 저장 포함.

실행:
  C:/Users/dwd00/anaconda3/envs/stetho_ai/python.exe train.py
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import build_noise_gate, compile_model

# ─── 경로 ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── 하이퍼파라미터 ───────────────────────────────────────────────────────────
BATCH_SIZE    = 64
EPOCHS        = 100
LR_INITIAL    = 1e-3
LR_MIN        = 1e-5
PATIENCE      = 15       # 조기 종료 대기 에포크
RANDOM_SEED   = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_data():
    """전처리된 데이터 로드."""
    files = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"]
    for f in files:
        path = os.path.join(DATA_DIR, f)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"데이터 파일 없음: {path}\n"
                f"먼저 prepare_data.py를 실행하세요."
            )

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))

    print(f"학습 데이터:  {X_train.shape}, 레이블: {y_train.shape}")
    print(f"검증 데이터:  {X_val.shape}, 레이블: {y_val.shape}")
    print(f"양성 비율:    학습={y_train.mean():.2%}, 검증={y_val.mean():.2%}")

    return X_train, y_train, X_val, y_val


def plot_history(history, save_path):
    """학습 곡선 저장."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["accuracy"],     label="Train Acc")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"학습 곡선 저장: {save_path}")


def main():
    print("=" * 60)
    print("온디바이스 BPM 노이즈 게이트 모델 학습")
    print("=" * 60)

    # 데이터 로드
    print("\n[1/4] 데이터 로드...")
    X_train, y_train, X_val, y_val = load_data()

    # 모델 생성
    print("\n[2/4] 모델 생성...")
    model = build_noise_gate(input_length=X_train.shape[1])
    compile_model(model, learning_rate=LR_INITIAL)
    model.summary()

    # 콜백
    checkpoint_path = os.path.join(MODEL_DIR, "noise_gate_best.keras")
    callbacks = [
        # 최적 검증 정확도 모델 저장
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        # 조기 종료
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        # 학습률 감소 (5 에포크 개선 없으면 ×0.5)
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=LR_MIN,
            verbose=1
        ),
        # 텐서보드 (선택사항)
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(MODEL_DIR, "logs"),
            histogram_freq=0
        )
    ]

    # 학습
    print(f"\n[3/4] 학습 시작 (최대 {EPOCHS} 에포크, 조기종료 patience={PATIENCE})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # 평가
    print("\n[4/4] 최종 평가...")
    best_model = tf.keras.models.load_model(checkpoint_path)
    val_loss, val_acc, val_auc = best_model.evaluate(X_val, y_val, verbose=0)
    print(f"\n검증 결과:")
    print(f"  Loss:     {val_loss:.4f}")
    print(f"  Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")
    print(f"  AUC:      {val_auc:.4f}")

    # 혼동 행렬
    y_pred = (best_model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
    tp = int(((y_pred == 1) & (y_val == 1)).sum())
    tn = int(((y_pred == 0) & (y_val == 0)).sum())
    fp = int(((y_pred == 1) & (y_val == 0)).sum())
    fn = int(((y_pred == 0) & (y_val == 1)).sum())
    print(f"\n혼동 행렬:")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  Precision: {tp/(tp+fp+1e-10):.3f}")
    print(f"  Recall:    {tp/(tp+fn+1e-10):.3f}")

    # 학습 곡선 저장
    plot_history(history, os.path.join(MODEL_DIR, "training_curve.png"))

    # 최종 모델 저장 (TFLite 변환용)
    final_path = os.path.join(MODEL_DIR, "noise_gate_final.keras")
    best_model.save(final_path)
    print(f"\n모델 저장: {final_path}")
    print(f"다음 단계: python quantize.py")


if __name__ == "__main__":
    main()
