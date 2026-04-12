"""
심음 노이즈 제거 추론 스크립트 (TUNet-v2, TFLite)
사용법: python predict_denoise.py <wav_file> [output_file]

출력 파일 미지정 시 <원본명>_denoised.wav 로 저장
"""

import sys
import os
import numpy as np
import librosa
import soundfile as sf

# Python 3.12+ 에서 제거된 imp 모듈 호환 shim
import importlib
import sys
if "imp" not in sys.modules:
    import types
    imp_shim = types.ModuleType("imp")
    imp_shim.reload = importlib.reload
    imp_shim.find_module = lambda name, path=None: None
    imp_shim.load_module = lambda name: importlib.import_module(name)
    imp_shim.is_builtin = lambda name: name in sys.builtin_module_names
    imp_shim.is_frozen = lambda name: False
    sys.modules["imp"] = imp_shim

# TFLite 인터프리터
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
except ImportError:
    import tensorflow as tf
    TFLiteInterpreter = tf.lite.Interpreter

# ── 경로 설정 ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "tunet_v2_best.tflite")

# 학습 시 사용한 파라미터 (generate dataset.py 기준)
SR        = 4000
CHUNK_SEC = 5
CHUNK_LEN = SR * CHUNK_SEC   # 20000 samples


# ── 모델 로드 ──────────────────────────────────────────────
def load_model():
    interpreter = TFLiteInterpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


# ── 전처리: WAV → 5초 청크 ─────────────────────────────────
def wav_to_chunks(wav_path):
    y, _ = librosa.load(wav_path, sr=SR)

    # Peak normalize
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val * 0.9

    chunks  = []
    indices = []

    if len(y) <= CHUNK_LEN:
        pad = np.pad(y, (0, CHUNK_LEN - len(y)))
        chunks.append(pad)
        indices.append((0, len(y)))
    else:
        hop = CHUNK_LEN // 2
        for start in range(0, len(y) - CHUNK_LEN + 1, hop):
            chunks.append(y[start: start + CHUNK_LEN])
            indices.append((start, start + CHUNK_LEN))
        if indices[-1][1] < len(y):
            start = len(y) - CHUNK_LEN
            chunks.append(y[start:])
            indices.append((start, len(y)))

    return np.array(chunks, dtype=np.float32), indices, len(y)


# ── TFLite 추론 (1개 청크씩) ──────────────────────────────
def _run_tflite(interpreter, chunk):
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 입력: (1, chunk_len, 1)
    input_data = chunk.reshape(1, -1, 1).astype(np.float32)

    # 입력 크기가 모델과 다르면 resize
    if list(input_data.shape) != list(input_details[0]['shape']):
        interpreter.resize_tensor_input(input_details[0]['index'], list(input_data.shape))
        interpreter.allocate_tensors()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output.squeeze()   # (chunk_len,)


# ── 추론 및 청크 재조합 ────────────────────────────────────
def denoise(wav_path, interpreter):
    chunks, indices, orig_len = wav_to_chunks(wav_path)

    # 청크별 추론
    predictions = []
    for chunk in chunks:
        pred = _run_tflite(interpreter, chunk)
        predictions.append(pred)

    # 오버랩-Add 방식으로 재조합
    output  = np.zeros(orig_len, dtype=np.float32)
    weights = np.zeros(orig_len, dtype=np.float32)

    for i, (start, end) in enumerate(indices):
        seg_len = end - start
        output[start:end]  += predictions[i][:seg_len]
        weights[start:end] += 1.0

    weights = np.maximum(weights, 1.0)
    output  = output / weights

    return output, SR


# ── 메인 ───────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python predict_denoise.py <wav_file> [output_file]")
        sys.exit(1)

    wav_path = sys.argv[1]
    if not os.path.exists(wav_path):
        print(f"파일 없음: {wav_path}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    else:
        base  = os.path.splitext(wav_path)[0]
        out_path = base + "_denoised.wav"

    print("모델 로드 중...")
    interpreter = load_model()

    print(f"노이즈 제거 중: {wav_path}")
    denoised, sr = denoise(wav_path, interpreter)

    sf.write(out_path, denoised, sr)
    print(f"\n저장 완료: {out_path}")
    print(f"  샘플 수: {len(denoised)}  ({len(denoised)/sr:.2f}초)")
