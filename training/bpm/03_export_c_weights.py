"""
PyTorch 모델 weight -> C 헤더 내보내기
======================================
TFLite 변환 대신, 학습된 PyTorch weight를 C 배열로 직접 내보내고
ESP32에서 순수 C로 forward pass 구현.

모델이 Conv1D 3층 + BN 3층 + Dense 1층으로 아주 작아서
TFLite Micro 런타임 (~100KB) 없이 순수 C로 ~2KB 코드로 추론 가능.

BN은 Conv에 fuse해서 Conv weight/bias로 합침 -> 추론 시 BN 불필요.

사용법:
  C:/Users/dwd00/anaconda3/envs/stetho_ai/python.exe 03_export_c_weights.py
"""

import os
import numpy as np
import torch

OUTPUT_DIR = "G:/stetho_ai/BPM_ondevice/models"
HEADER_PATH = os.path.join(OUTPUT_DIR, "beat_cnn_weights.h")
IMPL_PATH = os.path.join(OUTPUT_DIR, "beat_cnn_inference.h")


def fuse_conv_bn(conv_w, bn_gamma, bn_beta, bn_mean, bn_var, eps=1e-5):
    """
    Conv weight + BatchNorm -> fused Conv weight + bias.
    Conv has no bias, BN absorbs into weight and creates bias.

    conv_w: (out_ch, in_ch, kernel)
    Returns: fused_w (same shape), fused_b (out_ch,)
    """
    # BN scale factor per output channel
    std = np.sqrt(bn_var + eps)
    scale = bn_gamma / std  # (out_ch,)

    # Fuse into conv weight
    # w_fused[o, i, k] = w[o, i, k] * scale[o]
    fused_w = conv_w * scale[:, None, None]

    # Fuse bias: b_fused[o] = -mean[o] * scale[o] + beta[o]
    fused_b = -bn_mean * scale + bn_beta

    return fused_w, fused_b


def export_array(f, name, arr, dtype="float"):
    """C 배열로 내보내기."""
    flat = arr.flatten()
    f.write(f"// shape: {list(arr.shape)}, total: {len(flat)}\n")
    f.write(f"static const {dtype} {name}[] = {{\n")
    for i in range(0, len(flat), 8):
        chunk = flat[i:i+8]
        vals = ", ".join(f"{v:.8f}f" for v in chunk)
        f.write(f"    {vals},\n")
    f.write("};\n\n")


def main():
    # Load PyTorch model
    import sys
    sys.path.insert(0, "G:/stetho_ai/BPM_ondevice")
    from importlib.machinery import SourceFileLoader
    mod = SourceFileLoader("trainer", "G:/stetho_ai/BPM_ondevice/02_train_beat_detector.py").load_module()

    model = mod.TinyBeatCNN()
    model.load_state_dict(
        torch.load(os.path.join(OUTPUT_DIR, "beat_cnn_best.pth"),
                    map_location="cpu", weights_only=True)
    )
    model.eval()
    sd = {k: v.numpy() for k, v in model.state_dict().items()}

    # Print model info
    total_params = sum(v.size for v in sd.values())
    print(f"Total parameters: {total_params:,}")

    # Fuse Conv+BN for each block
    blocks = [
        ("features.0", "features.1"),  # Conv1, BN1
        ("features.3", "features.4"),  # Conv2, BN2
        ("features.6", "features.7"),  # Conv3, BN3
    ]

    fused_weights = []
    for conv_prefix, bn_prefix in blocks:
        conv_w = sd[f"{conv_prefix}.weight"]  # (out, in, k)
        bn_g = sd[f"{bn_prefix}.weight"]
        bn_b = sd[f"{bn_prefix}.bias"]
        bn_m = sd[f"{bn_prefix}.running_mean"]
        bn_v = sd[f"{bn_prefix}.running_var"]

        fw, fb = fuse_conv_bn(conv_w, bn_g, bn_b, bn_m, bn_v)
        fused_weights.append((fw, fb))
        print(f"  {conv_prefix}+{bn_prefix}: w{fw.shape} b{fb.shape}")

    # Dense layer
    dense_w = sd["classifier.2.weight"]  # (1, 32)
    dense_b = sd["classifier.2.bias"]    # (1,)
    print(f"  dense: w{dense_w.shape} b{dense_b.shape}")

    # Verify fused model matches original
    test_input = torch.randn(1, 1, 400)
    with torch.no_grad():
        original_out = model(test_input).item()

    # Manual forward pass with fused weights
    x = test_input.numpy().squeeze(0)  # (1, 400)

    for i, (fw, fb) in enumerate(fused_weights):
        out_ch, in_ch, k = fw.shape
        stride = 2
        pad = [3, 2, 1][i]

        # Zero padding
        x_padded = np.pad(x, ((0, 0), (pad, pad)), mode='constant')
        in_len = x_padded.shape[1]
        out_len = (in_len - k) // stride + 1

        y = np.zeros((out_ch, out_len))
        for oc in range(out_ch):
            for ic in range(in_ch):
                for j in range(out_len):
                    start = j * stride
                    y[oc, j] += np.sum(x_padded[ic, start:start+k] * fw[oc, ic, :])
            y[oc, :] += fb[oc]

        # ReLU
        y = np.maximum(y, 0)
        x = y

    # Global average pooling
    gap = np.mean(x, axis=1)  # (32,)

    # Dense
    logit = np.dot(dense_w.flatten(), gap) + dense_b[0]

    print(f"\n  Original PyTorch output: {original_out:.6f}")
    print(f"  Fused manual output:    {logit:.6f}")
    print(f"  Diff: {abs(original_out - logit):.8f}")

    # Export C header with weights
    print(f"\nExporting C header: {HEADER_PATH}")
    with open(HEADER_PATH, 'w') as f:
        f.write("// Auto-generated: TinyBeatCNN fused weights\n")
        f.write("// Conv+BN fused into Conv(weight,bias) + ReLU\n")
        f.write("// Model: 6017 params, ~24KB float32\n")
        f.write("#pragma once\n\n")

        for i, (fw, fb) in enumerate(fused_weights):
            export_array(f, f"conv{i+1}_w", fw)
            export_array(f, f"conv{i+1}_b", fb)

        export_array(f, "dense_w", dense_w)
        export_array(f, "dense_b", dense_b)

        # Model config
        f.write("// Model config\n")
        f.write(f"#define BEAT_CNN_INPUT_LEN 400\n")
        f.write(f"#define BEAT_CNN_CONV1_OUT 16\n")
        f.write(f"#define BEAT_CNN_CONV1_K 7\n")
        f.write(f"#define BEAT_CNN_CONV1_PAD 3\n")
        f.write(f"#define BEAT_CNN_CONV2_OUT 32\n")
        f.write(f"#define BEAT_CNN_CONV2_K 5\n")
        f.write(f"#define BEAT_CNN_CONV2_PAD 2\n")
        f.write(f"#define BEAT_CNN_CONV3_OUT 32\n")
        f.write(f"#define BEAT_CNN_CONV3_K 3\n")
        f.write(f"#define BEAT_CNN_CONV3_PAD 1\n")
        f.write(f"#define BEAT_CNN_STRIDE 2\n")

    size_kb = os.path.getsize(HEADER_PATH) / 1024
    print(f"  Weight header: {size_kb:.1f} KB")

    # Export inference implementation
    print(f"Exporting inference code: {IMPL_PATH}")
    with open(IMPL_PATH, 'w') as f:
        f.write("""// Auto-generated: TinyBeatCNN pure-C inference
// No TFLite dependency. ~2KB code + 24KB weights.
#pragma once
#include "beat_cnn_weights.h"
#include <math.h>
#include <string.h>

// Scratch buffers (statically allocated)
static float _buf_a[32 * 210];  // max intermediate size
static float _buf_b[32 * 210];

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Conv1D + bias + ReLU, stride=2
// in: (in_ch, in_len), out: (out_ch, out_len)
static void conv1d_relu(
    const float* input, int in_ch, int in_len,
    const float* weight, const float* bias,
    int out_ch, int kernel, int pad,
    float* output, int* out_len_ptr)
{
    int padded_len = in_len + 2 * pad;
    int out_len = (padded_len - kernel) / BEAT_CNN_STRIDE + 1;
    *out_len_ptr = out_len;

    for (int oc = 0; oc < out_ch; oc++) {
        for (int j = 0; j < out_len; j++) {
            float sum = bias[oc];
            int start = j * BEAT_CNN_STRIDE - pad;
            for (int ic = 0; ic < in_ch; ic++) {
                for (int k = 0; k < kernel; k++) {
                    int idx = start + k;
                    float val = (idx >= 0 && idx < in_len) ? input[ic * in_len + idx] : 0.0f;
                    sum += val * weight[(oc * in_ch + ic) * kernel + k];
                }
            }
            // ReLU
            output[oc * out_len + j] = (sum > 0.0f) ? sum : 0.0f;
        }
    }
}

/**
 * Run beat detection inference.
 * @param audio_window  200ms of audio, 400 samples (raw waveform, float32)
 * @return probability of beat [0.0, 1.0]
 */
static float beat_cnn_predict(const float* audio_window)
{
    int len;

    // Block 1: (1, 400) -> (16, 200)
    conv1d_relu(audio_window, 1, BEAT_CNN_INPUT_LEN,
                conv1_w, conv1_b,
                BEAT_CNN_CONV1_OUT, BEAT_CNN_CONV1_K, BEAT_CNN_CONV1_PAD,
                _buf_a, &len);

    // Block 2: (16, 200) -> (32, 100)
    conv1d_relu(_buf_a, BEAT_CNN_CONV1_OUT, len,
                conv2_w, conv2_b,
                BEAT_CNN_CONV2_OUT, BEAT_CNN_CONV2_K, BEAT_CNN_CONV2_PAD,
                _buf_b, &len);

    // Block 3: (32, 100) -> (32, 50)
    conv1d_relu(_buf_b, BEAT_CNN_CONV2_OUT, len,
                conv3_w, conv3_b,
                BEAT_CNN_CONV3_OUT, BEAT_CNN_CONV3_K, BEAT_CNN_CONV3_PAD,
                _buf_a, &len);

    // Global Average Pooling: (32, 50) -> (32,)
    float gap[32];
    for (int ch = 0; ch < 32; ch++) {
        float sum = 0.0f;
        for (int j = 0; j < len; j++) {
            sum += _buf_a[ch * len + j];
        }
        gap[ch] = sum / (float)len;
    }

    // Dense: (32,) -> (1,)
    float logit = dense_b[0];
    for (int i = 0; i < 32; i++) {
        logit += dense_w[i] * gap[i];
    }

    return sigmoid(logit);
}
""")

    print(f"  Inference code: {os.path.getsize(IMPL_PATH) / 1024:.1f} KB")
    print(f"\n  ESP32 usage:")
    print(f"    #include \"beat_cnn_inference.h\"")
    print(f"    float prob = beat_cnn_predict(audio_400_samples);")
    print(f"    bool is_beat = (prob > 0.5f);")


if __name__ == "__main__":
    main()
