"""
심음 분류 추론 스크립트 (ResNet+CBAM + XGBoost)
사용법: python predict_heart.py <wav_file>
"""

import sys
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from xgboost import XGBClassifier

# ── 경로 설정 ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Heart_binary_classification", "resnet_cbam_best.pth")
XGB_PATH   = os.path.join(BASE_DIR, "Heart_binary_classification", "heart_xgb_model.json")

DEVICE = torch.device("cpu")

# Unknown 판정 임계값: Unknown 확률이 이 값 이상이면 argmax 무시하고 Unknown 처리
UNKNOWN_THRESHOLD = 30.0

# 전처리 파라미터
SR               = 4000
WINDOW_DURATION  = 5
SAMPLES_PER_WIN  = SR * WINDOW_DURATION   # 20000
HOP_LENGTH       = 256
N_MELS           = 64
STRIDE_SEC       = 2                       # 슬라이딩 스트라이드
FIXED_REF        = 1.0                     # dB 변환 고정 ref (학습과 동일)


# ── 모델 정의 ──────────────────────────────────────────────
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial   = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x).view(x.size(0), -1))
        mx  = self.fc(self.max_pool(x).view(x.size(0), -1))
        ch  = self.sigmoid_channel(avg + mx).view(x.size(0), x.size(1), 1, 1)
        x   = x * ch
        sp  = torch.cat([torch.mean(x, 1, keepdim=True),
                         torch.max(x, 1, keepdim=True)[0]], dim=1)
        x   = x * self.sigmoid_spatial(self.conv_spatial(sp))
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.2):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_ch)
        self.drop1    = nn.Dropout2d(dropout)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_ch)
        self.cbam     = CBAM(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop1(out)
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += self.shortcut(x)
        return F.relu(out)


class HeartSoundModel(nn.Module):
    def __init__(self, num_classes=3, dropout=0.5):
        super().__init__()
        self.conv1    = nn.Conv2d(1, 64, 3, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = ResNetBlock(64,  64,  dropout=0.2)
        self.layer2   = ResNetBlock(64,  128, stride=2, dropout=0.2)
        self.layer3   = ResNetBlock(128, 256, stride=2, dropout=0.3)
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.drop_fc  = nn.Dropout(dropout)
        self.fc       = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        out  = self.fc(self.drop_fc(feat))
        return out, feat


# ── 모델 로드 ──────────────────────────────────────────────
def load_models():
    resnet = HeartSoundModel(num_classes=3, dropout=0.5).to(DEVICE)
    resnet.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    resnet.eval()

    xgb = XGBClassifier()
    xgb.load_model(XGB_PATH)
    return resnet, xgb


# ── 전처리 ─────────────────────────────────────────────────
def wav_to_mel_segments(wav_path):
    y, _ = librosa.load(wav_path, sr=SR)
    # 피크 정규화 — 녹음 환경(BLE/직접녹음)에 따른 음량 차이 제거
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    segments = []
    if len(y) < SAMPLES_PER_WIN:
        segments.append(np.pad(y, (0, SAMPLES_PER_WIN - len(y))))
    else:
        for start in range(0, len(y) - SAMPLES_PER_WIN + 1, SR * STRIDE_SEC):
            segments.append(y[start: start + SAMPLES_PER_WIN])
    mels = [
        librosa.power_to_db(
            librosa.feature.melspectrogram(y=s, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH),
            ref=FIXED_REF,
        )
        for s in segments
    ]
    return np.array(mels)   # (N, n_mels, time)


# ── 추론 ───────────────────────────────────────────────────
def predict(wav_path, resnet, xgb):
    mels = wav_to_mel_segments(wav_path)
    mels_t = torch.FloatTensor(mels).unsqueeze(1).to(DEVICE)   # (N,1,mel,time)

    with torch.no_grad():
        _, features = resnet(mels_t)
    features_np = features.cpu().numpy()

    probs     = xgb.predict_proba(features_np)   # (N, 3)
    avg_probs = probs.mean(axis=0)

    prob_normal   = avg_probs[0] * 100
    prob_abnormal = avg_probs[1] * 100
    prob_unknown  = avg_probs[2] * 100

    # Unknown 우선 판정: prob_unknown >= threshold면 argmax 무시
    if prob_unknown >= UNKNOWN_THRESHOLD:
        label = "Unknown"
    else:
        # Unknown 제외하고 Normal vs Abnormal만 비교
        top_idx = int(np.argmax(avg_probs[:2]))
        label = {0: "Normal", 1: "Abnormal"}[top_idx]

    return {
        "label":         label,
        "prob_normal":   round(prob_normal,   2),
        "prob_abnormal": round(prob_abnormal, 2),
        "prob_unknown":  round(prob_unknown,  2),
        "segments":      len(mels),
    }


# ── 메인 ───────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python predict_heart.py <wav_file>")
        sys.exit(1)

    wav_path = sys.argv[1]
    if not os.path.exists(wav_path):
        print(f"파일 없음: {wav_path}")
        sys.exit(1)

    print("모델 로드 중...")
    resnet, xgb = load_models()

    print(f"추론 중: {wav_path}")
    result = predict(wav_path, resnet, xgb)

    print("\n===== 심음 분류 결과 =====")
    print(f"  판정:         {result['label']}")
    print(f"  정상 확률:    {result['prob_normal']:.2f}%")
    print(f"  비정상 확률:  {result['prob_abnormal']:.2f}%")
    print(f"  Unknown 확률: {result['prob_unknown']:.2f}%")
    print(f"  분석 구간:    {result['segments']}개")
    if result['label'] == "Unknown":
        print("  ⚠️ 심음으로 인식되지 않음 — 다시 청진해주세요.")
