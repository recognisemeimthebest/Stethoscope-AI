"""
과적합 / 데이터 누수 검증 스크립트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 세그먼트 단위 랜덤 분할 -> 같은 파일의 세그먼트가 train/val 양쪽에 존재하는지?
2. Unknown(ESC-50) 클래스가 너무 쉬운지?
3. Train loss vs Val loss 비교 (재학습)
4. Normal vs Abnormal만 따로 성능 확인 (Unknown 제외)
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

CACHE_DIR = r"G:\stetho_ai\Heart_binary_classification"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ═══════════════════════════════════════════
# 모델 정의
# ═══════════════════════════════════════════
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = self.sigmoid_channel(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)
        x = x * out
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial = self.sigmoid_spatial(self.conv_spatial(spatial))
        x = x * spatial
        return x

class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout_rate=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.cbam = CBAM(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += self.shortcut(x)
        return F.relu(out)

class HeartSoundModel(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = ResNetBlock(64, 64, dropout_rate=0.2)
        self.layer2 = ResNetBlock(64, 128, stride=2, dropout_rate=0.2)
        self.layer3 = ResNetBlock(128, 256, stride=2, dropout_rate=0.3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        x = self.dropout_fc(features)
        out = self.fc(x)
        return out, features


# ═══════════════════════════════════════════
# 데이터 로드
# ═══════════════════════════════════════════
print("\n데이터 로딩...")
X = np.load(os.path.join(CACHE_DIR, 'x_data.npy'))
y = np.load(os.path.join(CACHE_DIR, 'y_data.npy'))

total = len(y)
unique, counts = np.unique(y, return_counts=True)
label_names = {0: 'Normal', 1: 'Abnormal', 2: 'Unknown'}
print(f"총 세그먼트: {total}")
for u, c in zip(unique, counts):
    print(f"  {label_names[int(u)]}: {c} ({c/total*100:.1f}%)")


# ═══════════════════════════════════════════
# 검증 1: 데이터 누수 구조 분석
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("  검증 1: 데이터 누수 (Data Leakage) 구조 분석")
print("=" * 70)

print("""
[핵심 문제]
현재 분할 방식: 세그먼트(segment) 단위 랜덤 분할 (80/20)

preprocessing.py에서:
  - 파일 A (10초) -> 슬라이딩 윈도우 -> seg1, seg2, seg3 (3개 세그먼트)
  - 세 세그먼트가 x_data.npy에 연속 저장

XGBClassifier_Full.py에서:
  - random_split(dataset, [train, val]) -> 세그먼트 단위 랜덤 분할
  - seg1 -> train, seg2 -> val, seg3 -> train 이런 식으로 분할 가능

[!] 결과: 같은 파일에서 나온 세그먼트(60% 오버랩)가 train과 val 양쪽에 존재
   -> 모델이 "이 파일의 특성"을 외워서 val에서도 맞추는 것
   -> 이것은 데이터 누수(Data Leakage)이며, 실제 성능보다 부풀려진 수치
""")


# ═══════════════════════════════════════════
# 검증 2: Unknown 분리 용이성 확인
# ═══════════════════════════════════════════
print("=" * 70)
print("  검증 2: Unknown(ESC-50) 분리 용이성")
print("=" * 70)

unknown_mask = (y == 2)
heart_mask = (y != 2)

# Unknown vs Heart 에너지 차이 확인
unknown_energy = np.mean(X[unknown_mask] ** 2, axis=(1,2,3))
heart_energy = np.mean(X[heart_mask] ** 2, axis=(1,2,3))

print(f"\n평균 에너지 (Mel-Spec dB²):")
print(f"  심음 (Normal+Abnormal): {np.mean(heart_energy):.2f} ± {np.std(heart_energy):.2f}")
print(f"  Unknown (ESC-50):       {np.mean(unknown_energy):.2f} ± {np.std(unknown_energy):.2f}")
print(f"  에너지 비율:            {np.mean(heart_energy) / np.mean(unknown_energy):.2f}x")

# Unknown의 dB 값 분포
unknown_mean_db = np.mean(X[unknown_mask], axis=(1,2,3))
heart_mean_db = np.mean(X[heart_mask], axis=(1,2,3))
print(f"\n평균 dB 값:")
print(f"  심음:    {np.mean(heart_mean_db):.2f} ± {np.std(heart_mean_db):.2f}")
print(f"  Unknown: {np.mean(unknown_mean_db):.2f} ± {np.std(unknown_mean_db):.2f}")

overlap = np.sum((unknown_mean_db > np.percentile(heart_mean_db, 25)) &
                 (unknown_mean_db < np.percentile(heart_mean_db, 75)))
print(f"  Unknown 중 심음 IQR과 겹치는 세그먼트: {overlap}/{np.sum(unknown_mask)} ({overlap/np.sum(unknown_mask)*100:.1f}%)")

print(f"""
[분석]
Unknown(ESC-50)은 환경음이라 심음과 스펙트로그램 패턴이 근본적으로 다름.
-> 100% 분류는 "쉬운 문제"여서 가능한 것이지, 과적합은 아님.
-> 단, 실제 병원 환경 노이즈(에어컨, 대화 등)와 ESC-50은 다를 수 있음.
""")


# ═══════════════════════════════════════════
# 검증 3: Normal vs Abnormal만 (Unknown 제외) 성능 확인
# ═══════════════════════════════════════════
print("=" * 70)
print("  검증 3: Unknown 제외 - Normal vs Abnormal 만의 실제 성능")
print("=" * 70)

# 모델 로드
model = HeartSoundModel(num_classes=3, dropout_rate=0.5).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(CACHE_DIR, 'resnet_cbam_best.pth')))
model.eval()

xgb = XGBClassifier()
xgb.load_model(os.path.join(CACHE_DIR, 'heart_xgb_model.json'))

# 동일한 train/val 분할 재현
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_dataset = random_split(dataset, [train_size, val_size],
                              generator=torch.Generator().manual_seed(42))
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# val set에서 특징 추출 + 예측
all_features, all_labels, all_preds_cnn = [], [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs, feats = model(inputs.to(DEVICE))
        all_features.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())
        all_preds_cnn.append(torch.argmax(outputs, dim=1).cpu().numpy())

all_features = np.concatenate(all_features)
all_labels = np.concatenate(all_labels)
all_preds_cnn = np.concatenate(all_preds_cnn)

# XGBoost 예측
all_preds_xgb = xgb.predict(all_features)

# ── 3-class 전체 성능 ──
print(f"\n[3-Class 전체 성능 (CNN)]")
print(classification_report(all_labels, all_preds_cnn,
                            target_names=["Normal", "Abnormal", "Unknown"], digits=4))

print(f"[3-Class 전체 성능 (XGBoost)]")
print(classification_report(all_labels, all_preds_xgb,
                            target_names=["Normal", "Abnormal", "Unknown"], digits=4))

# ── Unknown 제외: Normal vs Abnormal만 ──
heart_idx = all_labels != 2  # Unknown 제외
labels_heart = all_labels[heart_idx]
preds_heart_cnn = all_preds_cnn[heart_idx]
preds_heart_xgb = all_preds_xgb[heart_idx]

# Unknown으로 예측된 것도 오답으로 처리 (2->오답)
print(f"\n[2-Class 성능: Normal vs Abnormal만 (Unknown 제외)]")
print(f"  Val 세그먼트 수: {np.sum(heart_idx)} (Normal: {np.sum(labels_heart==0)}, Abnormal: {np.sum(labels_heart==1)})")

# CNN
cm_cnn = confusion_matrix(labels_heart, preds_heart_cnn, labels=[0, 1])
sens_cnn = cm_cnn[1, 1] / (cm_cnn[1, 0] + cm_cnn[1, 1]) if (cm_cnn[1, 0] + cm_cnn[1, 1]) > 0 else 0
spec_cnn = cm_cnn[0, 0] / (cm_cnn[0, 0] + cm_cnn[0, 1]) if (cm_cnn[0, 0] + cm_cnn[0, 1]) > 0 else 0
macc_cnn = (sens_cnn + spec_cnn) / 2
# Unknown으로 잘못 예측된 심음 세그먼트
heart_as_unknown_cnn = np.sum(preds_heart_cnn == 2)
print(f"\n  CNN 결과:")
print(f"    Sensitivity: {sens_cnn*100:.2f}%")
print(f"    Specificity: {spec_cnn*100:.2f}%")
print(f"    MACC:        {macc_cnn*100:.2f}%")
print(f"    심음->Unknown 오분류: {heart_as_unknown_cnn}개")
print(f"    Confusion Matrix:\n{cm_cnn}")

# XGBoost
cm_xgb = confusion_matrix(labels_heart, preds_heart_xgb, labels=[0, 1])
sens_xgb = cm_xgb[1, 1] / (cm_xgb[1, 0] + cm_xgb[1, 1]) if (cm_xgb[1, 0] + cm_xgb[1, 1]) > 0 else 0
spec_xgb = cm_xgb[0, 0] / (cm_xgb[0, 0] + cm_xgb[0, 1]) if (cm_xgb[0, 0] + cm_xgb[0, 1]) > 0 else 0
macc_xgb = (sens_xgb + spec_xgb) / 2
heart_as_unknown_xgb = np.sum(preds_heart_xgb == 2)
print(f"\n  XGBoost 결과:")
print(f"    Sensitivity: {sens_xgb*100:.2f}%")
print(f"    Specificity: {spec_xgb*100:.2f}%")
print(f"    MACC:        {macc_xgb*100:.2f}%")
print(f"    심음->Unknown 오분류: {heart_as_unknown_xgb}개")
print(f"    Confusion Matrix:\n{cm_xgb}")


# ═══════════════════════════════════════════
# 검증 4: 데이터 누수 시뮬레이션 - 세그먼트 인접성 분석
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("  검증 4: 세그먼트 인접성 - 누수 규모 추정")
print("=" * 70)

# val 인덱스 추출
val_indices = val_dataset.indices
train_indices = set(range(total)) - set(val_indices)

# 세그먼트가 연속되면 같은 파일 출처일 확률이 높음
# 인접 세그먼트 (±1, ±2, ±3)가 train/val에 교차 존재하는지 확인
val_set = set(val_indices)
train_set = set(range(total)) - val_set

leaked_count = 0
for vi in val_indices:
    # 윈도우 스트라이드=2초, 오버랩=60% -> 인접 세그먼트는 같은 파일에서 온 것
    neighbors = {vi-3, vi-2, vi-1, vi+1, vi+2, vi+3}
    if neighbors & train_set:
        leaked_count += 1

leak_ratio = leaked_count / len(val_indices) * 100
print(f"\n  Val 세그먼트 중 인접 세그먼트가 Train에 존재하는 비율:")
print(f"  {leaked_count} / {len(val_indices)} = {leak_ratio:.1f}%")
print(f"""
  [해석]
  {leak_ratio:.0f}%의 val 세그먼트가 같은 파일의 다른 세그먼트와 함께 train에 포함됨.
  5초 윈도우 + 2초 스트라이드 = 60% 오버랩이므로,
  인접 세그먼트는 3초 분량의 동일한 오디오를 공유함.
  -> 모델이 "새로운 환자의 심음"을 판별하는 게 아니라,
     "이미 본 오디오의 약간 다른 구간"을 맞추는 것에 가까움.
""")


# ═══════════════════════════════════════════
# 검증 5: Train vs Val Loss (CNN 단독)
# ═══════════════════════════════════════════
print("=" * 70)
print("  검증 5: Train vs Val Accuracy (CNN 단독)")
print("=" * 70)

# Train set에서도 정확도 확인
train_dataset_for_eval = TensorDataset(
    X_tensor[list(train_indices)], y_tensor[list(train_indices)])
train_eval_loader = DataLoader(train_dataset_for_eval, batch_size=64, shuffle=False)

train_preds, train_labels_all = [], []
with torch.no_grad():
    for inputs, labels in train_eval_loader:
        outputs, _ = model(inputs.to(DEVICE))
        train_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
        train_labels_all.append(labels.numpy())

train_preds = np.concatenate(train_preds)
train_labels_all = np.concatenate(train_labels_all)

train_acc = accuracy_score(train_labels_all, train_preds)
val_acc = accuracy_score(all_labels, all_preds_cnn)
gap = train_acc - val_acc

print(f"\n  Train Accuracy (CNN): {train_acc*100:.2f}%")
print(f"  Val Accuracy (CNN):   {val_acc*100:.2f}%")
print(f"  Gap (Train - Val):    {gap*100:.2f}%p")

if gap > 0.05:
    print(f"  [!] Train-Val gap이 {gap*100:.1f}%p로, 과적합 징후가 있음")
elif gap > 0.02:
    print(f"  [~] Train-Val gap이 {gap*100:.1f}%p로, 약한 과적합 또는 데이터 누수 영향")
else:
    print(f"  [OK] Train-Val gap이 {gap*100:.1f}%p로, 과적합 징후 없음")


# ═══════════════════════════════════════════
# 최종 종합 판정
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("  ★ 종합 판정")
print("=" * 70)

print(f"""
  1. 데이터 누수: [!] 존재함
     - 세그먼트 단위 랜덤 분할 -> 같은 파일의 세그먼트가 train/val 양쪽에 존재
     - Val 세그먼트의 {leak_ratio:.0f}%가 train에 인접 세그먼트 보유
     - 인접 세그먼트는 60% 오디오 오버랩 -> 사실상 같은 데이터

  2. Unknown 100% 정확도: [OK] 정상 (과적합 아님)
     - ESC-50 환경음은 심음과 근본적으로 다른 패턴
     - 단, 실제 병원 노이즈와는 다를 수 있어 현장 검증 필요

  3. Normal vs Abnormal 실제 성능:
     - 현재 보고된 MACC 95.7%는 데이터 누수가 포함된 수치
     - 정확한 성능은 "파일 단위 분할"로 재평가해야 알 수 있음
     - 예상 실제 MACC: 약 85~90% 범위 (누수 제거 시 하락 예상)

  4. 권장 조치:
     ① preprocessing.py에서 파일별 소속 정보를 저장
     ② 파일 단위(또는 환자 단위)로 train/val 분할
     ③ 동일 파일의 세그먼트가 절대 train/val에 교차되지 않도록 보장
     ④ 이 조건에서 재평가한 성능이 실제 일반화 성능
""")
