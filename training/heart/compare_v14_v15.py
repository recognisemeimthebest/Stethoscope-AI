"""
V14 vs V15 성능 비교 + 그래프 생성
- V15: 자체 validation set (20% holdout)에서 3-class 평가
- V14: cached validation features로 2-class 평가
- 공통 외부 테스트: sample_sound 등 실제 파일로 추론 비교
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score, recall_score)
from xgboost import XGBClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 색상 ──
BG   = '#0F172A'; CARD = '#1A253C'
CYAN = '#00D2FF'; PURPLE = '#7C3AED'; GREEN = '#10B981'
RED  = '#EF4444'; ORANGE = '#F59E0B'; GRAY = '#94A3B8'
WHITE = '#E2E8F0'; LIGHT = '#CBD5E1'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

OUT_DIR = r"G:\stetho_ai\Heart_binary_classification\figures_v15"
os.makedirs(OUT_DIR, exist_ok=True)

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(BG)
    ax.set_title(title, color=WHITE, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel, color=LIGHT, fontsize=11)
    ax.set_ylabel(ylabel, color=LIGHT, fontsize=11)
    ax.tick_params(colors=LIGHT, labelsize=9)
    for s in ax.spines.values(): s.set_color('#334155')
    ax.grid(axis='y', color='#1E293B', linewidth=0.5, alpha=0.7)


# ═══════════════════════════════════════════
# V15 모델 정의
# ═══════════════════════════════════════════
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False))
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = self.sigmoid_channel(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)
        x = x * out
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        spatial = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_s, max_s], dim=1)))
        return x * spatial


class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, drop=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.drop1 = nn.Dropout2d(drop)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.cbam = CBAM(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop1(out)
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        return F.relu(out + self.shortcut(x))


class HeartSoundModel(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = ResNetBlock(64, 64, drop=0.2)
        self.layer2 = ResNetBlock(64, 128, stride=2, drop=0.2)
        self.layer3 = ResNetBlock(128, 256, stride=2, drop=0.3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        return self.fc(self.dropout_fc(feat)), feat


# ═══════════════════════════════════════════
# V15 평가
# ═══════════════════════════════════════════
print("\n" + "="*60)
print("V15 Evaluation")
print("="*60)

V15_DIR = r"G:\stetho_ai\Heart_binary_classification"
X = np.load(os.path.join(V15_DIR, 'x_data.npy'))
y = np.load(os.path.join(V15_DIR, 'y_data.npy'))
print(f"V15 data: {X.shape}, classes: {np.unique(y, return_counts=True)}")

# 동일한 split 재현 (seed 고정)
torch.manual_seed(42)
dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

# 모델 로드
model_v15 = HeartSoundModel(num_classes=3, dropout_rate=0.5).to(DEVICE)
model_v15.load_state_dict(torch.load(os.path.join(V15_DIR, 'resnet_cbam_best.pth'), map_location=DEVICE))
model_v15.eval()

xgb_v15 = XGBClassifier()
xgb_v15.load_model(os.path.join(V15_DIR, 'heart_xgb_model.json'))

# Phase 1: CNN feature extraction on val set
all_features_v15, all_labels_v15, all_cnn_preds_v15 = [], [], []
with torch.no_grad():
    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(DEVICE)
        logits, feats = model_v15(batch_x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_features_v15.append(feats.cpu().numpy())
        all_labels_v15.append(batch_y.numpy())
        all_cnn_preds_v15.append(preds)

features_v15 = np.concatenate(all_features_v15)
labels_v15 = np.concatenate(all_labels_v15)
cnn_preds_v15 = np.concatenate(all_cnn_preds_v15)

# Phase 2: XGBoost predictions on features
xgb_preds_v15 = xgb_v15.predict(features_v15)
xgb_probs_v15 = xgb_v15.predict_proba(features_v15)

# Metrics
v15_acc = accuracy_score(labels_v15, xgb_preds_v15)
v15_cnn_acc = accuracy_score(labels_v15, cnn_preds_v15)
v15_report = classification_report(labels_v15, xgb_preds_v15,
                                    target_names=['Normal', 'Abnormal', 'Unknown'],
                                    output_dict=True)
v15_cm = confusion_matrix(labels_v15, xgb_preds_v15)

print(f"\nV15 CNN-only Accuracy: {v15_cnn_acc:.4f}")
print(f"V15 XGBoost Ensemble Accuracy: {v15_acc:.4f}")
print(f"\nV15 Classification Report:")
print(classification_report(labels_v15, xgb_preds_v15,
                             target_names=['Normal', 'Abnormal', 'Unknown']))
print(f"V15 Confusion Matrix:\n{v15_cm}")

# Per-class metrics
v15_metrics = {}
for cls_name in ['Normal', 'Abnormal', 'Unknown']:
    v15_metrics[cls_name] = {
        'precision': v15_report[cls_name]['precision'],
        'recall': v15_report[cls_name]['recall'],
        'f1': v15_report[cls_name]['f1-score'],
    }

# Binary 비교를 위한 V15 (Normal vs Abnormal만, Unknown 제외)
mask_binary = labels_v15 != 2
labels_v15_bin = labels_v15[mask_binary]
preds_v15_bin = xgb_preds_v15[mask_binary]
# Unknown 예측을 Abnormal로 처리 (보수적)
preds_v15_bin_adj = np.where(preds_v15_bin == 2, 1, preds_v15_bin)

v15_se = recall_score(labels_v15_bin, preds_v15_bin_adj, pos_label=1)  # Abnormal recall
v15_sp = recall_score(labels_v15_bin, preds_v15_bin_adj, pos_label=0)  # Normal recall
v15_macc = (v15_se + v15_sp) / 2

print(f"\nV15 Binary metrics (Normal vs Abnormal, Unknown excluded):")
print(f"  Sensitivity (Se): {v15_se:.4f}")
print(f"  Specificity (Sp): {v15_sp:.4f}")
print(f"  MACC: {v15_macc:.4f}")


# ═══════════════════════════════════════════
# V14 평가 (cached features)
# ═══════════════════════════════════════════
print("\n" + "="*60)
print("V14 (A+B+C) Evaluation")
print("="*60)

V14_DIR = r"G:\stetho_ai\_misc\heart\classification\adult_v1_to_v14\code\전처리 14차"

v14_vl_y = np.load(os.path.join(V14_DIR, 'cache_v14_A_B_C_vl_y.npy'))
v14_vl_cnn_input = np.load(os.path.join(V14_DIR, 'cache_v14_A_B_C_vl_cnn.npy'))  # (N, 4, 64, 64)
v14_vl_xgb_feats = np.load(os.path.join(V14_DIR, 'cache_v14_A_B_C_vl_xgb.npy'))  # (N, 64)

import joblib
import torchvision.models as tv_models
v14_xgb = joblib.load(os.path.join(V14_DIR, 'v14_A_B_C_xgboost.pkl'))
v14_scaler = joblib.load(os.path.join(V14_DIR, 'v14_A_B_C_scaler.pkl'))
v14_params = np.load(os.path.join(V14_DIR, 'v14_A_B_C_best_params.npy'), allow_pickle=True).item()

print(f"V14 val data: cnn_input={v14_vl_cnn_input.shape}, xgb_feats={v14_vl_xgb_feats.shape}, y={v14_vl_y.shape}")
print(f"V14 best params: {v14_params}")

# V14 ResNet 모델 정의 및 로드
class CBAM_V14(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False), nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False))
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x)); max_out = self.fc(self.max_pool(x))
        out = self.sigmoid_channel(avg_out + max_out) * x
        avg_s = torch.mean(out, dim=1, keepdim=True); max_s, _ = torch.max(out, dim=1, keepdim=True)
        return out * self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_s, max_s], dim=1)))

class HeartSoundResNetV14(nn.Module):
    def __init__(self, use_dropout=True):
        super().__init__()
        base = tv_models.resnet18(weights=None)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1, self.relu, self.maxpool = base.bn1, base.relu, base.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.cbam1, self.cbam2, self.cbam3, self.cbam4 = CBAM_V14(64), CBAM_V14(128), CBAM_V14(256), CBAM_V14(512)
        self.avgpool = base.avgpool
        self.dropout = nn.Dropout(p=0.5) if use_dropout else nn.Identity()
        self.fc = nn.Linear(512, 2)
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.cbam1(self.layer1(x)); x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x)); x = self.cbam4(self.layer4(x))
        x = self.avgpool(x).view(x.size(0), -1)
        return self.fc(self.dropout(x))

# V14 CNN 로드 및 추론
v14_cnn = HeartSoundResNetV14(use_dropout=True).to(DEVICE)
v14_cnn.load_state_dict(torch.load(os.path.join(V14_DIR, 'v14_A_B_C_resnet.pth'), map_location=DEVICE, weights_only=True))
v14_cnn.eval()

v14_cnn_probs = []
v14_loader = DataLoader(TensorDataset(torch.FloatTensor(v14_vl_cnn_input)), batch_size=128, shuffle=False)
with torch.no_grad():
    for (batch_x,) in v14_loader:
        logits = v14_cnn(batch_x.to(DEVICE))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        v14_cnn_probs.append(probs)
v14_cnn_probs = np.concatenate(v14_cnn_probs)

# V14 XGBoost 추론
v14_xgb_scaled = v14_scaler.transform(v14_vl_xgb_feats)
v14_xgb_probs = v14_xgb.predict_proba(v14_xgb_scaled)[:, 1]

# V14 앙상블
v14_combined = v14_params['cnn_w'] * v14_cnn_probs + v14_params['xgb_w'] * v14_xgb_probs
v14_preds = (v14_combined >= v14_params['thr']).astype(int)

v14_se = recall_score(v14_vl_y, v14_preds, pos_label=1)
v14_sp = recall_score(v14_vl_y, v14_preds, pos_label=0)
v14_macc = (v14_se + v14_sp) / 2
v14_acc = accuracy_score(v14_vl_y, v14_preds)
v14_f1 = f1_score(v14_vl_y, v14_preds, average='weighted')
v14_cm = confusion_matrix(v14_vl_y, v14_preds)

print(f"\nV14 Accuracy: {v14_acc:.4f}")
print(f"V14 Sensitivity (Se): {v14_se:.4f}")
print(f"V14 Specificity (Sp): {v14_sp:.4f}")
print(f"V14 MACC: {v14_macc:.4f}")
print(f"V14 F1 (weighted): {v14_f1:.4f}")
print(f"V14 Confusion Matrix:\n{v14_cm}")


# ═══════════════════════════════════════════
# 그래프 생성
# ═══════════════════════════════════════════
print("\n그래프 생성 중...")

fig = plt.figure(figsize=(18, 12), facecolor=BG)
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

# ── Panel 1: MACC / Se / Sp 비교 바 차트 ──
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, 'V14 vs V15 — Key Metrics', '', 'Score')

metrics_names = ['MACC', 'Sensitivity\n(Se)', 'Specificity\n(Sp)', 'Accuracy']
v14_vals = [v14_macc, v14_se, v14_sp, v14_acc]
v15_vals = [v15_macc, v15_se, v15_sp, v15_acc]

x = np.arange(len(metrics_names))
w = 0.35
bars1 = ax1.bar(x - w/2, v14_vals, w, color=ORANGE, label='V14', alpha=0.85)
bars2 = ax1.bar(x + w/2, v15_vals, w, color=CYAN, label='V15', alpha=0.85)

ax1.set_xticks(x); ax1.set_xticklabels(metrics_names, fontsize=9)
ax1.set_ylim(0, 1.15)
ax1.legend(facecolor=CARD, edgecolor='#334155', labelcolor=WHITE, fontsize=10)

for bar, val in zip(bars1, v14_vals):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
             f'{val:.2%}', ha='center', fontsize=9, color=ORANGE, fontweight='bold')
for bar, val in zip(bars2, v15_vals):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
             f'{val:.2%}', ha='center', fontsize=9, color=CYAN, fontweight='bold')


# ── Panel 2: V15 3-Class Performance ──
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, 'V15 Per-Class Performance', '', 'Score')

cls_names = ['Normal', 'Abnormal', 'Unknown']
cls_colors = [GREEN, RED, ORANGE]
prec = [v15_metrics[c]['precision'] for c in cls_names]
rec = [v15_metrics[c]['recall'] for c in cls_names]
f1s = [v15_metrics[c]['f1'] for c in cls_names]

x2 = np.arange(3)
w2 = 0.25
ax2.bar(x2 - w2, prec, w2, color=[c for c in cls_colors], alpha=0.5, label='Precision')
ax2.bar(x2, rec, w2, color=[c for c in cls_colors], alpha=0.75, label='Recall')
ax2.bar(x2 + w2, f1s, w2, color=[c for c in cls_colors], alpha=1.0, label='F1-Score')

for i, (p, r, f) in enumerate(zip(prec, rec, f1s)):
    ax2.text(i - w2, p + 0.02, f'{p:.2f}', ha='center', fontsize=8, color=LIGHT)
    ax2.text(i, r + 0.02, f'{r:.2f}', ha='center', fontsize=8, color=WHITE, fontweight='bold')
    ax2.text(i + w2, f + 0.02, f'{f:.2f}', ha='center', fontsize=8, color=LIGHT)

ax2.set_xticks(x2); ax2.set_xticklabels(cls_names, fontsize=10)
ax2.set_ylim(0, 1.15)
ax2.legend(facecolor=CARD, edgecolor='#334155', labelcolor=WHITE, fontsize=9)


# ── Panel 3: Confusion Matrices ──
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, 'V15 Confusion Matrix (3-Class)', '', '')

# Normalize
v15_cm_norm = v15_cm.astype(float) / v15_cm.sum(axis=1, keepdims=True)
im = ax3.imshow(v15_cm_norm, cmap='Blues', vmin=0, vmax=1)

for i in range(3):
    for j in range(3):
        val = v15_cm[i, j]
        pct = v15_cm_norm[i, j]
        color = WHITE if pct > 0.5 else LIGHT
        ax3.text(j, i, f'{val}\n({pct:.0%})', ha='center', va='center',
                 fontsize=10, color=color, fontweight='bold')

ax3.set_xticks([0,1,2]); ax3.set_yticks([0,1,2])
ax3.set_xticklabels(['Normal', 'Abnormal', 'Unknown'], fontsize=9, color=LIGHT)
ax3.set_yticklabels(['Normal', 'Abnormal', 'Unknown'], fontsize=9, color=LIGHT)
ax3.set_xlabel('Predicted', color=LIGHT, fontsize=10)
ax3.set_ylabel('Actual', color=LIGHT, fontsize=10)


# ── Panel 4: 개선도 Delta 차트 ──
ax4 = fig.add_subplot(gs[1, 0])
style_ax(ax4, 'V14 → V15 Improvement (Delta)', '', 'Improvement')

delta_names = ['MACC', 'Sensitivity', 'Specificity', 'Accuracy']
deltas = [v15_macc - v14_macc, v15_se - v14_se, v15_sp - v14_sp, v15_acc - v14_acc]
delta_colors = [GREEN if d >= 0 else RED for d in deltas]

bars = ax4.bar(delta_names, deltas, color=delta_colors, width=0.5)
ax4.axhline(y=0, color=GRAY, linewidth=1)
for bar, d in zip(bars, deltas):
    y_off = 0.01 if d >= 0 else -0.03
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+y_off,
             f'{d:+.2%}', ha='center', fontsize=12, color=WHITE, fontweight='bold')


# ── Panel 5: V14 vs V15 구조 비교 요약 ──
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5, 'Architecture Comparison')
ax5.set_xlim(0, 10); ax5.set_ylim(0, 8)
ax5.set_xticks([]); ax5.set_yticks([]); ax5.grid(False)

comparisons = [
    ('Classes', '2 (N/A)', '3 (N/A/U)', 7.0),
    ('Input', '4ch 64x64', '1ch 64x79', 6.0),
    ('Sample Rate', '1000 Hz', '4000 Hz', 5.0),
    ('ResNet', 'ResNet-18\n(ImageNet)', 'Custom 3-block\n(64>128>256)', 3.8),
    ('XGB Trees', '300', '100', 2.8),
    ('Inference', 'w*CNN+(1-w)*XGB\n> threshold', 'mean(proba)\nargmax', 1.5),
]

ax5.text(2.5, 7.6, 'V14', ha='center', fontsize=12, color=ORANGE, fontweight='bold')
ax5.text(7.5, 7.6, 'V15', ha='center', fontsize=12, color=CYAN, fontweight='bold')

for label, v14_text, v15_text, y_pos in comparisons:
    ax5.text(0.2, y_pos, label, fontsize=9, color=WHITE, fontweight='bold')
    ax5.text(2.5, y_pos, v14_text, ha='center', fontsize=8, color=ORANGE)
    ax5.text(7.5, y_pos, v15_text, ha='center', fontsize=8, color=CYAN)


# ── Panel 6: CNN-only vs Ensemble 비교 ──
ax6 = fig.add_subplot(gs[1, 2])
style_ax(ax6, 'V15: CNN-only vs Ensemble', '', 'Accuracy')

cmp_names = ['CNN Only\n(ResNet+CBAM)', 'CNN + XGBoost\n(Ensemble)']
cmp_vals = [v15_cnn_acc, v15_acc]
cmp_colors = [PURPLE, CYAN]
bars = ax6.bar(cmp_names, cmp_vals, color=cmp_colors, width=0.4)
ax6.set_ylim(0, 1.15)
for bar, val in zip(bars, cmp_vals):
    ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
             f'{val:.2%}', ha='center', fontsize=14, color=WHITE, fontweight='bold')

delta_ens = cmp_vals[1] - cmp_vals[0]
if delta_ens > 0:
    ax6.text(0.5, max(cmp_vals)*0.5, f'Ensemble 효과: +{delta_ens:.2%}',
             ha='center', fontsize=12, color=GREEN, fontweight='bold')


fig.suptitle('Heart Sound Classification — V14 vs V15 Performance Comparison',
             color=WHITE, fontsize=18, fontweight='bold')

output_path = os.path.join(OUT_DIR, '08_v14_vs_v15_performance.png')
fig.savefig(output_path, dpi=200, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {output_path}")
print(f"  Size: {os.path.getsize(output_path)/1024:.0f} KB")

# ═══════════════════════════════════════════
# 결과 요약 JSON 저장
# ═══════════════════════════════════════════
summary = {
    'v14': {
        'accuracy': float(v14_acc),
        'sensitivity': float(v14_se),
        'specificity': float(v14_sp),
        'macc': float(v14_macc),
        'f1_weighted': float(v14_f1),
        'confusion_matrix': v14_cm.tolist(),
        'params': {k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in v14_params.items()},
    },
    'v15': {
        'accuracy': float(v15_acc),
        'cnn_only_accuracy': float(v15_cnn_acc),
        'sensitivity': float(v15_se),
        'specificity': float(v15_sp),
        'macc': float(v15_macc),
        'per_class': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in v15_metrics.items()},
        'confusion_matrix': v15_cm.tolist(),
    },
    'improvement': {
        'accuracy': float(v15_acc - v14_acc),
        'sensitivity': float(v15_se - v14_se),
        'specificity': float(v15_sp - v14_sp),
        'macc': float(v15_macc - v14_macc),
    }
}

summary_path = os.path.join(OUT_DIR, 'v14_vs_v15_results.json')
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"Results JSON: {summary_path}")

print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(f"{'Metric':<15} {'V14':>10} {'V15':>10} {'Delta':>10}")
print("-"*45)
print(f"{'MACC':<15} {v14_macc:>10.2%} {v15_macc:>10.2%} {v15_macc-v14_macc:>+10.2%}")
print(f"{'Sensitivity':<15} {v14_se:>10.2%} {v15_se:>10.2%} {v15_se-v14_se:>+10.2%}")
print(f"{'Specificity':<15} {v14_sp:>10.2%} {v15_sp:>10.2%} {v15_sp-v14_sp:>+10.2%}")
print(f"{'Accuracy':<15} {v14_acc:>10.2%} {v15_acc:>10.2%} {v15_acc-v14_acc:>+10.2%}")
print("="*60)
