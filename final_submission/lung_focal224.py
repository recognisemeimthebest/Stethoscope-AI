import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms, models
import os
import copy


# ---------------------------------------------------------
# [⭐ 핵심 기술 1] Focal Loss (초점 손실 함수) 정의
# 맞추기 쉬운 Normal(정상)은 대충 넘어가고,
# 헷갈리는 Crackle(수포음)을 틀리면 벌점을 기하급수적으로 때립니다.
# ---------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma  # 감마값이 클수록 어려운 문제에 더 집착함 (보통 2 사용)
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# 1. 경로 설정 및 기본 세팅
data_dir = r"G:\stetho_ai\_misc\lung\classification\LUNG_SOUND\Spectrograms_Balanced"
batch_size = 32
num_epochs = 50
learning_rate = 0.001
early_stopping_patience = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 연산 장치: {device}")

# ---------------------------------------------------------
# [⭐ 핵심 기술 2] 해상도 224x224 (개안 수술)
# MobileNetV2가 원래 가장 잘 보던 크기입니다. 미세한 점(Crackle)이 살아납니다.
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 128에서 224로 업그레이드!
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(train_dataset.classes)

# 3. 모델 구축: Dropout 0.5 유지
model = models.mobilenet_v2(weights='DEFAULT')
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.last_channel, num_classes)
)
model = model.to(device)

# 4. 손실 함수(Focal Loss로 교체!)와 최적화 도구
criterion = FocalLoss(gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

best_val_loss = float('inf')
epochs_no_improve = 0
best_model_wts = copy.deepcopy(model.state_dict())

# 5. 본격적인 학습 루프
print(f"\n--- AI 호흡음 학습 V2 (해상도 224 & Focal Loss 적용) ---")
print(f"클래스 목록: {train_dataset.classes} ({num_classes}개)")
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss_avg = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    # 6. 중간 점검
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss_avg = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total

    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch [{epoch + 1:02d}/{num_epochs}] (LR: {current_lr:.6f})")
    print(f"  Train Loss: {train_loss_avg:.4f}, Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss_avg:.4f}, Acc: {val_acc:.2f}%")

    scheduler.step(val_loss_avg)

    # 7. 조기 종료 로직
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        epochs_no_improve = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        print(f"  🌟 검증 손실 감소! 최적의 모델 가중치 저장 (Val Loss: {best_val_loss:.4f})")
    else:
        epochs_no_improve += 1
        print(f"  ⚠️ 성능 미개선 카운트: {epochs_no_improve}/{early_stopping_patience}")

    if epochs_no_improve >= early_stopping_patience:
        print(f"\n🛑 조기 종료 발동! {epoch + 1}번째 에포크에서 훈련 중단.")
        break

# 8. V2 모델로 이름 변경하여 최종 저장
model.load_state_dict(best_model_wts)
save_path = r"G:\stetho_ai\_misc\lung\classification\LUNG_SOUND\stetho_mobilenetv2_224_focal.pth"
torch.save(model.state_dict(), save_path)
print(f"\n최종 라즈베리파이 탑재용 V2 모델 저장 완료: {save_path}")