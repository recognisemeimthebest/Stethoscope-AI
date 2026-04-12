import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 추가된 스케줄러 모듈
from torchvision import datasets, transforms, models
import os
import copy

# 1. 경로 설정 및 기본 세팅
data_dir = r"G:\stetho_ai\LUNG SOUND\Spectrograms_Balanced"
batch_size = 32
num_epochs = 50
learning_rate = 0.001
early_stopping_patience = 5  # 조기 종료는 5번 참음

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 연산 장치: {device}")

# 2. 데이터 로더 세팅 (기존과 동일 128x128)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
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

# 4. 손실 함수와 최적화 도구
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# [⭐ 새로 추가된 LR Scheduler]
# 검증 손실(Val Loss)이 2번 연속 안 떨어지면, 학습률을 기존의 0.5배(절반)로 줄입니다.
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# [Early Stopping을 위한 추적 변수들]
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_wts = copy.deepcopy(model.state_dict())

# 5. 본격적인 학습 루프
print("\n--- AI 호흡음 학습 시작 (LR Scheduler & Early Stopping 적용) ---")
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

    # 현재 적용 중인 학습률 확인
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch [{epoch + 1:02d}/{num_epochs}] (LR: {current_lr:.6f})")
    print(f"  Train Loss: {train_loss_avg:.4f}, Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss_avg:.4f}, Acc: {val_acc:.2f}%")

    # [⭐ LR 스케줄러 작동 구간]
    # 모델에게 이번 에포크의 검증 손실 결과를 알려주어 학습률을 줄일지 말지 결정하게 합니다.
    scheduler.step(val_loss_avg)

    # 7. [Early Stopping 로직 작동 구간]
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        epochs_no_improve = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        print(f"  🌟 검증 손실 감소! 최적의 모델 가중치 임시 저장됨 (Val Loss: {best_val_loss:.4f})")
    else:
        epochs_no_improve += 1
        print(f"  ⚠️ 성능 미개선 카운트: {epochs_no_improve}/{early_stopping_patience}")

    if epochs_no_improve >= early_stopping_patience:
        print(f"\n🛑 조기 종료 발동! 더 이상 학습해도 의미가 없어 {epoch + 1}번째 에포크에서 훈련을 강제 중단합니다.")
        break

# 8. 가장 똑똑했던 시절의 뇌로 복구하여 최종 저장
model.load_state_dict(best_model_wts)
save_path = r"G:\stetho_ai\LUNG SOUND\stetho_mobilenetv2_best.pth"
torch.save(model.state_dict(), save_path)
print(f"\n최종 라즈베리파이 탑재용 모델 저장 완료: {save_path}")