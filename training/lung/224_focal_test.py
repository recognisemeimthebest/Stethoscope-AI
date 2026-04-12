import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os
from sklearn.metrics import classification_report, confusion_matrix

# 1. 경로 설정 (새로 만든 V2 모델 파일 장착!)
test_dir = r"G:\stetho_ai\LUNG SOUND\Spectrograms_Balanced\test"
model_path = r"G:\stetho_ai\LUNG SOUND\stetho_mobilenetv2_224_focal.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 연산 장치: {device}\n개안 수술(224x224) 완료 모델로 채점을 시작합니다...\n")

# 2. 데이터 로더 세팅 (⭐ 테스트할 때도 동일하게 224x224 돋보기 장착!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
classes = test_dataset.classes

# 3. 모델 불러오기
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.last_channel, len(classes))
)
model.load_state_dict(
    torch.load(model_path, map_location=device, weights_only=True))  # warning 방지용 weights_only=True 추가
model = model.to(device)
model.eval()

# 4. 전체 데이터 평가
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 5. 성적표 출력
print("=" * 60)
print(" 🩺 AI 청진기 V2 (해상도 업그레이드) 최종 성적표")
print("=" * 60)

correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
total = len(all_labels)
print(f"▶ 총 테스트 데이터: {total}개")
print(f"▶ 맞춘 개수: {correct}개")
print(f"▶ 최종 정답률 (Accuracy): {100 * correct / total:.2f}%\n")

print("-" * 60)
print(" 📊 혼동 행렬 (세로: 실제 정답 / 가로: AI의 예측)")
print("-" * 60)
cm = confusion_matrix(all_labels, all_preds)
print(f"{'':<10} " + " ".join([f"{c[:3]:>6}" for c in classes]))
for i, row in enumerate(cm):
    print(f"{classes[i]:<10} " + " ".join([f"{val:>6}" for val in row]))

print("\n" + "-" * 60)
print(" 📋 질환별 세부 성적표 (Precision & Recall)")
print("-" * 60)
report = classification_report(all_labels, all_preds, target_names=classes)
print(report)