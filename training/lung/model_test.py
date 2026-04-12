import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os
from sklearn.metrics import classification_report, confusion_matrix

# 1. 경로 설정 (이미 변환된 스펙트로그램 테스트 폴더 사용)
test_dir = r"G:\stetho_ai\LUNG SOUND\Spectrograms_Balanced\test"
model_path = r"G:\stetho_ai\LUNG SOUND\stetho_mobilenetv2_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 연산 장치: {device}\n채점을 시작합니다. 잠시만 기다려주세요...\n")

# 2. 데이터 로더 세팅 (학습할 때와 동일한 규격)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
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
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()  # 평가 모드 (매우 중요)

# 4. 전체 데이터 평가 (일제고사)
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # 예측값과 정답을 리스트에 모아둠
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 5. 성적표 출력
print("=" * 60)
print(" 🩺 AI 청진기 최종 테스트 성적표 (Test Dataset)")
print("=" * 60)

# 정답률 계산
correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
total = len(all_labels)
print(f"▶ 총 테스트 데이터: {total}개")
print(f"▶ 맞춘 개수: {correct}개")
print(f"▶ 최종 정답률 (Accuracy): {100 * correct / total:.2f}%\n")

# 혼동 행렬 (Confusion Matrix) 출력
print("-" * 60)
print(" 📊 혼동 행렬 (세로: 실제 정답 / 가로: AI의 예측)")
print("-" * 60)
cm = confusion_matrix(all_labels, all_preds)
# 보기 좋게 출력
print(f"{'':<10} " + " ".join([f"{c[:3]:>6}" for c in classes]))  # 클래스 이름 앞 3글자만
for i, row in enumerate(cm):
    print(f"{classes[i]:<10} " + " ".join([f"{val:>6}" for val in row]))

print("\n" + "-" * 60)
print(" 📋 질환별 세부 성적표 (Precision & Recall)")
print("-" * 60)
report = classification_report(all_labels, all_preds, target_names=classes)
print(report)