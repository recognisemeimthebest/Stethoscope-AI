import torch
import torch.nn as nn
from torchvision import transforms, models
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import warnings
import os

warnings.filterwarnings('ignore')

# 1. 경로 설정 (직접 녹음한 파일 경로로 수정하세요)
model_path = r"G:\stetho_ai\LUNG SOUND\stetho_mobilenetv2_224_focal.pth"
custom_audio_path = r"G:\stetho_ai\raw_heart_sound_x4.wav"  # 실제 파일 경로로 수정!

classes = ['Complex', 'Crackle', 'Normal', 'Stridor', 'Wheeze']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. V2 모델 불러오기
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.last_channel, len(classes))
)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model = model.to(device)
model.eval()


# 3. 소리 전처리 함수 (에러 수정 버전)
def process_custom_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)

    # 2초보다 길면 앞의 2초만 사용, 짧으면 패딩
    if len(y) > 2 * sr:
        y = y[:2 * sr]
    else:
        y = librosa.util.fix_length(y, size=2 * sr)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmin=50, fmax=2000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # [에러 해결 구간] Matplotlib 버전에 상관없이 이미지를 텐서로 변환하는 안전한 방법
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(S_dB, aspect='auto', origin='lower', cmap='magma')
    ax.axis('off')

    # canvas를 RGBA 버퍼로 변환
    fig.canvas.draw()
    # tostring_rgb 대신 buffer_rgba 사용 후 RGB로 변환
    rgba_buffer = fig.canvas.buffer_rgba()
    img = Image.frombuffer('RGBA', fig.canvas.get_width_height(), rgba_buffer, 'raw', 'RGBA', 0, 1).convert('RGB')

    plt.close(fig)
    return img


# 4. 진단 시작
try:
    if not os.path.exists(custom_audio_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {custom_audio_path}")

    img = process_custom_audio(custom_audio_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0) * 100
        _, predicted = torch.max(outputs, 1)

    predicted_class = classes[predicted.item()]

    print("\n" + "=" * 50)
    print(f"🎤 내 녹음 파일: {os.path.basename(custom_audio_path)}")
    print("=" * 50)
    print(f"🩺 AI 청진기 진단 결과: [{predicted_class}]")
    print("-" * 50)
    for i, cls in enumerate(classes):
        marker = " ◀◀ (유력)" if i == predicted.item() else ""
        print(f"  - {cls:<10}: {probabilities[i]:>6.2f}% {marker}")
    print("=" * 50 + "\n")

except Exception as e:
    print(f"\n❌ 진단 중 오류 발생!")
    print(f"에러 내용: {e}")