# 🩺 Stethoscope-AI

> **A smart electronic stethoscope built by a nursing student — where clinical insight meets embedded AI.**

---

## 💡 About This Project

**Stethoscope-AI**는 간호학과 학생이 직접 설계하고 개발하는 AI 기반 전자 청진기 프로젝트입니다.

단순한 소리 측정 장치를 넘어, 심장음·호흡음 등 생체 음향 데이터를 실시간으로 수집·분석하여 임상 현장에서 실질적으로 활용 가능한 인사이트를 제공하는 것을 목표로 합니다. 간호사의 시선으로 설계되었기에, 기술보다 **환자 중심**의 가치를 우선합니다.

> *"청진기를 귀에 꽂는 순간, 환자의 이야기가 들린다."*  
> 이 프로젝트는 그 이야기를 더 잘 듣기 위한 도구입니다.

---

## ✨ Key Features

- 🎙️ **실시간 생체 음향 수집** — 심장음 및 호흡음 고품질 녹음
- 🤖 **AI 기반 음향 분석** — 이상음 패턴 감지 및 분류
- 📡 **무선 데이터 전송** — BLE를 통한 실시간 스트리밍
- 📊 **시각화 대시보드** — 파형 및 주파수 분석 시각화
- 🏥 **임상 친화적 UI** — 간호 실무를 고려한 직관적 인터페이스

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Hardware** | Seeed XIAO nRF52840 Sense |
| **Backend / AI** | Python |
| **AI Development** | Claude Code |

### Hardware — Seeed XIAO nRF52840 Sense
초소형 폼팩터에 내장 IMU, PDM 마이크, BLE 5.0을 탑재한 보드. 청진기의 소형화와 무선 연결성을 동시에 실현합니다.

### Software — Python
음향 신호 처리, AI 모델 추론, 데이터 시각화 파이프라인 전반을 담당합니다.

### AI Development — Claude Code
임베디드 펌웨어부터 신호 처리 알고리즘까지, Claude Code와의 협업으로 개발 속도와 코드 품질을 높입니다.

---

## 📁 Project Structure

```
Stethoscope-AI/
├── firmware/           # nRF52840 펌웨어 (BLE, 마이크 드라이버)
├── backend/            # Python 신호 처리 및 AI 추론
│   ├── signal/         # 음향 전처리 모듈
│   ├── model/          # AI 분류 모델
│   └── api/            # 데이터 수신 API
├── dashboard/          # 시각화 대시보드
├── data/               # 수집 음향 샘플 (익명화)
└── docs/               # 설계 문서 및 회로도
```

---

## 🚀 Getting Started

```bash
# 저장소 클론
git clone https://github.com/your-username/Stethoscope-AI.git
cd Stethoscope-AI

# 의존성 설치
pip install -r requirements.txt

# 백엔드 실행
python backend/main.py
```

> 펌웨어 플래싱 및 하드웨어 셋업은 [`docs/setup.md`](docs/setup.md)를 참고하세요.

---

## 🗺️ Roadmap

- [x] 프로젝트 기획 및 하드웨어 선정
- [ ] nRF52840 마이크 드라이버 구현
- [ ] BLE 실시간 스트리밍
- [ ] 음향 전처리 파이프라인 구축
- [ ] AI 이상음 분류 모델 학습
- [ ] 임상 테스트 및 검증
- [ ] 최종 프로토타입 완성

---

## 👩‍⚕️ Developer

**간호학과 학생 개발자** — 임상 지식과 기술을 연결하는 다리를 만들고 있습니다.  
환자 곁에서 배운 것들을 코드로 옮기는 중입니다.

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

<p align="center">
  Made with ❤️ by a nursing student who codes
</p>
