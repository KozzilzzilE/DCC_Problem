🛠️ VS Code 로컬 개발 환경 세팅 가이드 (데이터+AI 혁신 챌린지)
1. 필수 VS Code 확장 프로그램 (Extensions) 설치
좌측 확장 프로그램 탭(Ctrl+Shift+X)에서 다음 항목들을 검색하여 설치합니다.

Python: (Microsoft) 파이썬 코드 하이라이팅 및 디버깅

Jupyter: (Microsoft) VS Code 내에서 .ipynb 파일 실행

Pylance: (Microsoft) 빠르고 정확한 파이썬 자동 완성

GitLens: (GitKraken) 팀원들과의 코드 협업 시 커밋 히스토리 추적

2. 프로젝트 폴더 생성 및 가상환경 세팅
VS Code 상단 메뉴에서 터미널(Terminal) > 새 터미널(New Terminal)을 열고 독립적인 파이썬 가상환경을 생성합니다.

Bash
# 1. 프로젝트 폴더 생성 및 이동 (필요시)
mkdir data_ai_challenge
cd data_ai_challenge

# 2. 가상환경 생성 (가상환경 이름: venv)
python -m venv venv

# 3. 가상환경 활성화
# Windows의 경우:
.\venv\Scripts\activate
# macOS/Linux의 경우:
source venv/bin/activate
(활성화되면 터미널 프롬프트 앞에 (venv)가 표시됩니다.)

3. 핵심 라이브러리 및 PyTorch (CUDA) 설치
보유하신 RTX 3060의 GPU 가속을 사용하기 위해 PyTorch는 반드시 CUDA 지원 버전으로 설치해야 합니다.

Bash
# 1. pip 업데이트
python -m pip install --upgrade pip

# 2. PyTorch (CUDA 지원 버전) 설치 (버전은 PyTorch 공식 홈페이지 설정 참고)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 오디오 및 텍스트 처리 필수 라이브러리 설치
pip install librosa soundfile transformers pandas scikit-learn matplotlib jupyter
4. requirements.txt 관리
팀원들과 동일한 패키지 환경을 맞추기 위해 현재 설치된 목록을 추출합니다.

Bash
pip freeze > requirements.txt
팀원들은 프로젝트를 클론한 뒤 pip install -r requirements.txt 명령어로 한 번에 세팅할 수 있습니다.

5. 작업 폴더 구조 세팅
대용량 데이터(87GB)를 효율적으로 다루기 위해 작업 폴더 구조를 다음과 같이 구성하는 것을 권장합니다. 용량이 큰 data/ 폴더는 외장 NVMe SSD 등에 두고 심볼릭 링크로 연결하는 것도 좋은 방법입니다.

Plaintext
data_ai_challenge/
│
├── venv/                   # 가상환경 (Git 업로드 제외)
├── data/                   # 데이터셋 폴더 (Git 업로드 제외)
│   ├── train/              # 서울 지역 데이터 (학습용)
│   └── val/                # Validation 데이터 (학습 사용 금지)
│
├── mission1_gender/        # 팀원 A, B 작업 디렉토리
├── mission2_speaker/       # 팀원 C 작업 디렉토리
├── mission3_symptom/       # 팀원 D, E 작업 디렉토리
│
├── requirements.txt        # 패키지 의존성 파일
└── inference.py            # 최종 추론 스크립트 (제출용)
6. VS Code에서 Jupyter Notebook 커널 연결
탐색기에서 새 파일 test.ipynb를 생성하고 엽니다.

우측 상단의 '커널 선택 (Select Kernel)'을 클릭합니다.

'Python Environments'를 누른 후, 방금 생성한 venv 경로 안의 Python을 선택합니다.

아래 코드를 셀에 입력하고 실행(Shift+Enter)하여 GPU가 정상적으로 인식되는지 테스트합니다.

Python
import torch
print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
print(f"GPU 기기명: {torch.cuda.get_device_name(0)}")