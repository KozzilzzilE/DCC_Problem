# 🏆 2026 데이터+AI 혁신 챌린지 환경 사용 가이드

본 가이드는 **데이터+AI 크리에이터 캠프 대학부 예선 출제문제(위급상황 음성/음향 119 신고접수 데이터)**를 풀기 위해 세팅된 환경의 사용법을 안내합니다.

## 1. 🛠️ 환경 셋업 완료 사항
현재 작업 폴더(코찔찔이)에 다음 사항들이 자동으로 세팅되었습니다.
- **Python 가상환경 (`venv`)**: `venv` 폴더에 독립적인 파이썬 환경이 생성되었습니다.
- **필수 라이브러리 설치**: PyTorch (CUDA 11.8 지원), `librosa`, `transformers`, `pandas`, `jupyter` 등 오디오 및 텍스트 처리에 필요한 패키지가 설치되었습니다.
- **작업 폴더 구조**:
  - `data/train/`, `data/val/`: 데이터셋을 배치할 폴더
  - `mission1_gender/`, `mission2_speaker/`, `mission3_symptom/`: 각 미션별 작업 폴더
  - `outputs/`: 추론 결과 CSV 파일이 저장될 폴더
  - `requirements.txt`: 설치된 패키지 목록
  - `inference.py`: **대회 제출 규격에 맞춘 최종 추론 스크립트 뼈대(Template)**

---

## 2. 💻 VS Code에서 Jupyter Notebook 사용하기 (.ipynb)

대회 분석 및 모델 학습을 위해 Jupyter Notebook(`.ipynb`)을 주로 사용하게 됩니다. 

1. **Jupyter 확장 프로그램 확인**: VS Code 좌측 확장(Extensions) 탭에서 `Jupyter`가 설치되어 있는지 확인합니다. (미설치 시 설치 필요)
2. **.ipynb 파일 생성**: 작업할 미션 폴더(예: `mission1_gender`) 안에 새로운 파일 `model_train.ipynb`를 생성합니다.
3. **커널(Kernel) 연결**: 
   - `.ipynb` 파일을 열고, 우측 상단의 **'커널 선택 (Select Kernel)'**을 클릭합니다.
   - **'Python Environments'**를 선택합니다.
   - 목록에서 방금 생성한 **`venv` 경로 안의 Python** (`.\venv\Scripts\python.exe`)을 선택합니다.
4. **GPU 확인 테스트**: 노트북의 첫 번째 셀에 다음 코드를 넣고 실행(`Shift + Enter`)하여 환경이 제대로 잡혔는지 확인합니다.
   ```python
   import torch
   print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
   ```

---

## 3. 🚀 추론(Inference) 코드 작성 및 실행 가이드

주최 측에서는 평가 시 `inference.py`를 단 1회 실행하여 결과를 확인합니다. 제공해드린 `inference.py`는 대회 요구사항인 **명령줄 인자(Argparse)**와 **CSV 출력 양식**을 미리 맞춰둔 템플릿입니다.

### 📝 inference.py 수정 방법
`inference.py` 파일을 열어보시면 `mission1_inference`, `mission2_inference`, `mission3_inference` 함수가 있습니다.
여러분은 학습을 완료한 모델 파일(`.pt`, `.pth` 등)을 불러오고(`ckpt_path` 활용), 주어진 오디오/라벨 폴더 안의 데이터를 읽어 추론한 뒤 결과 Dataframe을 반환하도록 **TODO** 부분만 채워 넣으시면 됩니다.

### ▶️ 평가 실행 명령어 (예시)
각 미션별로 아래와 같이 실행하여 출력이 정상적으로 나오는지 확인하세요.

**Mission 1 (성별 분류)**
```bash
# 터미널에서 가상환경 활성화 상태로 실행 (venv)
python inference.py --audio_dir ./data/val/audio --label_dir ./data/val/label --ckpt_path ./mission1_gender/best_model.pt --output ./outputs/mission1.csv
```

**Mission 2 (신고자/119대원 분류)**
```bash
python inference.py --audio_dir ./data/val/audio --label_dir ./data/val/label --ckpt_path ./mission2_speaker/best_model.pt --output ./outputs/mission2.csv
```

**Mission 3 (환자의 증상 인식)**
```bash
python inference.py --audio_dir ./data/val/audio --label_dir ./data/val/label --ckpt_path ./mission3_symptom/best_model.pt --output ./outputs/mission3.csv
```

---

## 4. 📌 미션별 주의사항 (PDF 참고)
- **제약 사항**: AI-Hub 제공 데이터만 사용 가능 (외부 데이터 금지), Validation 데이터는 학습 금지, 상용 API(GPT-4o 등) 사용 금지.
- **Mission 1**: 음성에서 성별(남/여) 분류. (짧은 음성은 Padding, 긴 음성은 Sliding Window 사용 권장)
- **Mission 2**: 한 사람의 발화 조각을 받아 신고자(1)/119대원(0) 분류. (`startAt`, `endAt` 외 텍스트 등 다른 Annotation 사용 불가)
- **Mission 3**: 대화 텍스트에서 증상 9종 추출. (출력 형태는 `['두통', '복통']` 같은 **String 타입**이어야 하며, 0개~9개 출력 가능)

> **💡 팁**: 음성 데이터는 `librosa` 등을 통해 MFCC나 Mel-spectrogram 2D 이미지로 변환 후 ResNet 등 이미지 모델을 적용하거나, Wav2Vec2 등 음성 특화 모델을 Fine-tuning 하는 등 다양한 방법을 시도해 보세요.
