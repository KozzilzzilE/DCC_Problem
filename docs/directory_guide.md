# DCC Project Directory Structure

이 문서는 프로젝트의 기본 패키지 구조와 용도를 설명합니다.
팀원들이 저장소를 클론(Clone)하거나 구글 코랩(Colab) 환경에서 세팅할 때 참고하기 위한 가이드입니다.

## 기본 디렉토리 구조

- data/
  - train/
    - audio/ : 학습용 wav 오디오 파일 위치
    - label/ : 학습용 json 라벨 파일 위치
  - val/
    - audio/ : 검증/테스트용 wav 오디오 파일 위치
    - label/ : 검증/테스트용 json 라벨 파일 위치
- docs/
  - directory_guide.md : 현재 문서 (프로젝트 구조 안내)
- mission2_speaker/
  - model_train.ipynb : Mission 2 (신고자/대원 분류) 모델 학습용 주피터 노트북
- outputs/
  - (추론 결과 csv 파일이 저장되는 빈 폴더)
- inference.py : 최종 제출을 위한 추론(Inference) 실행 스크립트

## 사용 방법

1. 데이터 세팅
   - 깃허브에는 실제 데이터 파일이 올라가지 않습니다.
   - 각 팀원은 제공받은 wav 및 json 데이터를 data/ 폴더 하위의 맞는 위치에 직접 복사해야 합니다.

2. 코랩(Colab) 환경 실행 시
   - 이 저장소의 코드를 코랩에 업로드하거나 Clone 합니다.
   - 구글 드라이브에 data 폴더를 만들고 원본 데이터를 업로드한 뒤, 코랩에서 마운트하여 경로를 맞춰줍니다.
   - model_train.ipynb 파일을 열고 순서대로 실행하여 학습을 진행합니다.

3. 모델 추론
   - 학습이 완료되어 best_model.pt 파일이 생성되면, 터미널에서 아래 명령어를 통해 추론을 진행합니다.
   - python inference.py --audio_dir data/val/audio --label_dir data/val/label --ckpt_path best_model.pt --output outputs/mission2.csv
