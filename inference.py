import argparse
import os
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

# librosa 와 torchvision 은 Mission 2 에서만 쓴다. 최상단에서 import 하면 이 둘이 없는
# 환경에서 스크립트가 import 단계에서 죽어, 텍스트 과제라 둘 다 필요 없는 Mission 3 까지
# 같이 0점이 된다. torchvision 은 requirements 에 +cu118 로 고정돼 있어 CPU 환경에서
# 설치가 실패할 수 있으므로 실제로 쓰는 자리에서 import 한다.

# Mission 1 패키지(mission1_gender/m1)를 import 가능하게 한다.
sys.path.insert(0, str(Path(__file__).resolve().parent / "mission1_gender"))

# 한국어 Windows 콘솔은 기본 인코딩이 cp949 라, 진행 메시지의 이모지/한글이
# UnicodeEncodeError 로 스크립트 전체를 죽인다. 평가 환경에서 결과 CSV 를 다 쓰고도
# 종료 코드가 1 이 되는 사고를 막기 위해 출력 스트림을 UTF-8 로 고정한다.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# ==========================================
# [공통] Mission 2 모델 아키텍처 정의
# ==========================================
class AudioResNet(nn.Module):
    def __init__(self, pretrained=False, dropout_rate=0.3):
        super(AudioResNet, self).__init__()
        from torchvision import models

        self.resnet = models.resnet50(weights=None)
        old_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 1)
        )
        
    def forward(self, x):
        return self.resnet(x)

def parse_args():
    parser = argparse.ArgumentParser(description="데이터+AI 크리에이터 캠프 본선 추론 스크립트")
    parser.add_argument("--audio_dir", type=str, required=True, help="wav 오디오 파일이 있는 폴더 경로")
    parser.add_argument("--label_dir", type=str, required=True, help="json 라벨(전사) 파일이 있는 폴더 경로")
    parser.add_argument("--ckpt_path", type=str, required=True, help="학습 완료된 모델 가중치 파일 경로 (.pt, .pth 등)")
    parser.add_argument("--output", type=str, required=True, help="결과를 저장할 CSV 파일 경로 (예: ./outputs/mission2.csv)")
    
    return parser.parse_args()

def mission1_inference(audio_dir, label_dir, ckpt_path):
    """신고자 성별 분류.

    라벨 JSON 에서는 startAt / endAt / speaker 만 읽어 신고자(speaker=1) 발화
    구간을 잘라내고, 조각별 확률을 통화 단위로 평균(soft voting)해 남/여를
    정한다. 전처리 설정은 체크포인트에 함께 저장돼 있어 자동 복원된다.
    """
    print("Mission 1 추론 시작 (신고자 조각 -> 통화 단위 소프트 보팅)...")

    from m1.infer import predict_directory

    return predict_directory(audio_dir, label_dir, ckpt_path)

def mission2_inference(audio_dir, label_dir, ckpt_path):
    """상황실 접수요원 vs 신고자 화자 이진 분류.

    대회 규정상 startAt / endAt 발화 구간만 오디오에서 잘라내어 추론한다.
    결정 임계값은 대회 규정대로 0.50 고정이며, 3대장 앙상블(사전 균등 가중치 1/3)
    또는 단일 모델로 학습 환경과 100% 일치하는 정규화 전처리로 추론한다.
    """
    print("Mission 2 추론 시작 (3.0초 정규화 윈도우 + 소프트 보팅)...")

    mission2_dir = Path(__file__).resolve().parent / "mission2_speaker"
    if str(mission2_dir) not in sys.path:
        sys.path.insert(0, str(mission2_dir))

    from m2.infer import predict_directory

    return predict_directory(audio_dir, label_dir, ckpt_path)

def mission3_inference(audio_dir, label_dir, ckpt_path):
    """환자 증상 9종 다중 라벨 분류.

    대화 전사 본문(`utterances[].text`)만 입력으로 쓴다. 대회 규정상 화자·시간·인적사항은
    사용할 수 없고, Mission 3 는 텍스트 과제라 `audio_dir` 도 읽지 않는다.

    결정 임계값은 대회 규정대로 0.5 고정이다. 학습 중 탐색한 class-wise threshold 는
    제출 경로에서 사용하지 않는다.

    `ckpt_path` 는 run 디렉터리(`best_model` 포함)나 `best_model` 디렉터리를 받는다.
    발화 경계 표현과 인코딩 설정은 그 안의 run_config 에서 복원한다.
    """
    print("Mission 3 추론 시작...")

    mission3_dir = Path(__file__).resolve().parent / "mission3_symptom"
    if str(mission3_dir) not in sys.path:
        sys.path.insert(0, str(mission3_dir))

    from m3.infer import predict_directory

    return predict_directory(label_dir, ckpt_path)

def main():
    args = parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    output_filename = os.path.basename(args.output).lower()
    
    if "mission1" in output_filename:
        df = mission1_inference(args.audio_dir, args.label_dir, args.ckpt_path)
    elif "mission2" in output_filename:
        df = mission2_inference(args.audio_dir, args.label_dir, args.ckpt_path)
    elif "mission3" in output_filename:
        df = mission3_inference(args.audio_dir, args.label_dir, args.ckpt_path)
    else:
        print("경고: output 파일 이름에 미션 번호가 포함되지 않았습니다.")
        df = mission1_inference(args.audio_dir, args.label_dir, args.ckpt_path)
        
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\n[확인] 추론 완료! 결과가 {args.output}에 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    main()
