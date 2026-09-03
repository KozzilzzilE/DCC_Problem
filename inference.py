import argparse
import os
import json
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
from torchvision import models
from tqdm.auto import tqdm

# ==========================================
# [공통] Mission 2 모델 아키텍처 정의
# ==========================================
class AudioResNet(nn.Module):
    def __init__(self):
        super(AudioResNet, self).__init__()
        self.model = models.resnet50(pretrained=False)
        old_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)
        
    def forward(self, x):
        return self.model(x)

def parse_args():
    parser = argparse.ArgumentParser(description="데이터+AI 크리에이터 캠프 본선 추론 스크립트")
    parser.add_argument("--audio_dir", type=str, required=True, help="wav 오디오 파일이 있는 폴더 경로")
    parser.add_argument("--label_dir", type=str, required=True, help="json 라벨(전사) 파일이 있는 폴더 경로")
    parser.add_argument("--ckpt_path", type=str, required=True, help="학습 완료된 모델 가중치 파일 경로 (.pt, .pth 등)")
    parser.add_argument("--output", type=str, required=True, help="결과를 저장할 CSV 파일 경로 (예: ./outputs/mission2.csv)")
    
    return parser.parse_args()

def mission1_inference(audio_dir, label_dir, ckpt_path):
    print("Mission 1 추론 시작...")
    results = [
        {"audio file name": "sample1.wav", "gender": "여"},
        {"audio file name": "sample2.wav", "gender": "남"}
    ]
    return pd.DataFrame(results)

def mission2_inference(audio_dir, label_dir, ckpt_path):
    print("Mission 2 추론 시작 (소프트 보팅 / Soft Voting 방식)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioResNet().to(device)
    
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("✅ 가중치 로드 성공!")
    except Exception as e:
        print(f"❌ 가중치 로드 실패: {e}")
        return pd.DataFrame()
        
    model.eval()
    results = []
    
    correct_count = 0
    total_count = 0
    
    json_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
    
    with torch.no_grad():
        for json_name in tqdm(json_files, desc="추론 진행 중"):
            json_path = os.path.join(label_dir, json_name)
            wav_name = json_name.replace('.json', '.wav')
            wav_path = os.path.join(audio_dir, wav_name)
            
            if not os.path.exists(wav_path):
                continue
                
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            sr = 16000
            y, _ = librosa.load(wav_path, sr=sr)
            max_time_steps = 256
            # 회원님의 아이디어: 조금씩 겹치게 자르기 (오버랩)
            stride = max_time_steps // 2  
            
            for utt in data.get('utterances', []):
                start_ms = utt['startAt']
                end_ms = utt['endAt']
                
                start_sample = int((start_ms / 1000.0) * sr)
                end_sample = int((end_ms / 1000.0) * sr)
                cut_audio = y[start_sample:end_sample]
                
                mel_spec = librosa.feature.melspectrogram(y=cut_audio, sr=sr, n_mels=128)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                current_length = mel_spec_db.shape[1]
                mel_windows = []
                
                if current_length <= max_time_steps:
                    pad_width = max_time_steps - current_length
                    window = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
                    mel_windows.append(window)
                else:
                    for start_idx in range(0, current_length - max_time_steps + 1, stride):
                        window = mel_spec_db[:, start_idx : start_idx + max_time_steps]
                        mel_windows.append(window)
                    if (current_length - max_time_steps) % stride != 0:
                        window = mel_spec_db[:, current_length - max_time_steps : current_length]
                        mel_windows.append(window)
                
                windows_tensor = torch.tensor(np.array(mel_windows), dtype=torch.float32).unsqueeze(1).to(device)
                
                # 예측 결과 (확률값 0.0 ~ 1.0) 반환
                outputs = model(windows_tensor)
                probs = torch.sigmoid(outputs).cpu().numpy()
                
                # 소프트 보팅(Soft Voting): 투표수가 동률(2:2)이 나와도, 확신하는 확률의 평균으로 결판을 냄!
                avg_prob = np.mean(probs)
                final_pred = 1 if avg_prob >= 0.5 else 0
                
                if 'speaker' in utt:
                    actual_speaker = int(utt['speaker'])
                    if final_pred == actual_speaker:
                        correct_count += 1
                    total_count += 1
                
                results.append({
                    "audio file name": wav_name,
                    "startAt": start_ms,
                    "endAt": end_ms,
                    "speaker": final_pred
                })

    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print(f"\n🎯 [소프트 보팅 평가 결과] 총 {total_count}개의 발화 중 {correct_count}개 정답!")
        print(f"📊 정답률(Accuracy): {accuracy:.2f}%")
    else:
        print("\n⚠️ JSON 파일에 정답(speaker) 정보가 없어 정답률을 계산할 수 없습니다.")

    return pd.DataFrame(results)

def mission3_inference(audio_dir, label_dir, ckpt_path):
    print("Mission 3 추론 시작...")
    results = [
        {"label file name": "sample1.json", "symptom": "['두통', '복통']"},
        {"label file name": "sample2.json", "symptom": "['고열']"}
    ]
    return pd.DataFrame(results)

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
    print(f"\n✅ 추론 완료! 결과가 {args.output}에 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    main()
