#!/usr/bin/env python3
"""
[로컬 RTX 3060 전용] AudioResNet-50 풀버전 고속 학습 스크립트
- 특징:
  1. 로컬 데이터 100% (max_files=None) 전체 발화 구간 정밀 학습
  2. SpecAugment (Time/Frequency Masking) 내장으로 과적합 원천 방지
  3. Mixed Precision (FP16 AMP) 가속으로 RTX 3060 VRAM 절약 및 속도 2배 향상
  4. 매 에포크 최고 성적 검증 시 checkpoints/best_resnet_full.pt 자동 저장
"""

import os
import sys
import glob
import json
import time
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import librosa
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


# =========================================================================
# 1. SpecAugment 탑재 검증 음향 데이터셋
# =========================================================================
class LocalSpeechDataset(Dataset):
    def __init__(self, split_dir, max_files=None, is_train=True, augment=False):
        self.split_dir = split_dir
        self.is_train = is_train
        self.augment = augment and is_train
        
        self.sr = 16000
        self.target_samples = int(self.sr * 3.0)  # 3.0초 윈도우 (Pre-3 입증 파라미터)
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        
        # 파일 매핑
        wav_files = glob.glob(f"{self.split_dir}/**/*.wav", recursive=True)
        self.wav_map = {Path(p).stem.replace("VS_", "VL_"): p for p in wav_files}
        for p in wav_files:
            self.wav_map[Path(p).stem] = p
            
        json_files = sorted(glob.glob(f"{self.split_dir}/**/*.json", recursive=True))
        if max_files and len(json_files) > max_files:
            random.seed(42)
            json_files = random.sample(json_files, max_files)
            
        self.samples = []
        for j_path in json_files:
            stem = Path(j_path).stem
            w_path = self.wav_map.get(stem)
            if not w_path or not os.path.exists(w_path):
                w_path = j_path.replace("2.라벨링데이터", "1.원천데이터").replace("VL_", "VS_").replace("TL_", "TS_").replace(".json", ".wav")
                if not os.path.exists(w_path):
                    continue
                    
            try:
                with open(j_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue
                
            dialogs = meta.get("utterances") or meta.get("dialogs") or meta.get("dialogue") or []
            for utt in dialogs:
                if 'speaker' in utt and ('startAt' in utt or 'start_time' in utt):
                    st = utt.get('startAt') if 'startAt' in utt else utt.get('start_time', 0)
                    et = utt.get('endAt') if 'endAt' in utt else utt.get('end_time', 0)
                    st, et = float(st), float(et)
                    if et - st > 100:
                        st, et = st / 1000.0, et / 1000.0
                    if et - st > 0.1:
                        spk_raw = str(utt['speaker']).strip()
                        label = 1 if spk_raw in ['1', '신고자', 'caller', 'c'] else 0
                        self.samples.append({
                            'wav': w_path,
                            'st': st,
                            'et': et,
                            'label': label
                        })
                        
        aug_tag = " (🔥SpecAugment ON)" if self.augment else ""
        print(f"[{'TRAIN' if is_train else 'VAL'}{aug_tag}] 총 {len(self.samples):,}개 발화 구간 로드 완료!")

    def __len__(self):
        return len(self.samples)

    def _apply_spec_augment(self, spec, max_time_mask=20, max_freq_mask=12):
        augmented = spec.copy()
        f = random.randint(0, max_freq_mask)
        f0 = random.randint(0, max(0, augmented.shape[0] - f))
        augmented[f0:f0+f, :] = 0.0
        t = random.randint(0, max_time_mask)
        t0 = random.randint(0, max(0, augmented.shape[1] - t))
        augmented[:, t0:t0+t] = 0.0
        return augmented

    def __getitem__(self, idx):
        item = self.samples[idx]
        clip_dur = max(0.01, item['et'] - item['st'])
        
        try:
            y, _ = librosa.load(item['wav'], sr=self.sr, offset=item['st'], duration=clip_dur)
        except Exception:
            y = np.zeros(self.target_samples, dtype=np.float32)
            
        cur_len = len(y)
        if cur_len < self.target_samples:
            y = np.pad(y, (0, self.target_samples - cur_len), mode='constant')
        else:
            if self.is_train:
                max_s = cur_len - self.target_samples
                s_idx = random.randint(0, max_s)
                y = y[s_idx : s_idx + self.target_samples]
            else:
                y = y[:self.target_samples]
                
        if len(y) < self.n_fft:
            y = np.pad(y, (0, self.n_fft - len(y)), mode='constant')
            
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = np.clip((mel_db + 80.0) / 80.0, 0.0, 1.0)
        
        if self.augment and random.random() < 0.5:
            mel_norm = self._apply_spec_augment(mel_norm, max_time_mask=20, max_freq_mask=12)
            
        feat = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)
        return feat, torch.tensor(item['label'], dtype=torch.float32)


# =========================================================================
# 2. AudioResNet-50 아키텍처 (ImageNet 사전학습 가중치 전이)
# =========================================================================
class AudioResNet50(nn.Module):
    def __init__(self, pretrained=True, dropout_rate=0.3):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet50(weights=weights)
        old_conv = self.resnet.conv1
        
        # 3채널 RGB 가중치 평균을 1채널 흑백 스펙트로그램에 이식
        new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride, padding=old_conv.padding, bias=False)
        if pretrained and old_conv.weight is not None:
            new_conv.weight.data = torch.mean(old_conv.weight.data, dim=1, keepdim=True)
            
        self.resnet.conv1 = new_conv
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 1)
        )
        
    def forward(self, x):
        return self.resnet(x)


# =========================================================================
# 3. 메인 학습 루틴
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="로컬 RTX 3060 전용 AudioResNet-50 풀학습")
    parser.add_argument("--train_dir", type=str, default="../../data/train", help="학습 데이터 경로")
    parser.add_argument("--val_dir", type=str, default="../../data/val", help="검증 데이터 경로")
    parser.add_argument("--epochs", type=int, default=10, help="총 학습 에포크 수 (기본: 10)")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기 (기본: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률 (기본: 1e-4)")
    parser.add_argument("--max_files", type=int, default=None, help="최대 파일 수 (None=전체 데이터 풀학습)")
    parser.add_argument("--augment", action="store_true", default=True, help="SpecAugment 활성화")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="체크포인트 저장 디렉토리")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("🚀 [로컬 RTX 3060] AudioResNet-50 풀버전 학습 시작")
    print(f"🖥️ 연산 가속기: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU 디바이스: {torch.cuda.get_device_name(0)}")
        print(f"📊 VRAM 용량: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 70)

    os.makedirs(args.save_dir, exist_ok=True)

    # 데이터로더 생성
    train_ds = LocalSpeechDataset(args.train_dir, max_files=args.max_files, is_train=True, augment=args.augment)
    val_ds   = LocalSpeechDataset(args.val_dir,   max_files=args.max_files, is_train=False, augment=False)

    num_workers = 4 if os.name != 'nt' else 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 모델, 손실함수, 옵티마이저
    model = AudioResNet50(pretrained=True, dropout_rate=0.3).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()  # FP16 혼합 정밀도 가속기

    best_acc, best_f1 = 0.0, 0.0
    start_total_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device).unsqueeze(1)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(bx)
                loss = criterion(out, by)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / len(train_loader)

        # 검증 단계
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                with torch.cuda.amp.autocast():
                    prob = torch.sigmoid(model(bx)).squeeze(-1).cpu().numpy()
                all_preds.extend((prob >= 0.5).astype(int))
                all_labels.extend(by.numpy().astype(int))

        val_acc = accuracy_score(all_labels, all_preds) * 100.0
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        epoch_dur = round(time.time() - epoch_start, 1)

        print(f"[Ep {epoch:02d}/{args.epochs:02d}] "
              f"Tr Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Macro F1: {val_f1:.4f} | "
              f"시간: {epoch_dur}s")

        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_f1
            save_path = os.path.join(args.save_dir, "best_resnet_full.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  👉 🌟 최고 점수 갱신! 가중치 저장: {save_path} ({val_acc:.2f}%)")

    total_min = round((time.time() - start_total_time) / 60.0, 1)
    print("\n" + "=" * 70)
    print(f"🎉 [학습 완료] 최고 Val Acc: {best_acc:.2f}% | Macro F1: {best_f1:.4f}")
    print(f"⏱️ 총 소요 시간: {total_min}분")
    print(f"💾 최종 산출 가중치: {os.path.join(args.save_dir, 'best_resnet_full.pt')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
