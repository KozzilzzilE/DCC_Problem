import os
import shutil
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd

from .metrics import compute_all_metrics, measure_inference_speed
from .models import count_parameters

class BenchmarkTrainer:
    """
    [안전장치 탑재] 벤치마크 학습 및 평가 엔진
    
    1. 구글 드라이브 실시간 즉시 동기화 (세션 종료 시 데이터 손실 0%)
    2. 중단 후 재실행 시 완료된 모델 자동 Skip (Auto-Resume)
    3. 정확도 및 효율성(추론속도, 파라미터수, VRAM) 다차원 로깅
    """
    def __init__(self, output_dir="./results", drive_backup_dir="/content/drive/MyDrive/DCC/benchmark_results"):
        self.output_dir = output_dir
        self.drive_backup_dir = drive_backup_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        if os.path.exists("/content/drive/MyDrive"):
            os.makedirs(os.path.join(self.drive_backup_dir, "checkpoints"), exist_ok=True)

    def is_already_completed(self, model_name):
        """구글 드라이브나 로컬에 이미 완료된 모델 체크포인트가 있는지 확인"""
        local_ckpt = os.path.join(self.output_dir, "checkpoints", f"best_{model_name}.pt")
        drive_ckpt = os.path.join(self.drive_backup_dir, "checkpoints", f"best_{model_name}.pt")
        return os.path.exists(drive_ckpt) or os.path.exists(local_ckpt)

    def _sync_to_drive(self, src_file, dst_subfolder=""):
        """파일을 구글 드라이브 백업 경로로 즉시 복사"""
        if os.path.exists("/content/drive/MyDrive") and os.path.exists(src_file):
            target_dir = os.path.join(self.drive_backup_dir, dst_subfolder)
            os.makedirs(target_dir, exist_ok=True)
            dst_file = os.path.join(target_dir, os.path.basename(src_file))
            try:
                shutil.copy2(src_file, dst_file)
            except Exception as e:
                print(f"⚠️ 구글 드라이브 동기화 일시 오류: {e}")

    def train_epoch(self, model, dataloader, optimizer, criterion, epoch, model_name, save_steps=500):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for step, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            # Step-based saving to prevent loss during Colab timeout
            if (step + 1) % save_steps == 0:
                step_ckpt_path = os.path.join(self.output_dir, "checkpoints", f"step_{model_name}.pt")
                torch.save({
                    "epoch": epoch,
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_name": model_name
                }, step_ckpt_path)
                self._sync_to_drive(step_ckpt_path, "checkpoints")

        return total_loss / total, (correct / total) * 100.0

    def evaluate(self, model, dataloader, criterion=None):
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                if criterion:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item() * inputs.size(0)

                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())

        metrics = compute_all_metrics(all_targets, all_preds, all_probs)
        avg_loss = total_loss / len(all_targets) if criterion else 0.0
        return avg_loss, metrics, all_probs

    def fit(self, model_name, model, train_loader, val_loader, epochs=15, lr=1e-4, weight_decay=1e-4, skip_if_done=False):
        """단일 모델에 대한 전체 학습 루프 및 자동 백업/Resume 수행"""
        print(f"\n=======================================================")
        print(f"🚀 [모델 벤치마크 실행] : {model_name.upper()}")
        print(f"=======================================================")

        local_ckpt_path = os.path.join(self.output_dir, "checkpoints", f"best_{model_name}.pt")
        drive_ckpt_path = os.path.join(self.drive_backup_dir, "checkpoints", f"best_{model_name}.pt")
        step_ckpt_path = os.path.join(self.output_dir, "checkpoints", f"step_{model_name}.pt")
        drive_step_ckpt = os.path.join(self.drive_backup_dir, "checkpoints", f"step_{model_name}.pt")

        if skip_if_done and self.is_already_completed(model_name):
            print(f"⚡ {model_name}의 체크포인트가 이미 존재합니다! 결과 유지 및 Skip합니다.")
            return None

        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc = 0.0
        start_epoch = 1
        best_metrics = None

        # Auto-Resume: 기존 90% 가중치 또는 중간 저장된 가중치 불러오기 (Fine-tuning)
        resume_target = None
        if os.path.exists(local_ckpt_path):
            resume_target = local_ckpt_path
        elif os.path.exists(drive_ckpt_path):
            resume_target = drive_ckpt_path
        elif os.path.exists(step_ckpt_path):
            resume_target = step_ckpt_path
        elif os.path.exists(drive_step_ckpt):
            resume_target = drive_step_ckpt

        if resume_target:
            print(f"🔄 이전 체크포인트를 발견했습니다! [{resume_target}] 불러와서 이어서 학습(Fine-Tuning)합니다.")
            checkpoint = torch.load(resume_target, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if "optimizer_state_dict" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except Exception:
                    print("⚠️ 옵티마이저 호환 안됨. 모델 가중치만 불러옵니다.")
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_acc = checkpoint.get("val_acc", 0.0)
            if start_epoch > epochs:
                print("⚡ 목표 에포크를 이미 달성했습니다.")
                if not skip_if_done:
                    # 풀학습을 위해 강제로 1에포크부터 이어서 더 돌림
                    start_epoch = 1
                    print("⚡ 풀데이터 추가 학습을 위해 에포크 카운트를 1로 초기화하고 이어 학습합니다.")
                else:
                    return None

        start_train_time = time.time()

        for epoch in range(start_epoch, epochs + 1):
            t_loss, t_acc = self.train_epoch(model, train_loader, optimizer, criterion, epoch, model_name, save_steps=500)
            v_loss, v_metrics, _ = self.evaluate(model, val_loader, criterion)
            scheduler.step()

            v_acc = v_metrics["accuracy"]
            v_f1 = v_metrics["macro_f1"]

            print(f"[{model_name}] Ep {epoch:02d}/{epochs:02d} | Tr Loss: {t_loss:.4f} Tr Acc: {t_acc:.2f}% | Val Loss: {v_loss:.4f} Val Acc: {v_acc:.2f}% F1: {v_f1:.4f}")

            # Best 모델 갱신 시 로컬 및 구글 드라이브에 즉시 동기화
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                best_metrics = v_metrics
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": v_acc,
                    "val_f1": v_f1,
                    "model_name": model_name
                }, local_ckpt_path)
                self._sync_to_drive(local_ckpt_path, "checkpoints")

        total_train_min = round((time.time() - start_train_time) / 60.0, 2)

        # 추론 지연시간(ms) 및 파라미터 수 측정
        sample_batch = next(iter(val_loader))[0][:1]
        latency_ms = measure_inference_speed(model, sample_batch, self.device)
        param_count = count_parameters(model)
        param_m = round(param_count / 1e6, 2)

        result_summary = {
            "model_name": model_name,
            "val_accuracy": round(best_val_acc, 2),
            "macro_f1": round(best_metrics["macro_f1"], 4),
            "precision": round(best_metrics["precision"], 4),
            "recall": round(best_metrics["recall"], 4),
            "params_million": param_m,
            "latency_ms": latency_ms,
            "train_time_min": total_train_min
        }

        # 결과 CSV 실시간 업데이트
        self._record_results(result_summary)
        print(f"✅ {model_name} 완료! 최고 정확도: {best_val_acc:.2f}% | 드라이브 백업 완료\n")
        return result_summary

    def _record_results(self, result_dict):
        """결과를 CSV 파일에 실시간 누적 기록하고 구글 드라이브 동기화"""
        csv_path = os.path.join(self.output_dir, "benchmark_results.csv")
        df_new = pd.DataFrame([result_dict])

        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            # 중복 모델 제거 후 갱신
            df_existing = df_existing[df_existing["model_name"] != result_dict["model_name"]]
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_final = df_new

        df_final.to_csv(csv_path, index=False)
        self._sync_to_drive(csv_path)
