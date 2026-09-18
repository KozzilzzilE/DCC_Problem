"""
[로컬 RTX 3060 환경 패치 스크립트]
benchmark_suite 코드를 윈도우 로컬 환경에서 에러 없이 돌아가도록 수정합니다.

수정 내역:
1. benchmark.py: num_workers=2 → num_workers=0 (윈도우 Jupyter Deadlock 방지)
2. trainer.py: train_epoch에 tqdm 진행률 바 추가 (실시간 진행 확인용)
3. trainer.py: AMP(혼합 정밀도) 추가 (RTX 3060 VRAM 절약 + 속도 2배)
4. dataset.py: replace() 경로 구분자를 윈도우 백슬래시에도 대응하도록 보강

사용법:
  cd C:\\Users\\user\\Desktop\\DCC\\mission2_speaker
  python patch_benchmark_for_local.py
"""
import os
import re

BENCHMARK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_suite")

def patch_file(filepath, replacements):
    """파일 내용에서 old → new 텍스트 교체"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    changed = False
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            changed = True
            print(f"  ✅ 패치 적용: '{old[:60]}...' → 수정 완료")
        else:
            print(f"  ⏭️ 이미 패치됨 또는 해당 없음: '{old[:60]}...'")
    
    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  💾 파일 저장 완료: {filepath}")
    return changed

def main():
    print("=" * 60)
    print("🔧 [로컬 윈도우 환경 패치] benchmark_suite 코드 수정 시작")
    print("=" * 60)
    
    # ──────────────────────────────────────────────────────────
    # 1. benchmark.py: num_workers=2 → num_workers=0
    # ──────────────────────────────────────────────────────────
    print("\n📁 [1/3] benchmark.py 패치 (num_workers 수정)...")
    benchmark_path = os.path.join(BENCHMARK_DIR, "benchmark.py")
    patch_file(benchmark_path, [
        ("num_workers=2, pin_memory=True)", "num_workers=0, pin_memory=True)"),
    ])
    
    # ──────────────────────────────────────────────────────────
    # 2. trainer.py: tqdm 진행률 바 + AMP(혼합 정밀도) 추가
    # ──────────────────────────────────────────────────────────
    print("\n📁 [2/3] trainer.py 패치 (tqdm + AMP 추가)...")
    trainer_path = os.path.join(BENCHMARK_DIR, "trainer.py")
    
    with open(trainer_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 2-1. train_epoch에 tqdm 감싸기 + AMP autocast 추가
    old_train_loop = """    def train_epoch(self, model, dataloader, optimizer, criterion):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        return total_loss / total, (correct / total) * 100.0"""
    
    new_train_loop = """    def train_epoch(self, model, dataloader, optimizer, criterion):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc="  Training", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / total, (correct / total) * 100.0"""
    
    if old_train_loop in content:
        content = content.replace(old_train_loop, new_train_loop)
        print("  ✅ train_epoch: tqdm + AMP autocast 패치 적용 완료")
    else:
        print("  ⏭️ train_epoch: 이미 패치됨 또는 구조 변경됨")
    
    # 2-2. __init__에 GradScaler 추가
    old_init = """        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")"""
    new_init = """        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None"""
    
    if old_init in content and "self.scaler" not in content:
        content = content.replace(old_init, new_init)
        print("  ✅ __init__: GradScaler 추가 완료")
    
    # 2-3. evaluate에도 AMP autocast 추가
    old_eval = """                outputs = model(inputs)
                if criterion:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item() * inputs.size(0)

                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()"""
    
    new_eval = """                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    outputs = model(inputs)
                if criterion:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item() * inputs.size(0)

                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()"""
    
    if old_eval in content:
        content = content.replace(old_eval, new_eval)
        print("  ✅ evaluate: AMP autocast 패치 적용 완료")
    
    with open(trainer_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("  💾 trainer.py 저장 완료")
    
    # ──────────────────────────────────────────────────────────
    # 3. dataset.py: 윈도우 경로 백슬래시 대응
    # ──────────────────────────────────────────────────────────
    print("\n📁 [3/3] dataset.py 패치 (윈도우 경로 호환)...")
    dataset_path = os.path.join(BENCHMARK_DIR, "dataset.py")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        ds_content = f.read()
    
    # label→audio 경로 변환에서 백슬래시도 처리
    old_path_replace = '''w_candidate = j_path.replace("/label/", "/audio/").replace(".json", ".wav")'''
    new_path_replace = '''w_candidate = j_path.replace(os.sep + "label" + os.sep, os.sep + "audio" + os.sep).replace(".json", ".wav")'''
    
    if old_path_replace in ds_content:
        ds_content = ds_content.replace(old_path_replace, new_path_replace)
        print("  ✅ label→audio 경로 변환: 윈도우 백슬래시 호환 패치 적용")
        
        with open(dataset_path, "w", encoding="utf-8") as f:
            f.write(ds_content)
        print("  💾 dataset.py 저장 완료")
    else:
        print("  ⏭️ 이미 패치됨 또는 해당 없음")
    
    print("\n" + "=" * 60)
    print("🎉 모든 패치 적용 완료! 이제 벤치마크를 실행할 수 있습니다.")
    print("=" * 60)

if __name__ == "__main__":
    main()
