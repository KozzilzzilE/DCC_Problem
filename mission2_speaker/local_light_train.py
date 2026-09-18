import os
import torch
from benchmark_suite.dataset import UniversalSpeakerDataset
from benchmark_suite.models import build_model
from benchmark_suite.trainer import BenchmarkTrainer
from benchmark_suite.config import MODEL_REGISTRY
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import multiprocessing

if __name__ == '__main__':
    # 멀티프로세싱 지원을 위해 freeze_support 호출
    multiprocessing.freeze_support()
    
    DATA_ROOT = r"C:\Users\user\Desktop\DCC\data"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    VAL_DIR = os.path.join(DATA_ROOT, "val")
    OUTPUT_DIR = "./test_results"
    
    print("🚀 로컬(RTX 3060) 경량 모델 분산 학습 시작!")
    
    # 코랩이 아닌 로컬 폴더로 경로 설정
    trainer = BenchmarkTrainer(
        output_dir=OUTPUT_DIR,
        drive_backup_dir=OUTPUT_DIR # 로컬이므로 드라이브 경로를 동일하게 둠
    )
    
    PHASE2_MODELS = ["resnet50", "redimnet", "ecapa_tdnn", "campp"]
    
    for model_name in PHASE2_MODELS:
        cfg = MODEL_REGISTRY.get(model_name, {})
        
        train_dataset = UniversalSpeakerDataset(TRAIN_DIR, model_name=model_name, config_dict=cfg, is_train=True)
        val_dataset = UniversalSpeakerDataset(VAL_DIR, model_name=model_name, config_dict=cfg, is_train=False)
        
        # 로컬은 VRAM이 한정적이므로 batch_size를 안전하게 16으로 설정
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
        
        model = build_model(model_name, num_classes=2, pretrained=True)
        
        # epochs 15 (100% full dataset)
        trainer.fit(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=15,
            lr=1e-4,
            skip_if_done=False
        )
    
    print("✅ 로컬 학습이 모두 완료되었습니다!")
