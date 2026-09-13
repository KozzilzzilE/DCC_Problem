import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .models import build_model
from .dataset import UniversalSpeakerDataset
from .config import MODEL_REGISTRY
from .metrics import compute_all_metrics

class EnsembleEvaluator:
    """
    다중 모델의 예측 확률을 가중 결합하는 Soft Voting 앙상블 엔진.
    CNN 계열의 국소 단서 + 화자 특화 모델의 임베딩 + 트랜스포머의 문맥을 융합합니다.
    """
    def __init__(self, ckpt_dir="./results/checkpoints", data_dir="/content/data", sample_rate=16000):
        self.ckpt_dir = ckpt_dir
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_model_predictions(self, model_name, val_dir, max_files=None):
        """특정 모델의 가중치를 불러와 검증 데이터셋에 대한 예측 확률 반환"""
        ckpt_path = os.path.join(self.ckpt_dir, f"best_{model_name}.pt")
        if not os.path.exists(ckpt_path):
            print(f"⚠️ 체크포인트가 없습니다: {ckpt_path}")
            return None, None

        cfg = MODEL_REGISTRY.get(model_name, {})
        val_dataset = UniversalSpeakerDataset(
            val_dir,
            model_name=model_name,
            config_dict=cfg,
            is_train=False,
            max_files=max_files,
            sample_rate=self.sample_rate
        )
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

        model = build_model(model_name, num_classes=2, pretrained=False)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        all_probs = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_targets.extend(targets.numpy())

        return np.array(all_probs), np.array(all_targets)

    def evaluate_soft_voting(self, model_names, val_dir, weights=None, max_files=None):
        """
        선택된 모델들의 예측 확률을 Soft Voting으로 결합하여 앙상블 메트릭 산출
        """
        prob_matrix = []
        targets = None

        valid_models = []
        for name in model_names:
            p, t = self.get_model_predictions(name, val_dir, max_files)
            if p is not None:
                prob_matrix.append(p)
                valid_models.append(name)
                if targets is None:
                    targets = t

        if not prob_matrix:
            print("❌ 앙상블을 수행할 유효한 모델 체크포인트가 없습니다.")
            return None

        prob_matrix = np.array(prob_matrix) # (num_models, N)
        num_valid = len(valid_models)

        if weights is None:
            # 기본값: 균등 가중치
            w = np.ones(num_valid) / num_valid
        else:
            w = np.array(weights) / np.sum(weights)

        ensemble_probs = np.tensordot(w, prob_matrix, axes=(0, 0))
        ensemble_preds = (ensemble_probs >= 0.5).astype(int)

        metrics = compute_all_metrics(targets, ensemble_preds, ensemble_probs)
        print(f"\n✨ [Soft Voting 앙상블 결과] (모델: {', '.join(valid_models)})")
        print(f"  👉 Val Accuracy : {metrics['accuracy']:.2f}%")
        print(f"  👉 Macro F1     : {metrics['macro_f1']:.4f}")
        return metrics
