import time
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def compute_all_metrics(y_true, y_pred, y_probs=None):
    """
    모든 다차원 평가 지표를 일괄 계산합니다.
    - Accuracy, Macro F1, Precision, Recall
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred) * 100.0
    f1 = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm.tolist()
    }

def measure_inference_speed(model, sample_input, device, num_warmup=10, num_runs=50):
    """
    단일 샘플당 평균 추론 지연시간(Latency, ms)을 정밀 측정합니다.
    """
    model.eval()
    sample_input = sample_input.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(sample_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(sample_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    total_time = (time.time() - start_time) * 1000.0 # ms
    ms_per_sample = total_time / num_runs
    return round(ms_per_sample, 2)
