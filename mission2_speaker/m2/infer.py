"""Mission 2 제출용 추론 파이프라인.

대회 규칙:
    python inference.py --audio_dir <wav 폴더> --label_dir <json 폴더> \
                        --ckpt_path <ckpt_path> --output ./outputs/mission2.csv

1. 입력: 16kHz mono WAV 오디오, JSON 라벨 (startAt/endAt 발화 구간만 사용)
2. 전처리:
   - 3.0초 고정 윈도우 (Zero Padding / Center Crop, 48,000 샘플)
   - Mel-Spectrogram: ReDimNet/ECAPA (80-mel, n_fft=512, hop=160), ResNet-50 (128-mel, n_fft=2048, hop=512)
   - 데시벨 변환 후 정규화: (mel_db + 80.0) / 80.0, [0.0, 1.0] 클리핑
3. 모델 추론 및 앙상블:
   - 3대 챔피언 모델 (ECAPA-TDNN, ReDimNet2-B2, AudioResNet-50)
   - 대회 규정 준수 사전 균등 가중치: (p_ecapa + p_redim + p_resnet) / 3.0 (Soft Voting)
   - 결정 임계값: 0.50 고정 (대회 공통 FAQ 규정 준수)
4. 출력 CSV:
   [audio file name], [startAt], [endAt], [speaker]  (speaker: 0 또는 1)
"""
from __future__ import annotations

import os
import sys
import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

OUTPUT_COLUMNS = ["audio file name", "startAt", "endAt", "speaker"]
DECISION_THRESHOLD = 0.50


def build_audio_resnet50(dropout_rate: float = 0.3) -> nn.Module:
    """단일 채널 AudioResNet-50 모델 생성."""
    from torchvision.models import resnet50
    model = resnet50(weights=None)
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        1, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, 1)
    )
    return model


def load_model_weights(model: nn.Module, ckpt_path: Union[str, Path], device: torch.device) -> nn.Module:
    """체크포인트 텐서를 모델에 유연하고 엄격하게 로드."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        m_dict = model.state_dict()
        
        # 키 네이밍 접두사 매핑 대응 (backbone., resnet., model. 등)
        cleaned_sd = {}
        for k, v in sd.items():
            clean_k = k
            if clean_k.startswith("backbone."):
                clean_k = clean_k.replace("backbone.", "")
            if clean_k.startswith("model."):
                clean_k = clean_k.replace("model.", "resnet.")
            cleaned_sd[clean_k] = v

        matched = {k: v for k, v in cleaned_sd.items() if k in m_dict and v.shape == m_dict[k].shape}
        if len(matched) == 0:
            # resnet 직접 매핑 시도
            matched = {k: v for k, v in sd.items() if hasattr(model, "resnet") and k in model.resnet.state_dict() and v.shape == model.resnet.state_dict()[k].shape}
            if len(matched) > 0:
                model.resnet.load_state_dict(matched)
                model.eval()
                return model

        m_dict.update(matched)
        model.load_state_dict(m_dict)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"가중치 파일 로드 실패 ({ckpt_path}): {e}") from e


class Mission2InferenceEngine:
    """3대장 앙상블 및 단일 모델 추론 엔진."""
    def __init__(self, ckpt_path: Union[str, Path], device: torch.device | None = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        ckpt_path = Path(ckpt_path).resolve()
        self.models = {}

        # benchmark_suite 모듈 임포트
        repo_root = Path(__file__).resolve().parents[2]
        m2_root = Path(__file__).resolve().parents[1]
        for p in [str(m2_root), str(repo_root)]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from benchmark_suite.models.redimnet import ReDimNet2_B2
        from benchmark_suite.models.ecapa_tdnn import ECAPA_TDNN

        # 체크포인트 경로 분석 (폴더인지, 단일 파일인지)
        if ckpt_path.is_dir():
            ckpt_dir = ckpt_path
        else:
            ckpt_dir = ckpt_path.parent

        p_redim = ckpt_dir / "best_redimnet.pt"
        p_ecapa = ckpt_dir / "best_ecapa_tdnn.pt"
        p_resnet = ckpt_dir / "best_resnet50.pt"

        # 3개 파일이 모두 존재하는지 확인
        if p_redim.exists() and p_ecapa.exists() and p_resnet.exists():
            print(f"🚀 [Mission 2] 3대장 앙상블 모드로 초기화합니다. (위치: {ckpt_dir})")
            m_r = ReDimNet2_B2(num_classes=1).to(self.device)
            m_e = ECAPA_TDNN(in_channels=80, channels=512, num_classes=1).to(self.device)
            m_res = build_audio_resnet50().to(self.device)

            self.models["redimnet"] = load_model_weights(m_r, p_redim, self.device)
            self.models["ecapa_tdnn"] = load_model_weights(m_e, p_ecapa, self.device)
            self.models["resnet50"] = load_model_weights(m_res, p_resnet, self.device)
            self.is_ensemble = True
            print("✅ 3대 모델 가중치 로드 완료 (ReDimNet2-B2, ECAPA-TDNN, AudioResNet-50)")
        elif ckpt_path.is_file():
            print(f"📦 [Mission 2] 단일 모델 모드로 초기화합니다: {ckpt_path.name}")
            # 단일 모델 로드
            fname = ckpt_path.name.lower()
            if "redim" in fname:
                m = ReDimNet2_B2(num_classes=1).to(self.device)
                self.models["redimnet"] = load_model_weights(m, ckpt_path, self.device)
            elif "ecapa" in fname:
                m = ECAPA_TDNN(in_channels=80, channels=512, num_classes=1).to(self.device)
                self.models["ecapa_tdnn"] = load_model_weights(m, ckpt_path, self.device)
            else:
                m = build_audio_resnet50().to(self.device)
                self.models["resnet50"] = load_model_weights(m, ckpt_path, self.device)
            self.is_ensemble = False
            print(f"✅ 단일 모델 가중치 로드 완료: {list(self.models.keys())[0]}")
        else:
            raise FileNotFoundError(f"유효한 체크포인트 파일 또는 디렉토리를 찾을 수 없습니다: {ckpt_path}")

    def extract_features(self, clip_wav: np.ndarray, sr: int = 16000) -> Dict[str, torch.Tensor]:
        """학습(Local_Light_Train.ipynb)과 100% 동일한 3.0초 윈도우 및 (dB+80)/80 정규화."""
        import librosa

        target_samples = int(sr * 3.0)  # 3.0초 = 48,000 샘플
        cur_len = len(clip_wav)

        if cur_len < target_samples:
            y = np.pad(clip_wav, (0, target_samples - cur_len), mode="constant")
        elif cur_len > target_samples:
            start_idx = (cur_len - target_samples) // 2
            y = clip_wav[start_idx : start_idx + target_samples]
        else:
            y = clip_wav

        features = {}

        # 80-mel (ReDimNet & ECAPA용)
        if "redimnet" in self.models or "ecapa_tdnn" in self.models:
            y80 = np.pad(y, (0, 512 - len(y)), mode="constant") if len(y) < 512 else y
            m80 = librosa.feature.melspectrogram(y=y80, sr=sr, n_fft=512, hop_length=160, n_mels=80)
            m80_db = librosa.power_to_db(m80, ref=np.max)
            m80_norm = np.clip((m80_db + 80.0) / 80.0, 0.0, 1.0)
            features["80"] = torch.tensor(m80_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        # 128-mel (ResNet-50용)
        if "resnet50" in self.models:
            y128 = np.pad(y, (0, 2048 - len(y)), mode="constant") if len(y) < 2048 else y
            m128 = librosa.feature.melspectrogram(y=y128, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
            m128_db = librosa.power_to_db(m128, ref=np.max)
            m128_norm = np.clip((m128_db + 80.0) / 80.0, 0.0, 1.0)
            features["128"] = torch.tensor(m128_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        return features

    @torch.no_grad()
    def predict_clip(self, clip_wav: np.ndarray, sr: int = 16000) -> float:
        """클립 음성에 대한 소프트 확률(0.0 ~ 1.0) 반환."""
        features = self.extract_features(clip_wav, sr=sr)
        probs = []

        if "redimnet" in self.models:
            out = self.models["redimnet"](features["80"])
            probs.append(torch.sigmoid(out).item())

        if "ecapa_tdnn" in self.models:
            out = self.models["ecapa_tdnn"](features["80"])
            probs.append(torch.sigmoid(out).item())

        if "resnet50" in self.models:
            out = self.models["resnet50"](features["128"])
            probs.append(torch.sigmoid(out).item())

        # 규정 준수: 사전 균등 가중치 (1/3씩, 단순 평균)
        return float(np.mean(probs))


def predict_directory(
    audio_dir: Union[str, Path],
    label_dir: Union[str, Path],
    ckpt_path: Union[str, Path]
) -> pd.DataFrame:
    """오디오/라벨 디렉토리 전체 추론 함수."""
    import librosa

    audio_dir = Path(audio_dir)
    label_dir = Path(label_dir)
    ckpt_path = Path(ckpt_path)

    if not audio_dir.exists():
        raise FileNotFoundError(f"오디오 폴더가 존재하지 않습니다: {audio_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"라벨 폴더가 존재하지 않습니다: {label_dir}")

    engine = Mission2InferenceEngine(ckpt_path)

    json_files = sorted(list(label_dir.glob("*.json")))
    if not json_files:
        json_files = sorted(list(label_dir.glob("**/*.json")))

    if not json_files:
        raise FileNotFoundError(f"라벨 폴더에서 JSON 파일을 찾을 수 없습니다: {label_dir}")

    results = []
    correct_count = 0
    total_count = 0
    sr = 16000

    print(f"📂 총 {len(json_files)}개 통화 파일에 대해 추론을 진행합니다...")

    for jf in tqdm(json_files, desc="Mission 2 추론 진행 중"):
        wav_name = jf.stem + ".wav"
        wav_path = audio_dir / wav_name
        if not wav_path.exists():
            # 하위 폴더 재탐색
            found = list(audio_dir.glob(f"**/{wav_name}"))
            if found:
                wav_path = found[0]
            else:
                continue

        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        try:
            full_audio, _ = librosa.load(str(wav_path), sr=sr)
        except Exception as e:
            print(f"⚠️ 오디오 로드 실패 ({wav_name}): {e}")
            continue

        dialog_list = data.get("utterances") or data.get("dialogs") or data.get("dialogue") or []

        for utt in dialog_list:
            start_ms = utt.get("startAt") if "startAt" in utt else utt.get("start_time", 0)
            end_ms = utt.get("endAt") if "endAt" in utt else utt.get("end_time", 0)

            # 초 단위 변환 보정
            if start_ms < 100 and end_ms < 100 and (end_ms - start_ms) > 0.05:
                start_ms = int(start_ms * 1000)
                end_ms = int(end_ms * 1000)
            else:
                start_ms = int(start_ms)
                end_ms = int(end_ms)

            start_sample = int((start_ms / 1000.0) * sr)
            end_sample = int((end_ms / 1000.0) * sr)
            clip = full_audio[start_sample:end_sample]

            # 모델 확률 추론
            avg_prob = engine.predict_clip(clip, sr=sr)

            # 규정 준수: 0.50 고정 결정 임계값
            final_pred = 1 if avg_prob >= DECISION_THRESHOLD else 0

            if "speaker" in utt:
                actual = int(utt["speaker"])
                if final_pred == actual:
                    correct_count += 1
                total_count += 1

            results.append({
                "audio file name": wav_name,
                "startAt": start_ms,
                "endAt": end_ms,
                "speaker": final_pred
            })

    if total_count > 0:
        accuracy = (correct_count / total_count) * 100.0
        print(f"\n[평가 결과] [평가 결과] 총 {total_count:,}개 발화 중 {correct_count:,}개 정답!")
        print(f"정확도 (Accuracy): {accuracy:.2f}% (결정 임계값 {DECISION_THRESHOLD} 기준)")
    else:
        print("\n[안내] 정답(speaker) 라벨이 제공되지 않아 정확도 계산을 생략합니다.")

    return pd.DataFrame(results, columns=OUTPUT_COLUMNS)
