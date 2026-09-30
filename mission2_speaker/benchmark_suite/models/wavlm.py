import torch
import torch.nn as nn

class WavLMClassifier(nn.Module):
    """
    WavLM (Microsoft WavLM-Base+) 기반 음성 화자 분류기.
    입력: Raw Waveform 1D 텐서 (B, samples=48000)
    대규모 화자(Speaker) 인식 사전학습 가중치 활용 + Mean Pooling + Linear 분류 헤드
    """
    def __init__(self, model_name_or_path="microsoft/wavlm-base-plus", num_classes=2, freeze_feature_extractor=True):
        super().__init__()
        try:
            from transformers import WavLMModel
            self.wavlm = WavLMModel.from_pretrained(model_name_or_path)
            hidden_size = self.wavlm.config.hidden_size
            
            # CNN 특징 추출기(Waveform -> Latent) 고정하여 VRAM 및 연산량 대폭 절약
            if freeze_feature_extractor:
                self.wavlm.feature_extractor._freeze_parameters()
                
            self.is_hf = True
        except Exception:
            # HuggingFace 미설치 또는 오프라인 환경을 위한 자체 1D CNN+Transformer fallback
            self.is_hf = False
            hidden_size = 256
            self.fallback_encoder = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=10, stride=5),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=8, stride=4),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 256, kernel_size=4, stride=2),
                nn.BatchNorm1d(256),
                nn.ReLU()
            )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, samples) or (B, 1, samples)
        if x.dim() == 3:
            x = x.squeeze(1)
            
        if self.is_hf:
            outputs = self.wavlm(x)
            hidden_states = outputs.last_hidden_state # (B, T, hidden_size)
            pooled = torch.mean(hidden_states, dim=1) # Global Average Pooling
        else:
            x_in = x.unsqueeze(1) # (B, 1, samples)
            feats = self.fallback_encoder(x_in) # (B, 256, T)
            pooled = torch.mean(feats, dim=-1) # (B, 256)

        logits = self.classifier(pooled)
        return logits

    def get_embedding(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        if self.is_hf:
            outputs = self.wavlm(x)
            return torch.mean(outputs.last_hidden_state, dim=1)
        else:
            return torch.mean(self.fallback_encoder(x.unsqueeze(1)), dim=-1)
