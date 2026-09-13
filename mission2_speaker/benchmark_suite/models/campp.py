import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextAwareMasking(nn.Module):
    """CAM++ 핵심 모듈: 국소 특징 추출 후 문맥 마스킹을 통한 유효 특징 선별"""
    def __init__(self, channels):
        super().__init__()
        self.conv_context = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T)
        mask = self.conv_context(x)
        return x * mask

class DenseTDNNLayer(nn.Module):
    """Dense 결합 TDNN 레이어"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=pad)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return torch.cat([x, out], dim=1)

class CAMPPlus(nn.Module):
    """
    [분석 문서 비교 기준선 1] CAM++ 아키텍처 (~7M 파라미터).
    - 2D Conv 프론트엔드로 성도/공명 국소 특징 보존
    - Dense TDNN 스택 + Context-Aware Masking (CAM)
    - Attentive Pooling 및 화자 분류
    """
    def __init__(self, in_channels=80, num_classes=2, emb_dim=192):
        super().__init__()
        # 1. 2D 프론트엔드
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(1, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 2. 1D 변환 및 Dense 스택
        self.trans = nn.Sequential(
            nn.Conv1d(64 * 40, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.dense1 = DenseTDNNLayer(256, 64, kernel_size=3, dilation=1)
        self.cam1 = ContextAwareMasking(256 + 64)
        self.dense2 = DenseTDNNLayer(256 + 64, 64, kernel_size=3, dilation=2)
        self.cam2 = ContextAwareMasking(256 + 128)

        total_channels = 256 + 128
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_emb = nn.Sequential(
            nn.Linear(total_channels, emb_dim),
            nn.BatchNorm1d(emb_dim)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, 80, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv2d(x) # (B, 64, 40, T)
        B, C, F_dim, T_dim = x.shape
        x = x.view(B, C * F_dim, T_dim)
        x = self.trans(x)

        x = self.cam1(self.dense1(x))
        x = self.cam2(self.dense2(x))

        pooled = self.pool(x).squeeze(-1)
        emb = self.fc_emb(pooled)
        return self.classifier(emb)

    def get_embedding(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv2d(x)
        B, C, F_dim, T_dim = x.shape
        x = x.view(B, C * F_dim, T_dim)
        x = self.trans(x)
        x = self.cam1(self.dense1(x))
        x = self.cam2(self.dense2(x))
        pooled = self.pool(x).squeeze(-1)
        return self.fc_emb(pooled)
