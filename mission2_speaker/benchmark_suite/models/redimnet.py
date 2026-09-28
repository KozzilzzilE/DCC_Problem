import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2DFrontEnd(nn.Module):
    """
    국소적 음향 특징(Pitch, Formants)을 추출하기 위한 2D Conv 프론트엔드.
    speaker_diarization_analysis.md 원칙 준수: '국소 특징을 뭉개지 않고 정밀하게 보존'
    """
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, 1, F=80, T)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x))) # F: 80 -> 40
        return x

class ReDimBlock1D(nn.Module):
    """1D Residual Conv 블록 (시간축 장기 문맥 수집)"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)

class ReDimNet2_B2(nn.Module):
    """
    [분석 문서 1순위 추천] ReDimNet2-B2 경량 혼합 아키텍처 (~3.6M 파라미터).
    - 2D Conv 프론트엔드로 Pitch/Formant 국소 단서 정밀 추출
    - 1D Dilated Conv 스택으로 화자 특성 압축
    - Multi-Head Attention 기반 Time Pooling으로 가중 집계
    """
    def __init__(self, num_classes=2, emb_dim=192):
        super().__init__()
        self.frontend = Conv2DFrontEnd(in_channels=1, out_channels=32)
        
        # 2D -> 1D 차원 축소: 64 채널 * 40 Freq bins = 2560 -> 256차원으로 프로젝션
        self.proj = nn.Sequential(
            nn.Conv1d(64 * 40, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # 1D Residual 스택
        self.layers = nn.Sequential(
            ReDimBlock1D(256),
            ReDimBlock1D(256),
            ReDimBlock1D(256),
            ReDimBlock1D(256)
        )

        # Multi-Head Attention Time Pooling (8 heads)
        self.mha_pool = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, 256))

        # 화자 임베딩 레이어
        self.fc_emb = nn.Sequential(
            nn.Linear(256, emb_dim),
            nn.BatchNorm1d(emb_dim)
        )

        # 최종 분류기
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, 80, T) or (B, 1, 80, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        B, C, F_bins, T_steps = x.shape

        # 1. 2D 국소 특징 추출
        feat2d = self.frontend(x) # (B, 64, 40, T)
        B, C2, F2, T2 = feat2d.shape
        feat1d = feat2d.view(B, C2 * F2, T2) # (B, 2560, T)

        # 2. 1D 장기 시퀀스 학습
        feat1d = self.proj(feat1d) # (B, 256, T)
        feat1d = self.layers(feat1d) # (B, 256, T)

        # 3. Multi-Head Attention Pooling
        seq = feat1d.transpose(1, 2) # (B, T, 256)
        q = self.query.expand(B, -1, -1) # (B, 1, 256)
        attn_out, _ = self.mha_pool(q, seq, seq) # (B, 1, 256)
        pooled = attn_out.squeeze(1) # (B, 256)

        # 4. 임베딩 및 분류
        emb = self.fc_emb(pooled) # (B, 192)
        logits = self.classifier(emb)
        return logits

    def get_embedding(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        B, C, F_bins, T_steps = x.shape
        feat2d = self.frontend(x)
        B, C2, F2, T2 = feat2d.shape
        feat1d = feat2d.view(B, C2 * F2, T2)
        feat1d = self.proj(feat1d)
        feat1d = self.layers(feat1d)
        seq = feat1d.transpose(1, 2)
        q = self.query.expand(B, -1, -1)
        attn_out, _ = self.mha_pool(q, seq, seq)
        pooled = attn_out.squeeze(1)
        return self.fc_emb(pooled)
