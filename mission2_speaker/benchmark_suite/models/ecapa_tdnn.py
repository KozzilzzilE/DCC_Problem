import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 채널 어텐션 모듈"""
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, bottleneck, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(bottleneck, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, T)
        s = x.mean(dim=-1, keepdim=True)
        s = self.relu(self.fc1(s))
        s = self.sigmoid(self.fc2(s))
        return x * s

class Res2NetBlock(nn.Module):
    """멀티스케일 1D Res2Net 블록 + SE 모듈"""
    def __init__(self, channels, kernel_size=3, dilation=2, scale=8):
        super().__init__()
        self.scale = scale
        width = channels // scale
        self.width = width
        self.nums = scale - 1

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(channels)

        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation // 2)
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(self.nums)])

        self.conv3 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(channels)

        self.se = SEBlock(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.width, dim=1)

        sp = spx[0]
        out_chunks = [sp]
        for i in range(self.nums):
            if i == 0:
                sp = spx[i + 1]
            else:
                sp = sp + spx[i + 1]
            sp = self.relu(self.bns[i](self.convs[i](sp)))
            out_chunks.append(sp)

        out = torch.cat(out_chunks, dim=1)
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        return self.relu(out + residual)

class AttentiveStatsPool(nn.Module):
    """화자 특화 평균+표준편차 어텐션 통계 풀링"""
    def __init__(self, in_dim, bottleneck=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, in_dim, kernel_size=1),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x: (B, C, T)
        alpha = self.attn(x)
        mean = torch.sum(alpha * x, dim=-1)
        residuals = torch.sum(alpha * (x ** 2), dim=-1) - (mean ** 2)
        std = torch.sqrt(torch.clamp(residuals, min=1e-5))
        return torch.cat([mean, std], dim=-1) # (B, 2*C)

class ECAPA_TDNN(nn.Module):
    """
    화자 인식 표준 모델 ECAPA-TDNN (~6M 파라미터).
    입력: Log Mel-Filterbank (B, 80, T)
    출력: 2-class logits (0: 대원, 1: 신고자)
    """
    def __init__(self, in_channels=80, channels=512, emb_dim=192, num_classes=2):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(channels)
        )
        self.layer2 = Res2NetBlock(channels, kernel_size=3, dilation=2)
        self.layer3 = Res2NetBlock(channels, kernel_size=3, dilation=3)
        self.layer4 = Res2NetBlock(channels, kernel_size=3, dilation=4)

        self.mfa = nn.Sequential(
            nn.Conv1d(channels * 3, channels * 3, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(channels * 3)
        )
        self.asp = AttentiveStatsPool(channels * 3)
        self.fc = nn.Sequential(
            nn.Linear(channels * 6, emb_dim),
            nn.BatchNorm1d(emb_dim)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, C, T)
        if x.dim() == 4:
            x = x.squeeze(1) # (B, 1, C, T) -> (B, C, T)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        out = torch.cat([x2, x3, x4], dim=1)
        out = self.mfa(out)
        stats = self.asp(out)
        emb = self.fc(stats)
        logits = self.classifier(emb)
        return logits

    def get_embedding(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        out = torch.cat([x2, x3, x4], dim=1)
        out = self.mfa(out)
        stats = self.asp(out)
        return self.fc(stats)
