"""
Mission 2: Speaker Classification Architecture Definitions.
- ReDimNet2_B2: 2D+1D Hybrid CNN with Multi-Head Attention Pooling (~2.57M params)
- ECAPA_TDNN: Multi-scale 1D Res2Net with Attentive Statistics Pooling (~5.80M params)
- AudioResNet: 1-Channel Modified ResNet-50 (~23.50M params)

Total learnable parameters for three num_classes=1 models: 31,866,563 (~31.87M)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. AudioResNet (1-Channel ResNet-50)
# =====================================================================
class AudioResNet(nn.Module):
    """
    1-Channel 2D Spectrogram Texture Classifier based on ResNet-50.
    Modified first convolution layer (3 channels -> 1 channel).
    """
    def __init__(self, dropout_rate=0.3, num_classes=1):
        super(AudioResNet, self).__init__()
        from torchvision import models

        self.resnet = models.resnet50(weights=None)
        old_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, n_mels=128, T)
        return self.resnet(x)


# =====================================================================
# 2. ReDimNet2_B2
# =====================================================================
class Conv2DFrontEnd(nn.Module):
    """2D Convolution Front-End for Local Acoustic Feature Extraction (Pitch, Formants)."""
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
        x = self.relu(self.bn2(self.conv2(x)))  # F: 80 -> 40
        return x


class ReDimBlock1D(nn.Module):
    """1D Residual Convolution Block for temporal sequence modeling."""
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
    ReDimNet2-B2 Lightweight Hybrid Architecture (~2.57M params).
    Input: Log Mel-Filterbank (B, 1, 80, T) or (B, 80, T)
    Output: 2-class logits (0: Dispatcher, 1: Caller)
    """
    def __init__(self, num_classes=2, emb_dim=192):
        super().__init__()
        self.frontend = Conv2DFrontEnd(in_channels=1, out_channels=32)

        self.proj = nn.Sequential(
            nn.Conv1d(64 * 40, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.layers = nn.Sequential(
            ReDimBlock1D(256),
            ReDimBlock1D(256),
            ReDimBlock1D(256),
            ReDimBlock1D(256)
        )

        self.mha_pool = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, 256))

        self.fc_emb = nn.Sequential(
            nn.Linear(256, emb_dim),
            nn.BatchNorm1d(emb_dim)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x):
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

        emb = self.fc_emb(pooled)
        logits = self.classifier(emb)
        return logits


# =====================================================================
# 3. ECAPA_TDNN
# =====================================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention module."""
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, bottleneck, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(bottleneck, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = x.mean(dim=-1, keepdim=True)
        s = self.relu(self.fc1(s))
        s = self.sigmoid(self.fc2(s))
        return x * s


class Res2NetBlock(nn.Module):
    """Multi-scale 1D Res2Net block with SE module."""
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
    """Attentive Statistics Pooling (mean + std)."""
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
        alpha = self.attn(x)
        mean = torch.sum(alpha * x, dim=-1)
        residuals = torch.sum(alpha * (x ** 2), dim=-1) - (mean ** 2)
        std = torch.sqrt(torch.clamp(residuals, min=1e-5))
        return torch.cat([mean, std], dim=-1)


class ECAPA_TDNN(nn.Module):
    """
    ECAPA-TDNN Speaker Architecture (~5.80M params).
    Input: Log Mel-Filterbank (B, 80, T) or (B, 1, 80, T)
    Output: 2-class logits (0: Dispatcher, 1: Caller)
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
        if x.dim() == 4:
            x = x.squeeze(1)
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
