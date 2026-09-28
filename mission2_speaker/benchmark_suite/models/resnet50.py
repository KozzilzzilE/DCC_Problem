import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class AudioResNet50(nn.Module):
    """
    1-Channel Mel-Spectrogram을 위한 ResNet-50 아키텍처.
    ImageNet의 3채널 사전학습 가중치를 1채널 평균(Mean Blending)으로 초기화하여 전이 학습 지식을 보존합니다.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)

        # 첫 번째 Conv2d를 1채널 입력으로 개조
        orig_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False
        )

        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))

        # 최종 분류 헤드
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        # x: (batch_size, 1, n_mels, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.backbone(x)

    def get_embedding(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)
