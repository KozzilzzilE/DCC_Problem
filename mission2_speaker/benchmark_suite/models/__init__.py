from .resnet50 import AudioResNet50
from .ecapa_tdnn import ECAPA_TDNN
from .redimnet import ReDimNet2_B2
from .campp import CAMPPlus
from .hubert import HuBERTClassifier
from .ssast import SSAST_Tiny

def build_model(model_name: str, num_classes: int = 2, pretrained: bool = True):
    """
    지정된 모델명에 맞춰 초기화된 모델 인스턴스를 반환합니다.
    """
    model_name = model_name.lower()
    if model_name == "resnet50":
        return AudioResNet50(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "ecapa_tdnn":
        return ECAPA_TDNN(in_channels=80, channels=512, num_classes=num_classes)
    elif model_name == "redimnet":
        return ReDimNet2_B2(num_classes=num_classes)
    elif model_name == "campp":
        return CAMPPlus(in_channels=80, num_classes=num_classes)
    elif model_name == "hubert":
        return HuBERTClassifier(num_classes=num_classes)
    elif model_name == "ssast":
        return SSAST_Tiny(num_classes=num_classes)
    else:
        raise ValueError(f"알 수 없는 모델 이름입니다: {model_name}. 지원 모델: resnet50, ecapa_tdnn, redimnet, campp, hubert, ssast")

def count_parameters(model):
    """학습 가능한 총 파라미터 수를 반환합니다."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
