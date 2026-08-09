import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

CLASSES = [
    "boeing_737",
    "boeing_747",
    "boeing_777",
    "airbus_a320",
    "airbus_a330",
    "airbus_a350",
]

DISPLAY_NAMES = {
    "boeing_737": "Boeing 737",
    "boeing_747": "Boeing 747",
    "boeing_777": "Boeing 777",
    "airbus_a320": "Airbus A320",
    "airbus_a330": "Airbus A330",
    "airbus_a350": "Airbus A350",
}


def build_model(num_classes: int = len(CLASSES), pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False