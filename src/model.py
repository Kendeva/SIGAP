"""Shared model factory for SIGAP training and local inference."""
from __future__ import annotations

from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

ARCHITECTURE = "mobilenet_v3_small"


def build_model(num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    """Create MobileNetV3-Small and replace its final classifier layer."""
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model
