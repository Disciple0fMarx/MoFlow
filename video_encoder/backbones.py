"""Pluggable, frozen CNN backbones that produce a per-frame feature vector."""
from __future__ import annotations

from typing import Callable
import torch
from torch import nn


def _resnet(name: str) -> tuple[nn.Module, int]:
    import torchvision
    from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

    if name == "resnet18":
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        dim = 512
    elif name == "resnet50":
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        dim = 2048
    else:
        raise ValueError(name)
    m.fc = nn.Identity()
    return m, dim


def build_backbone(name: str = "resnet18") -> tuple[nn.Module, int, Callable]:
    """Return (model, feature_dim, preprocess_fn).

    preprocess_fn takes a PIL.Image and returns a normalized 3xHxW float tensor.
    """
    from torchvision import transforms

    if name in ("resnet18", "resnet50"):
        model, dim = _resnet(name)
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        raise ValueError(
            f"Unknown backbone '{name}'. Supported: resnet18, resnet50."
        )

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, dim, preprocess
