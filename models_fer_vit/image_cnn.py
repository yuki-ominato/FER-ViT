"""
2D Image CNN for facial expression recognition.
Uses torchvision ResNet as backbone with ImageNet pretraining option.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)


class ImageCNN2D(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = False,
        num_classes: int = 7,
        dropout: float = 0.0,
    ):
        """
        torchvision ResNet wrapper for FER classification.

        Args:
            backbone: "resnet18" | "resnet34" | "resnet50"
            pretrained: Load ImageNet1K weights. The original fc is replaced with
                        a randomly initialized head for num_classes (full fine-tuning).
            num_classes: Number of output emotion classes.
            dropout: Dropout probability inserted before the final linear layer.
        """
        super().__init__()

        if backbone == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet18(weights=weights)
        elif backbone == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet34(weights=weights)
        elif backbone == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet50(weights=weights)
        else:
            raise ValueError(
                f"Unknown backbone: {backbone!r}. Choose 'resnet18', 'resnet34', or 'resnet50'."
            )

        in_features = base.fc.in_features
        # Drop the original classification head; keep everything up to global avg-pool
        self.features = nn.Sequential(*list(base.children())[:-1])

        if dropout > 0.0:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) image tensor
        Returns:
            logits: (B, num_classes)
        """
        x = self.features(x)   # (B, C, 1, 1)
        x = x.flatten(1)       # (B, C)
        return self.head(x)    # (B, num_classes)
