import torch
import torch.nn as nn


class LayerWiseNorm(nn.Module):
    """
    Layer-wise Normalization
    w+の各層に独立したLayerNormを適用する。
    通常のViTが全トークン共通のLayerNormを使うのに対し、
    w+の層別意味構造の違いを考慮した正規化を行う。

    Args:
        num_layers (int): w+のシーケンス長。pSpの場合は18。
        d_model (int): 潜在ベクトルの次元数。pSpの場合は512。
    """

    def __init__(self, num_layers: int = 18, d_model: int = 512):
        super().__init__()
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

    def forward(self, w_plus: torch.Tensor) -> torch.Tensor:
        """
        Args:
            w_plus: (B, num_layers, d_model)
        Returns:
            (B, num_layers, d_model)
        """
        out = torch.stack(
            [self.norms[i](w_plus[:, i, :]) for i in range(len(self.norms))],
            dim=1
        )
        return out  # (B, L, D)
