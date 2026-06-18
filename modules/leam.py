import torch
import torch.nn as nn


class LEAM(nn.Module):
    """
    Layer-wise Expression Attention Mask
    w+の各層に学習可能な重みを掛けることで、
    表情認識に寄与する層を強調し、不要な層を抑制する。

    Args:
        num_layers (int): w+のシーケンス長。pSpの場合は18。
        init_coarse (float): 浅層（層1〜4）の初期重み。デフォルト0.5。
        init_fine (float): 深層（層13〜18）の初期重み。デフォルト0.5。
    """

    def __init__(
        self,
        num_layers: int = 18,
        init_coarse: float = 0.5,
        init_fine: float = 0.5,
    ):
        super().__init__()

        # 初期値：中間層=1.0、浅層・深層=0.5
        init = torch.ones(num_layers)
        init[:4] = init_coarse   # Coarse層（層1〜4）
        init[12:] = init_fine    # Fine層（層13〜18）
        self.layer_weights = nn.Parameter(init)

    def forward(self, w_plus: torch.Tensor) -> torch.Tensor:
        """
        Args:
            w_plus: (B, num_layers, latent_dim)
        Returns:
            (B, num_layers, latent_dim)
        """
        # sigmoid で0〜1に正規化してスケール
        weights = torch.sigmoid(self.layer_weights)          # (num_layers,)
        return w_plus * weights.unsqueeze(0).unsqueeze(-1)   # (B, L, D)

    def get_weights(self) -> torch.Tensor:
        """可視化用：学習済み重みを返す"""
        return torch.sigmoid(self.layer_weights).detach().cpu()
