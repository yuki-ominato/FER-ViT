import torch
import torch.nn as nn


class LayerWiseNorm(nn.Module):
    """
    Layer-wise Normalization
    w+の各層に独立したLayerNormを適用する。

    use_residual=True の場合、元のw+と正規化済み出力を
    学習可能なゲートで補間する（残差ゲート方式）。
    これによりスケール情報を保ちながら正規化の効果も得られる。

        out = w+ + sigmoid(gate) * (norm(w+) - w+)

    gate の初期値が -2.0 のため sigmoid(-2) ≈ 0.12 となり、
    学習初期はほぼ元のw+を維持する。

    Args:
        num_layers (int): w+のシーケンス長。pSpの場合は18。
        d_model (int): 潜在ベクトルの次元数。pSpの場合は512。
        use_residual (bool): 残差ゲート方式を使用するか。デフォルトFalse。
    """

    def __init__(self, num_layers: int = 18, d_model: int = 512, use_residual: bool = False):
        super().__init__()
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        self.use_residual = use_residual
        if use_residual:
            # 初期値 -2.0 → sigmoid(-2) ≈ 0.12（学習初期は元のw+に近い挙動）
            self.gate = nn.Parameter(torch.full((num_layers,), -2.0))

    def forward(self, w_plus: torch.Tensor) -> torch.Tensor:
        """
        Args:
            w_plus: (B, num_layers, d_model)
        Returns:
            (B, num_layers, d_model)
        """
        normed = torch.stack(
            [self.norms[i](w_plus[:, i, :]) for i in range(len(self.norms))],
            dim=1
        )  # (B, L, D)

        if self.use_residual:
            gate = torch.sigmoid(self.gate).unsqueeze(0).unsqueeze(-1)  # (1, L, 1)
            return w_plus + gate * (normed - w_plus)
        return normed
