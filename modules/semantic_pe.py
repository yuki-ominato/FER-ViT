import torch
import torch.nn as nn


# 各層が属するグループのID（Coarse=0, Medium=1, Fine=2）
_LAYER_GROUPS = [0, 0, 0, 0,          # 層1〜4:  Coarse（顔の骨格・全体構造）
                 1, 1, 1, 1, 1, 1, 1, 1,  # 層5〜12: Medium（表情・目・口の形状）
                 2, 2, 2, 2, 2, 2]        # 層13〜18: Fine（テクスチャ・肌色）


class SemanticPE(nn.Module):
    """
    Semantic Positional Encoding
    StyleGANのw+層が持つCoarse/Medium/Fineの階層構造を
    学習可能な位置ベクトルとして付与する。

    グループベクトル（3種）と層個別ベクトル（18種）の和を
    各トークンに加算する。

    Args:
        d_model (int): 潜在ベクトルの次元数。pSpの場合は512。
        num_layers (int): w+のシーケンス長。pSpの場合は18。
    """

    def __init__(self, d_model: int = 512, num_layers: int = 18):
        super().__init__()
        self.group_embed = nn.Embedding(3, d_model)   # Coarse/Medium/Fine
        self.layer_embed = nn.Embedding(num_layers, d_model)

        # グループIDをバッファとして登録（学習不要・デバイス追従）
        self.register_buffer(
            "groups",
            torch.tensor(_LAYER_GROUPS, dtype=torch.long)
        )

    def forward(self, w_plus: torch.Tensor) -> torch.Tensor:
        """
        Args:
            w_plus: (B, num_layers, d_model)
        Returns:
            (B, num_layers, d_model)
        """
        device = w_plus.device
        layers = torch.arange(w_plus.size(1), device=device)  # (L,)

        # グループ埋め込み + 層個別埋め込み
        pe = self.group_embed(self.groups) + self.layer_embed(layers)  # (L, D)
        return w_plus + pe.unsqueeze(0)   # (B, L, D)
