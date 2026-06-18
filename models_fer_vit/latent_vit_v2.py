import torch
import torch.nn as nn
from .latent_vit import LatentViT
from modules import LEAM, SemanticPE, LayerWiseNorm


class LatentViTv2(nn.Module):
    """
    LatentViT v2：前処理モジュール統合版

    既存のLatentViTに以下のモジュールをオプションで追加する。
        - use_lwn  : LayerWiseNorm（層別正規化）
        - use_spe  : SemanticPE（意味的位置エンコーディング）
        - use_leam : LEAM（層別アテンションマスク）

    前処理の適用順：
        w+ → [LayerWiseNorm] → [SemanticPE] → [LEAM] → LatentViT

    Args:
        latent_dim (int): w+の次元数。デフォルト512。
        seq_len (int): w+のシーケンス長。デフォルト18。
        embed_dim (int): Transformer内部次元数。デフォルト512。
        depth (int): Transformerのブロック数。デフォルト6。
        heads (int): Attentionヘッド数。デフォルト8。
        mlp_dim (int): FFNの中間次元数。デフォルト2048。
        num_classes (int): 分類クラス数。デフォルト7。
        dropout (float): ドロップアウト率。デフォルト0.1。
        use_lwn (bool): LayerWiseNormを使用するか。デフォルトFalse。
        use_spe (bool): SemanticPEを使用するか。デフォルトFalse。
        use_leam (bool): LEAMを使用するか。デフォルトFalse。
    """

    def __init__(
        self,
        latent_dim: int = 512,
        seq_len: int = 18,
        embed_dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 2048,
        num_classes: int = 7,
        dropout: float = 0.1,
        use_lwn: bool = False,
        use_spe: bool = False,
        use_leam: bool = False,
    ):
        super().__init__()

        # 前処理モジュール（オプション）
        self.lwn  = LayerWiseNorm(seq_len, latent_dim) if use_lwn  else nn.Identity()
        self.spe  = SemanticPE(latent_dim, seq_len)    if use_spe  else nn.Identity()
        self.leam = LEAM(seq_len)                      if use_leam else nn.Identity()

        # 既存のLatentViTをバックボーンとして使用
        self.backbone = LatentViT(
            latent_dim=latent_dim,
            seq_len=seq_len,
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        # フラグを保持（ログ・デバッグ用）
        self.use_lwn  = use_lwn
        self.use_spe  = use_spe
        self.use_leam = use_leam

    def forward(self, w_plus: torch.Tensor) -> torch.Tensor:
        """
        Args:
            w_plus: (B, seq_len, latent_dim)
        Returns:
            logits: (B, num_classes)
        """
        x = self.lwn(w_plus)   # LayerWiseNorm
        x = self.spe(x)        # SemanticPE
        x = self.leam(x)       # LEAM
        return self.backbone(x)

    def get_leam_weights(self) -> torch.Tensor:
        """LEAM重みを取得（可視化用）。use_leam=Falseの場合はNoneを返す。"""
        if self.use_leam:
            return self.leam.get_weights()
        return None

    def get_config(self) -> dict:
        """実験ログ用にモデル設定を返す"""
        return {
            "model": "LatentViTv2",
            "use_lwn": self.use_lwn,
            "use_spe": self.use_spe,
            "use_leam": self.use_leam,
        }
