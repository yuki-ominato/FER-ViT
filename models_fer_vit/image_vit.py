"""
画像から直接学習する標準的なViT
FER2013などの画像データセット用
"""

import torch
import torch.nn as nn
from typing import Optional


class PatchEmbedding(nn.Module):
    """画像をパッチに分割して埋め込み"""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # 畳み込みでパッチ埋め込み
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, N, embed_dim) where N = n_patches
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, N)
        x = x.transpose(1, 2)  # (B, N, embed_dim)
        return x


class ImageViT(nn.Module):
    """
    標準的なVision Transformer
    画像から直接感情認識を行う
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_dim: int = 3072,
        num_classes: int = 7,
        dropout: float = 0.1,
    ):
        """
        Args:
            img_size: 入力画像サイズ
            patch_size: パッチサイズ
            in_channels: 入力チャンネル数
            embed_dim: 埋め込み次元
            depth: Transformerレイヤー数
            heads: アテンションヘッド数
            mlp_dim: MLPの隠れ層次元
            num_classes: 出力クラス数
            dropout: ドロップアウト率
        """
        super().__init__()
        
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # パッチ埋め込み
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        
        # CLSトークン
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 位置埋め込み（CLS + パッチ数）
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches + 1, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False,  # Post-norm（標準ViT）
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )
        
        # 分類ヘッド
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 重み初期化
        self._init_weights()
    
    def _init_weights(self):
        """重みの初期化"""
        # 位置埋め込みとCLSトークンの初期化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 線形層の初期化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 画像テンソル
        Returns:
            (B, num_classes) ロジット
        """
        B = x.shape[0]
        
        # パッチ埋め込み
        x = self.patch_embed(x)  # (B, N, embed_dim)
        
        # CLSトークンを追加
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, embed_dim)
        
        # 位置埋め込みを追加
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer処理
        x = self.transformer(x)  # (B, N+1, embed_dim)
        
        # CLSトークンを取り出して分類
        cls_output = x[:, 0]  # (B, embed_dim)
        cls_output = self.norm(cls_output)
        logits = self.head(cls_output)  # (B, num_classes)
        
        return logits


def create_vit_small(num_classes: int = 7, img_size: int = 224) -> ImageViT:
    """ViT-Small/16 (パラメータ数: ~22M)"""
    return ImageViT(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        num_classes=num_classes,
    )


def create_vit_base(num_classes: int = 7, img_size: int = 224) -> ImageViT:
    """ViT-Base/16 (パラメータ数: ~86M)"""
    return ImageViT(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        num_classes=num_classes,
    )


def create_vit_tiny(num_classes: int = 7, img_size: int = 224) -> ImageViT:
    """ViT-Tiny/16 (小規模データセット用、パラメータ数: ~5M)"""
    return ImageViT(
        img_size=img_size,
        patch_size=16,
        embed_dim=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        num_classes=num_classes,
    )