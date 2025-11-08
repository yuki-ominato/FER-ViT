"""
事前学習ViTのTransformer Encoderを潜在コードに適用
パッチ埋め込み層をスキップし、Transformer部分のみを活用
"""

import torch
import torch.nn as nn
from typing import Optional, Literal

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Install with: pip install timm")


class HybridLatentViT(nn.Module):
    """
    事前学習済みViTのTransformer部分を潜在コードに適用
    
    アーキテクチャ:
    StyleGAN Latent (B, 18, 512) 
    → Linear Projection (B, 18, embed_dim)
    → Add Position Embedding
    → Pretrained Transformer Encoder [重要!]
    → Classification Head
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        seq_len: int = 18,
        pretrained_model_name: str = 'vit_small_patch16_224',
        num_classes: int = 7,
        use_pretrained: bool = True,
        freeze_transformer: bool = False,
        freeze_stages: Optional[int] = None,
        adapter_dim: Optional[int] = None,  # アダプター層の次元
    ):
        """
        Args:
            latent_dim: 潜在コードの次元 (StyleGAN: 512)
            seq_len: シーケンス長 (StyleGAN w+: 18)
            pretrained_model_name: 事前学習ViTモデル名
            num_classes: 出力クラス数
            use_pretrained: 事前学習済み重みを使用
            freeze_transformer: Transformer全体を凍結
            freeze_stages: 最初のN層を凍結
            adapter_dim: アダプター層を使う場合の次元（Noneなら不使用）
        """
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required. Install with: pip install timm")
        
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.pretrained_model_name = pretrained_model_name
        self.use_adapter = adapter_dim is not None
        
        # 事前学習モデルをロード
        print(f"\n{'='*60}")
        print(f"Loading pretrained model: {pretrained_model_name}")
        print(f"{'='*60}")
        
        pretrained_vit = timm.create_model(
            pretrained_model_name,
            pretrained=use_pretrained,
            num_classes=0,  # 分類ヘッドなし
        )
        
        # Transformerのembedding次元を取得
        self.embed_dim = pretrained_vit.embed_dim
        print(f"Pretrained model embedding dimension: {self.embed_dim}")
        
        # 1. 潜在コードをTransformerの次元に射影
        self.input_proj = nn.Linear(latent_dim, self.embed_dim)
        
        # 2. CLSトークン（新規学習 or 事前学習から継承）
        if hasattr(pretrained_vit, 'cls_token'):
            self.cls_token = nn.Parameter(pretrained_vit.cls_token.data.clone())
            print("Using pretrained CLS token")
        else:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            print("Initialized new CLS token")
        
        # 3. 位置埋め込み（サイズを調整）
        self.pos_embed = self._init_position_embedding(pretrained_vit, seq_len)
        
        # 4. 事前学習済みTransformer Encoderを抽出
        self.transformer = self._extract_transformer(pretrained_vit)
        
        # 5. アダプター層（オプション）
        if self.use_adapter:
            print(f"Using adapter layers with dim={adapter_dim}")
            self.adapters = nn.ModuleList([
                AdapterModule(self.embed_dim, adapter_dim)
                for _ in range(len(self.transformer))
            ])
        
        # 6. 凍結設定
        if freeze_transformer:
            self._freeze_transformer()
        elif freeze_stages is not None:
            self._freeze_stages(freeze_stages)
        
        # 7. 分類ヘッド（新規学習）
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, num_classes),
        )
        
        self._print_model_info()
    
    def _init_position_embedding(self, pretrained_vit, seq_len):
        """位置埋め込みの初期化"""
        if hasattr(pretrained_vit, 'pos_embed'):
            pretrained_pos = pretrained_vit.pos_embed  # (1, N+1, embed_dim)
            
            # 事前学習の位置埋め込みサイズ
            pretrained_seq_len = pretrained_pos.size(1) - 1  # CLSトークンを除く
            
            if seq_len == pretrained_seq_len:
                # サイズが一致する場合はそのまま使用
                print(f"Using pretrained position embeddings (seq_len={seq_len})")
                return nn.Parameter(pretrained_pos.data.clone())
            else:
                # サイズが異なる場合は補間
                print(f"Interpolating position embeddings: {pretrained_seq_len} → {seq_len}")
                
                # CLSトークンの位置埋め込み
                cls_pos = pretrained_pos[:, 0:1, :]  # (1, 1, embed_dim)
                
                # パッチの位置埋め込み
                patch_pos = pretrained_pos[:, 1:, :]  # (1, N, embed_dim)
                
                # 1D補間（シーケンス長を変更）
                patch_pos = patch_pos.permute(0, 2, 1)  # (1, embed_dim, N)
                patch_pos = nn.functional.interpolate(
                    patch_pos,
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                )
                patch_pos = patch_pos.permute(0, 2, 1)  # (1, seq_len, embed_dim)
                
                # 結合
                new_pos = torch.cat([cls_pos, patch_pos], dim=1)
                return nn.Parameter(new_pos)
        else:
            # 事前学習に位置埋め込みがない場合（稀）
            print(f"Initializing new position embeddings (seq_len={seq_len})")
            return nn.Parameter(torch.randn(1, seq_len + 1, self.embed_dim))
    
    def _extract_transformer(self, pretrained_vit):
        """Transformer Encoderを抽出"""
        if hasattr(pretrained_vit, 'blocks'):
            # timmのViTは`blocks`にTransformerレイヤーが格納されている
            transformer_blocks = pretrained_vit.blocks
            print(f"Extracted {len(transformer_blocks)} transformer blocks from pretrained model")
            return transformer_blocks
        else:
            raise AttributeError(
                f"Cannot extract transformer blocks from {self.pretrained_model_name}. "
                f"Model structure may be different."
            )
    
    def _freeze_transformer(self):
        """Transformer全体を凍結"""
        for param in self.transformer.parameters():
            param.requires_grad = False
        print("Transformer frozen")
    
    def _freeze_stages(self, n_stages: int):
        """最初のN個のTransformerブロックを凍結"""
        n_stages = min(n_stages, len(self.transformer))
        for i in range(n_stages):
            for param in self.transformer[i].parameters():
                param.requires_grad = False
        print(f"Frozen first {n_stages}/{len(self.transformer)} transformer blocks")
    
    def _print_model_info(self):
        """モデル情報を表示"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n{'='*60}")
        print("Model Information:")
        print(f"{'='*60}")
        print(f"  Input: Latent codes ({self.seq_len}, {self.latent_dim})")
        print(f"  Transformer embedding dim: {self.embed_dim}")
        print(f"  Number of transformer blocks: {len(self.transformer)}")
        print(f"  Output classes: {self.num_classes}")
        print(f"  Adapter: {'Yes' if self.use_adapter else 'No'}")
        print(f"\nParameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"  Frozen: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print(f"{'='*60}\n")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, latent_dim) 潜在コード
        Returns:
            (B, num_classes) ロジット
        """
        B = x.size(0)
        
        # 1. 線形射影で次元を合わせる
        x = self.input_proj(x)  # (B, seq_len, embed_dim)
        
        # 2. CLSトークンを追加
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, seq_len+1, embed_dim)
        
        # 3. 位置埋め込みを追加
        x = x + self.pos_embed  # (B, seq_len+1, embed_dim)
        
        # 4. Transformer処理
        if self.use_adapter:
            # アダプター付きTransformer
            for i, block in enumerate(self.transformer):
                x = block(x)
                x = self.adapters[i](x)
        else:
            # 標準Transformer
            for block in self.transformer:
                x = block(x)
        
        # 5. CLSトークンを取り出して分類
        cls_output = x[:, 0]  # (B, embed_dim)
        logits = self.head(cls_output)  # (B, num_classes)
        
        return logits
    
    def unfreeze_all(self):
        """全パラメータの凍結を解除"""
        for param in self.parameters():
            param.requires_grad = True
        print("All parameters unfrozen")
        self._print_model_info()


class AdapterModule(nn.Module):
    """
    Adapter層（Parameter-Efficient Fine-Tuning用）
    事前学習モデルを凍結したまま、小さなアダプター層のみを学習
    """
    
    def __init__(self, embed_dim: int, adapter_dim: int):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, embed_dim),
        )
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)  # スケーリング係数
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.adapter(x)


def create_hybrid_latent_vit(
    latent_dim: int = 512,
    seq_len: int = 18,
    model_size: Literal['tiny', 'small', 'base'] = 'small',
    num_classes: int = 7,
    use_pretrained: bool = True,
    freeze_transformer: bool = False,
    freeze_stages: Optional[int] = None,
    use_adapter: bool = False,
    adapter_dim: int = 64,
) -> HybridLatentViT:
    """
    ハイブリッドモデルの作成
    
    Args:
        latent_dim: StyleGAN潜在次元
        seq_len: シーケンス長
        model_size: 事前学習モデルのサイズ
        num_classes: クラス数
        use_pretrained: 事前学習済み重みを使用
        freeze_transformer: Transformer凍結
        freeze_stages: 部分凍結
        use_adapter: アダプター層を使用
        adapter_dim: アダプター次元
    """
    model_names = {
        'tiny': 'vit_tiny_patch16_224',
        'small': 'vit_small_patch16_224',
        'base': 'vit_base_patch16_224',
    }
    
    model_name = model_names.get(model_size, 'vit_small_patch16_224')
    
    return HybridLatentViT(
        latent_dim=latent_dim,
        seq_len=seq_len,
        pretrained_model_name=model_name,
        num_classes=num_classes,
        use_pretrained=use_pretrained,
        freeze_transformer=freeze_transformer,
        freeze_stages=freeze_stages,
        adapter_dim=adapter_dim if use_adapter else None,
    )


# 推奨設定
RECOMMENDED_STRATEGIES = {
    'full_finetune': {
        'freeze_transformer': False,
        'freeze_stages': None,
        'use_adapter': False,
        'lr': 1e-4,
        'description': '全パラメータを学習（最高精度、学習時間長）'
    },
    'partial_freeze': {
        'freeze_transformer': False,
        'freeze_stages': 6,
        'use_adapter': False,
        'lr': 3e-4,
        'description': '下位層凍結（バランス）'
    },
    'adapter': {
        'freeze_transformer': True,
        'freeze_stages': None,
        'use_adapter': True,
        'lr': 1e-3,
        'description': 'アダプター層のみ学習（最速、メモリ効率的）'
    },
    'linear_probe': {
        'freeze_transformer': True,
        'freeze_stages': None,
        'use_adapter': False,
        'lr': 1e-3,
        'description': '分類ヘッドのみ学習（ベースライン）'
    },
}


if __name__ == "__main__":
    print("="*60)
    print("Hybrid Latent ViT - Usage Examples")
    print("="*60)
    
    # 推奨戦略を表示
    print("\nRecommended Fine-tuning Strategies:")
    print("="*60)
    for name, config in RECOMMENDED_STRATEGIES.items():
        print(f"\n{name.upper()}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # モデル作成例
    print("\n" + "="*60)
    print("Creating model examples:")
    print("="*60)
    
    # 例1: 完全ファインチューニング
    print("\n1. Full Fine-tuning:")
    model1 = create_hybrid_latent_vit(
        model_size='small',
        use_pretrained=True,
        freeze_transformer=False,
    )
    
    # 例2: アダプター学習
    print("\n2. Adapter-based learning:")
    model2 = create_hybrid_latent_vit(
        model_size='small',
        use_pretrained=True,
        freeze_transformer=True,
        use_adapter=True,
        adapter_dim=64,
    )
    
    # 例3: 部分凍結
    print("\n3. Partial freeze:")
    model3 = create_hybrid_latent_vit(
        model_size='small',
        use_pretrained=True,
        freeze_stages=6,
    )
    
    # テスト
    print("\n" + "="*60)
    print("Testing forward pass:")
    print("="*60)
    batch_size = 4
    seq_len = 18
    latent_dim = 512
    
    dummy_input = torch.randn(batch_size, seq_len, latent_dim)
    output = model1(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, 7)")