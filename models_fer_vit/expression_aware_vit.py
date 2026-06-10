"""
InterFaceGAN の線形分離理論を組み込んだ表情認識モデル。

パイプライン:
    pSp Encoder -> w+ (B, 18, 512)
                -> LatentDecomposer -> w_expr (B, 18, 512)  <- 表情成分
                -> HybridLatentViT  -> 分類ヘッド -> ロジット

論文理論との対応:
    - LinearSVM の法線ベクトル = 潜在空間における表情方向
    - w_expr = その方向への射影 = 表情に関する情報のみを保持
    - w_id   = 直交補空間への射影 = アイデンティティ情報
    - ViT に w_expr のみを渡すことで ID 成分を除去し表情分類に特化させる
"""

import torch
import torch.nn as nn
from typing import Optional, Literal

from models_fer_vit.latent_decomposer import LatentDecomposer
from models_fer_vit.hybrid_latent_vit import HybridLatentViT, create_hybrid_latent_vit


class ExpressionAwareViT(nn.Module):
    """
    LatentDecomposer + HybridLatentViT を組み合わせた表情認識モデル。

    LatentDecomposer は固定 (SVM 由来の buffer)、
    HybridLatentViT の重みのみを学習する。
    """

    def __init__(
        self,
        decomposer: LatentDecomposer,
        vit_model: HybridLatentViT,
        output_mode: Literal['expr_only', 'id_only', 'enhanced', 'concat'] = 'expr_only',
        enhance_alpha: float = 2.0,
        decompose_mode: Literal['all_classes', 'max_class'] = 'all_classes',
    ):
        super().__init__()
        self.decomposer = decomposer
        self.vit = vit_model
        self.output_mode = output_mode
        self.enhance_alpha = enhance_alpha
        self.decompose_mode = decompose_mode

        print(f"\n[ExpressionAwareViT]")
        print(f"  decompose_mode : {decompose_mode}")
        print(f"  output_mode    : {output_mode}")
        if output_mode == 'enhanced':
            print(f"  enhance_alpha  : {enhance_alpha}")

    @classmethod
    def from_config(
        cls,
        directions_path: str,
        model_size: str = 'small',
        num_classes: int = 7,
        use_pretrained: bool = True,
        freeze_transformer: bool = False,
        freeze_stages: Optional[int] = None,
        use_adapter: bool = False,
        adapter_dim: int = 64,
        output_mode: Literal['expr_only', 'id_only', 'enhanced', 'concat'] = 'expr_only',
        enhance_alpha: float = 2.0,
        decompose_mode: Literal['all_classes', 'max_class'] = 'all_classes',
    ) -> 'ExpressionAwareViT':
        """
        設定値からモデルを作成するファクトリメソッド。

        Args:
            directions_path : compute_expression_directions.py で生成した .pt ファイル
            model_size      : 'tiny' | 'small' | 'base'
            num_classes     : 分類クラス数 (FER2013 は 7)
            use_pretrained  : ImageNet 事前学習重みを使用するか
            freeze_transformer: Transformer 全体を凍結するか
            freeze_stages   : 最初 N ブロックを凍結 (None=凍結なし)
            use_adapter     : アダプター層を使用するか
            adapter_dim     : アダプターのボトルネック次元
            output_mode     : 分解後の ViT 入力モード
            enhance_alpha   : 'enhanced' モード時の強調係数
            decompose_mode  : 分解計算のモード
        """
        decomposer = LatentDecomposer.from_file(directions_path)

        # 'concat' モードは表情+アイデンティティを連結するため seq_len が 2 倍
        seq_len = decomposer.seq_len * (2 if output_mode == 'concat' else 1)

        vit = create_hybrid_latent_vit(
            latent_dim=decomposer.latent_dim,
            seq_len=seq_len,
            model_size=model_size,
            num_classes=num_classes,
            use_pretrained=use_pretrained,
            freeze_transformer=freeze_transformer,
            freeze_stages=freeze_stages,
            use_adapter=use_adapter,
            adapter_dim=adapter_dim,
        )

        return cls(
            decomposer=decomposer,
            vit_model=vit,
            output_mode=output_mode,
            enhance_alpha=enhance_alpha,
            decompose_mode=decompose_mode,
        )

    def forward(self, w_plus: torch.Tensor) -> torch.Tensor:
        """
        Args:
            w_plus: (B, 18, 512) StyleGAN w+ 潜在コード
        Returns:
            logits: (B, num_classes)
        """
        x = self.decomposer(
            w_plus,
            output_mode=self.output_mode,
            enhance_alpha=self.enhance_alpha,
            decompose_mode=self.decompose_mode,
        )
        return self.vit(x)

    def get_trainable_params(self):
        """学習対象パラメータのみを返す (ViT 側のみ)"""
        return [p for p in self.vit.parameters() if p.requires_grad]

    def print_info(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n[ExpressionAwareViT] Parameters:")
        print(f"  Total      : {total:,}")
        print(f"  Trainable  : {trainable:,} ({trainable/total*100:.1f}%)")
        print(f"  Decomposer : fixed (SVM directions, not trained)")