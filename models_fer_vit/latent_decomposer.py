"""
InterFaceGAN 理論に基づく潜在コードの表情成分 / アイデンティティ成分分解モジュール。

論文: "Interpreting the Latent Space of GANs for Semantic Face Editing"
     Shen et al., CVPR 2020

理論:
    GAN の潜在空間は線形分離可能であり、二値 LinearSVM の決定境界の
    法線ベクトル n_expr が各意味的属性（表情など）の「編集方向」に対応する。

    w+ を n_expr 方向とその直交補空間に分解:
        w_expr = (w · n_expr) * n_expr    ← 表情成分
        w_id   = w - w_expr               ← アイデンティティ成分

    ViT に w_expr だけを入力することで、アイデンティティに依存しない
    純粋な表情特徴を学習させる。
"""

import torch
import torch.nn as nn
from typing import Dict, Literal, Optional


EMOTION_NAMES = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
    4: 'neutral', 5: 'sad', 6: 'surprise',
}


class LatentDecomposer(nn.Module):
    """
    事前計算済み表情方向ベクトルを用いて w+ を分解するモジュール。

    方向ベクトルは compute_expression_directions.py で算出した
    .pt ファイルからロードする。パラメータではなく buffer として保持するため
    学習中に変化しない。

    使い方:
        decomposer = LatentDecomposer.from_file('directions/binary_directions.pt')

        w_expr, w_id = decomposer.decompose(w_plus)  # (B,18,512) -> 2x(B,18,512)
        x = decomposer(w_plus, output_mode='expr_only')
    """

    def __init__(
        self,
        directions: Dict[int, torch.Tensor],
        seq_len: int = 18,
        latent_dim: int = 512,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.num_classes = len(directions)

        # (C, 18, 512) にスタックして再正規化し buffer に登録
        dirs = torch.stack(
            [directions[i] for i in range(self.num_classes)], dim=0
        )  # (C, 18, 512)
        dirs_flat = dirs.view(self.num_classes, -1)
        norms = dirs_flat.norm(dim=1, keepdim=True)
        dirs_flat = dirs_flat / (norms + 1e-12)
        dirs = dirs_flat.view(self.num_classes, seq_len, latent_dim)

        self.register_buffer('directions', dirs)  # (C, 18, 512)

    @classmethod
    def from_file(cls, path: str) -> 'LatentDecomposer':
        """保存済み方向ベクトルファイルからインスタンスを作成"""
        data = torch.load(path, map_location='cpu', weights_only=False)
        directions = data['directions']
        seq_len = data.get('seq_len', 18)
        latent_dim = data.get('latent_dim', 512)
        method = data.get('method', 'unknown')

        print(f"Loaded '{method}' expression directions: {path}")
        print(f"  Classes  : {list(directions.keys())}")
        print(f"  Direction shape: ({seq_len}, {latent_dim}) x {len(directions)} classes")

        return cls(directions, seq_len, latent_dim)

    def decompose(
        self,
        w_plus: torch.Tensor,
        mode: Literal['all_classes', 'max_class'] = 'all_classes',
    ):
        """
        w+ を表情成分とアイデンティティ成分に分解する。

        Args:
            w_plus: (B, 18, 512)
            mode:
                'all_classes' - 全クラス方向への射影の和を表情成分とする
                                  w_expr = sum_c (w・n_c) * n_c
                'max_class'   - 絶対値最大スコアのクラス方向のみを使用

        Returns:
            w_expr: (B, 18, 512) 表情成分
            w_id  : (B, 18, 512) アイデンティティ成分
        """
        B = w_plus.size(0)
        dirs_flat = self.directions.view(self.num_classes, -1)  # (C, D)
        w_flat = w_plus.reshape(B, -1)                          # (B, D)

        proj_coeffs = w_flat @ dirs_flat.T  # (B, C)

        if mode == 'all_classes':
            w_expr_flat = proj_coeffs @ dirs_flat               # (B, D)
        elif mode == 'max_class':
            best_cls = proj_coeffs.abs().argmax(dim=1)          # (B,)
            best_dirs = self.directions[best_cls]               # (B, 18, 512)
            best_coeffs = proj_coeffs[torch.arange(B), best_cls]
            w_expr_flat = (best_coeffs.view(B, 1, 1) * best_dirs).reshape(B, -1)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        w_expr = w_expr_flat.reshape(B, self.seq_len, self.latent_dim)
        w_id = w_plus - w_expr
        return w_expr, w_id

    def get_expression_scores(self, w_plus: torch.Tensor) -> torch.Tensor:
        """
        各クラスの表情スコアを返す（SVM 決定関数に相当）。
        Args:
            w_plus: (B, 18, 512)
        Returns:
            scores: (B, num_classes)
        """
        dirs_flat = self.directions.view(self.num_classes, -1)
        w_flat = w_plus.reshape(w_plus.size(0), -1)
        return w_flat @ dirs_flat.T

    def enhance_expression(
        self,
        w_plus: torch.Tensor,
        alpha: float = 2.0,
        mode: Literal['all_classes', 'max_class'] = 'all_classes',
    ) -> torch.Tensor:
        """
        表情成分を alpha 倍に強調した w+ を返す。
        InterFaceGAN の w' = w + alpha*n_expr の一般化。
        alpha=1 で元の w+ に等しい。
        """
        w_expr, w_id = self.decompose(w_plus, mode=mode)
        return w_id + alpha * w_expr

    def forward(
        self,
        w_plus: torch.Tensor,
        output_mode: Literal['expr_only', 'id_only', 'enhanced', 'concat'] = 'expr_only',
        enhance_alpha: float = 2.0,
        decompose_mode: Literal['all_classes', 'max_class'] = 'all_classes',
    ) -> torch.Tensor:
        """
        ViT 入力用に変換された潜在コードを返す。

        output_mode:
            'expr_only' -> (B, 18, 512)  表情成分のみ
            'id_only'   -> (B, 18, 512)  アイデンティティ成分のみ
            'enhanced'  -> (B, 18, 512)  表情を enhance_alpha 倍に強調
            'concat'    -> (B, 36, 512)  表情+アイデンティティを連結
        """
        w_expr, w_id = self.decompose(w_plus, mode=decompose_mode)

        if output_mode == 'expr_only':
            return w_expr
        elif output_mode == 'id_only':
            return w_id
        elif output_mode == 'enhanced':
            return w_id + enhance_alpha * w_expr
        elif output_mode == 'concat':
            return torch.cat([w_expr, w_id], dim=1)  # (B, 36, 512)
        else:
            raise ValueError(f"Unknown output_mode: {output_mode!r}")