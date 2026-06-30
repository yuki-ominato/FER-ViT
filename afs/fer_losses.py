"""
FER特化StyleExtractor用損失関数。

損失構成:
    L_expr   = CE( FC(h(w)),        label      )  表情コードの識別可能性
    L_neutral= CE( FC(w − h(w)),    neutral=4  )  残差コードの無表情化
    L_id     = 1 − cos( Arc(G(w_new)), Arc(G(w_src)) )  アイデンティティ保存
    L_sparse = mean|h(w)[:, non_expr_layers, :]|  非表情層のスパース性
    L_cons   = L1( h(w_new), stop_grad(h(w_tgt)) )  一貫性
    L_total  = λ_e*L_expr + λ_id*L_id + λ_n*L_neutral + λ_s*L_sparse + λ_c*L_cons

ExprClassifier は AFSFERLoss 内に保持され、StyleExtractor h と共同学習される。
→ optimizer には h.parameters() と criterion.classifier.parameters() を両方渡す。

W+ 層と解像度の対応 (1024px StyleGAN2 / 18層):
    0-1  : 4×4   (coarse pose / overall shape)
    2-3  : 8×8
    4-5  : 16×16 ←┐
    6-7  : 32×32   │  EXPR_LAYERS (表情が集中する中間層)
    8-9  : 64×64   │
    10-11: 128×128 ←┘
    12-13: 256×256
    14-17: 512-1024 (fine texture / color)
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_PSP_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'third_party', 'pixel2style2pixel')
)
if _PSP_ROOT not in sys.path:
    sys.path.insert(0, _PSP_ROOT)

from models.encoders.model_irse import Backbone   # ArcFace backbone


# ---------------------------------------------------------------------------
# ArcFace extractor (losses.py と同実装)
# ---------------------------------------------------------------------------

class ArcFaceExtractor(nn.Module):
    """Frozen ArcFace (IR-SE50). 入力 [B,3,256,256] → 出力 [B,512]"""

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.net = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.net.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.net.eval()
        self.pool = nn.AdaptiveAvgPool2d((112, 112))
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :, 35:223, 32:220]
        x = self.pool(x)
        return self.net(x)


# ---------------------------------------------------------------------------
# Joint expression classifier
# ---------------------------------------------------------------------------

class ExprClassifier(nn.Module):
    """
    h(w) または (w − h(w)) の層平均から感情クラスを予測する軽量 MLP。
    StyleExtractor h と共同学習される（勾配がそのまま h に流れる）。

    入力: (B, seq_len, latent_dim) — W+ 潜在コードまたはその部分
    出力: (B, num_classes)         — 感情ロジット
    """

    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 7,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # w: (B, S, D) → mean over S → (B, D) → (B, num_classes)
        return self.fc(w.mean(dim=1))


# ---------------------------------------------------------------------------
# FER-focused combined loss
# ---------------------------------------------------------------------------

class AFSFERLoss(nn.Module):
    """
    FER タスク向け StyleExtractor 損失関数。

    Args
    ----
    arcface_path  : model_ir_se50.pth へのパス（L_id 用）。
    generator     : 凍結済み StyleGAN2 Generator。None の場合 L_id = 0。
    latent_dim    : W+ 潜在コードの次元数（デフォルト 512）。
    num_classes   : 感情クラス数（デフォルト 7）。
    lambda_expr   : L_expr の係数（デフォルト 1.0）。
    lambda_id     : L_id   の係数（デフォルト 1.0）。
    lambda_neutral: L_neutral の係数（デフォルト 0.5）。
    lambda_sparse : L_sparse の係数（デフォルト 0.02）。
    lambda_cons   : L_cons  の係数（デフォルト 0.1）。

    Forward 引数
    ------------
    h_src    [B,18,512]  h(w_src)
    h_tgt    [B,18,512]  h(w_tgt)
    h_new    [B,18,512]  h(w_new)   w_new = (w_src − h_src) + h_tgt
    w_src    [B,18,512]  元の潜在コード（人物 A）
    w_tgt    [B,18,512]  元の潜在コード（人物 B / ターゲット表情）
    label_src [B,]  long  w_src の感情ラベル
    label_tgt [B,]  long  w_tgt の感情ラベル
    img_gen  [B,3,256,256]  G(w_new)  （generator が None の場合は使用しない）
    img_src  [B,3,256,256]  G(w_src)  （同上）

    Returns
    -------
    l_total  : スカラー損失
    metrics  : dict {"expr", "id", "neutral", "sparse", "cons"}
    """

    # 表情が集中する W+ 層インデックス（16×16 〜 128×128 に対応）
    EXPR_LAYERS: list[int] = list(range(4, 12))
    NON_EXPR_LAYERS: list[int] = [i for i in range(18) if i not in range(4, 12)]
    NEUTRAL_LABEL: int = 4

    def __init__(
        self,
        arcface_path: str,
        generator: Optional[nn.Module] = None,
        latent_dim: int = 512,
        num_classes: int = 7,
        lambda_expr: float    = 1.0,
        lambda_id: float      = 1.0,
        lambda_neutral: float = 0.5,
        lambda_sparse: float  = 0.02,
        lambda_cons: float    = 0.1,
    ) -> None:
        super().__init__()
        self.arcface  = ArcFaceExtractor(arcface_path)
        self.classifier = ExprClassifier(latent_dim, num_classes=num_classes)

        self.lambda_expr    = lambda_expr
        self.lambda_id      = lambda_id
        self.lambda_neutral = lambda_neutral
        self.lambda_sparse  = lambda_sparse
        self.lambda_cons    = lambda_cons

        self.ce = nn.CrossEntropyLoss()

        # generator はサブモジュール登録せずに保持（criterion.to(device) の汚染防止）
        if generator is not None:
            object.__setattr__(self, '_generator_ref', generator)
            print("AFSFERLoss: generator registered for L_id")
        else:
            object.__setattr__(self, '_generator_ref', None)
            print("AFSFERLoss: generator not provided — L_id = 0")

    # ------------------------------------------------------------------
    # individual loss components
    # ------------------------------------------------------------------

    def _l_expr(
        self,
        h_src: torch.Tensor,
        h_tgt: torch.Tensor,
        label_src: torch.Tensor,
        label_tgt: torch.Tensor,
    ) -> torch.Tensor:
        """L_expr = CE(FC(h(w_src)), label_src) + CE(FC(h(w_tgt)), label_tgt)"""
        logits_src = self.classifier(h_src)
        logits_tgt = self.classifier(h_tgt)
        return 0.5 * (self.ce(logits_src, label_src) + self.ce(logits_tgt, label_tgt))

    def _l_neutral(
        self,
        h_src: torch.Tensor,
        h_tgt: torch.Tensor,
        w_src: torch.Tensor,
        w_tgt: torch.Tensor,
    ) -> torch.Tensor:
        """L_neutral = CE(FC(w_rest), neutral=4) で残差が無表情を示すよう制約"""
        B = w_src.size(0)
        device = w_src.device
        neutral = torch.full((B,), self.NEUTRAL_LABEL, dtype=torch.long, device=device)
        logits_rest_src = self.classifier(w_src - h_src)
        logits_rest_tgt = self.classifier(w_tgt - h_tgt)
        return 0.5 * (self.ce(logits_rest_src, neutral) + self.ce(logits_rest_tgt, neutral))

    def _l_id(
        self,
        img_gen: Optional[torch.Tensor],
        img_src: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """L_id = 1 − cos(ArcFace(G(w_new)), ArcFace(G(w_src)))"""
        gen_ref = self.__dict__.get('_generator_ref')
        if gen_ref is None or img_gen is None or img_src is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        with torch.no_grad():
            feat_src = self.arcface(img_src)
        feat_gen = self.arcface(img_gen)
        return (1.0 - F.cosine_similarity(feat_gen, feat_src, dim=1)).mean()

    def _l_sparse(
        self,
        h_src: torch.Tensor,
        h_tgt: torch.Tensor,
    ) -> torch.Tensor:
        """L_sparse = mean |h(w)[:, non_expr_layers, :]| — 非表情層へのスパース正則化"""
        non_src = h_src[:, self.NON_EXPR_LAYERS, :]   # (B, 10, 512)
        non_tgt = h_tgt[:, self.NON_EXPR_LAYERS, :]
        return 0.5 * (non_src.abs().mean() + non_tgt.abs().mean())

    def _l_cons(
        self,
        h_new: torch.Tensor,
        h_tgt: torch.Tensor,
    ) -> torch.Tensor:
        """L_cons = L1(h(w_new), stop_grad(h(w_tgt)))"""
        return F.l1_loss(h_new, h_tgt.detach())

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        h_src:     torch.Tensor,
        h_tgt:     torch.Tensor,
        h_new:     torch.Tensor,
        w_src:     torch.Tensor,
        w_tgt:     torch.Tensor,
        label_src: torch.Tensor,
        label_tgt: torch.Tensor,
        img_gen:   Optional[torch.Tensor] = None,
        img_src:   Optional[torch.Tensor] = None,
    ):
        l_expr    = self._l_expr(h_src, h_tgt, label_src, label_tgt)
        l_neutral = self._l_neutral(h_src, h_tgt, w_src, w_tgt)
        l_id      = self._l_id(img_gen, img_src)
        l_sparse  = self._l_sparse(h_src, h_tgt)
        l_cons    = self._l_cons(h_new, h_tgt)

        l_total = (self.lambda_expr    * l_expr
                   + self.lambda_id      * l_id
                   + self.lambda_neutral * l_neutral
                   + self.lambda_sparse  * l_sparse
                   + self.lambda_cons    * l_cons)

        metrics = {
            "expr":    l_expr.item(),
            "id":      l_id.item(),
            "neutral": l_neutral.item(),
            "sparse":  l_sparse.item(),
            "cons":    l_cons.item(),
        }
        return l_total, metrics
