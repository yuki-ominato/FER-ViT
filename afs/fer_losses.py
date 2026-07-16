"""
FER特化StyleExtractor用損失関数。

損失構成:
    L_expr       = CE( FC(h(w)),        label      )  表情コードの識別可能性（潜在空間、補助）
    L_neutral    = CE( FC(w − h(w)),    neutral=4  )  残差コードの無表情化（潜在空間、補助）
    L_expr_img   = CE( R_FER(G(w_new)),       label_tgt )  画像ドメインでの表情転写評価
    L_neutral_img= CE( R_FER(G(w_src−h_src)), neutral=4 )  画像ドメインでの残差無表情化評価
    L_id         = 1 − cos( Arc(G(w_new)), Arc(G(w_src)) )  アイデンティティ保存
    L_sparse     = mean|h(w)[:, non_expr_layers, :]|  非表情層のスパース性
    L_cons       = L1( h(w_new), stop_grad(h(w_tgt)) )  一貫性
    L_total = λ_e*L_expr + λ_id*L_id + λ_n*L_neutral + λ_s*L_sparse + λ_c*L_cons
              + λ_ei*L_expr_img + λ_ni*L_neutral_img

背景（document/AFS_FER_diagnosis.md 参照）:
    当初の設計 (document/AFS_FER.md) は
        L_expr = 1 - cos(R_FER(G(w_expr + w_rest^target)), R_FER(x_expr_source))
    のように「生成画像を FER モデルに通して評価する」ことを想定していたが、
    L_expr/L_neutral は Generator を経由せず h(w) という生の潜在ベクトルに
    正則化なしの小さな MLP (ExprClassifier) を直接適用するだけになっていた。
    これは潜在空間上では表情ラベルを判別できる（train accuracy 60%台）のに、
    デコードした画像には表情の違いがほぼ現れない、という矛盾を生んでいた
    （ExprClassifier が大きな重みで極小の潜在方向を増幅する "近道" を学習できるため）。

    L_expr_img / L_neutral_img は、実際に G() でデコードした画像を
    画像ベースの FER モデル (train/train_image_vit.py で学習済みの ImageViT 等、
    afs/fer_image_classifier.py::FERImageClassifier) に通して評価することで、
    「視覚的に知覚できる表情変化」を直接強制する。ExprClassifier ベースの
    L_expr/L_neutral は計算コストの低い補助信号として残しているが、
    表情分離を実際に保証するのは L_expr_img/L_neutral_img 側である。

ExprClassifier は AFSFERLoss 内に保持され、StyleExtractor h と共同学習される。
→ optimizer には h.parameters() と criterion.classifier.parameters() を両方渡す。
FERImageClassifier (画像ベース FER モデル) は完全に凍結され、学習対象に含めない。

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
from afs.losses import _FeatureHook               # StyleGAN2 中間特徴フック
from afs.fer_image_classifier import FERImageClassifier  # 画像ベース FER モデル


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
    fer_image_ckpt: train_image_vit.py で学習した画像ベース FER モデルの
                    best_model.pt へのパス（L_expr_img/L_neutral_img 用）。
                    None の場合はこれらの損失を 0 として無効化する（後方互換）。
    latent_dim    : W+ 潜在コードの次元数（デフォルト 512）。
    num_classes   : 感情クラス数（デフォルト 7）。
    lambda_expr   : L_expr の係数（デフォルト 1.0、潜在空間・補助）。
    lambda_id     : L_id   の係数（デフォルト 1.0）。
    lambda_neutral: L_neutral の係数（デフォルト 0.5、潜在空間・補助）。
    lambda_sparse : L_sparse の係数（デフォルト 0.02）。
    lambda_cons   : L_cons  の係数（デフォルト 0.1）。
    lambda_expr_img   : L_expr_img の係数（デフォルト 1.0、画像ドメイン・本命）。
    lambda_neutral_img: L_neutral_img の係数（デフォルト 0.5、画像ドメイン・本命）。

    Forward 引数
    ------------
    h_src    [B,18,512]  h(w_src)
    h_tgt    [B,18,512]  h(w_tgt)
    h_new    [B,18,512]  h(w_new)   w_new = (w_src − h_src) + h_tgt
    w_src    [B,18,512]  元の潜在コード（人物 A）
    w_tgt    [B,18,512]  元の潜在コード（人物 B / ターゲット表情）
    label_src [B,]  long  w_src の感情ラベル
    label_tgt [B,]  long  w_tgt の感情ラベル
    img_gen    [B,3,256,256]  G(w_new)             （generator が None の場合は使用しない）
    img_src    [B,3,256,256]  G(w_src)             （同上）
    img_id_src [B,3,256,256]  G(w_src − h_src)      （L_neutral_img 用。None なら 0）

    Returns
    -------
    l_total  : スカラー損失
    metrics  : dict {"expr", "id", "neutral", "sparse", "cons", "expr_img", "neutral_img"}
    """

    # 表情が集中する W+ 層インデックス（16×16 〜 128×128 に対応）
    EXPR_LAYERS: list[int] = list(range(4, 12))
    NON_EXPR_LAYERS: list[int] = [i for i in range(18) if i not in range(4, 12)]
    NEUTRAL_LABEL: int = 4

    def __init__(
        self,
        arcface_path: str,
        generator: Optional[nn.Module] = None,
        fer_image_ckpt: Optional[str] = None,
        latent_dim: int = 512,
        num_classes: int = 7,
        lambda_expr: float    = 1.0,
        lambda_id: float      = 1.0,
        lambda_neutral: float = 0.5,
        lambda_sparse: float  = 0.02,
        lambda_cons: float    = 0.1,
        lambda_feat: float    = 3.5,
        lambda_expr_img: float    = 1.0,
        lambda_neutral_img: float = 0.5,
    ) -> None:
        super().__init__()
        self.arcface    = ArcFaceExtractor(arcface_path)
        self.classifier = ExprClassifier(latent_dim, num_classes=num_classes)

        self.lambda_expr    = lambda_expr
        self.lambda_id      = lambda_id
        self.lambda_neutral = lambda_neutral
        self.lambda_sparse  = lambda_sparse
        self.lambda_cons    = lambda_cons
        self.lambda_feat    = lambda_feat
        self.lambda_expr_img    = lambda_expr_img
        self.lambda_neutral_img = lambda_neutral_img

        self.ce = nn.CrossEntropyLoss()

        # generator と feature hook を __dict__ に直接格納し、
        # nn.Module のサブモジュール登録をバイパスする（criterion.to(device) の汚染防止）
        if generator is not None:
            object.__setattr__(self, '_generator_ref', generator)
            object.__setattr__(self, '_feat_hook', _FeatureHook(generator.convs[5]))
            print(f"AFSFERLoss: generator registered for L_id and L_feat "
                  f"(lambda_feat={lambda_feat})")
        else:
            object.__setattr__(self, '_generator_ref', None)
            object.__setattr__(self, '_feat_hook',     None)
            print("AFSFERLoss: generator not provided — L_id = L_feat = 0")

        # 画像ベース FER モデル。arcface と同様にパスから内部で新規生成するため、
        # 通常の submodule として登録する（criterion.to(device) で追従させる）。
        # generator/feat_hook は「外部で管理され既に device 上にある共有インスタンス」
        # の二重登録を避けるためにバイパスしているが、こちらは自前で生成するため不要。
        if fer_image_ckpt is not None:
            self.fer_image = FERImageClassifier(fer_image_ckpt)
            print(f"AFSFERLoss: image-based FER model registered for "
                  f"L_expr_img/L_neutral_img (lambda_expr_img={lambda_expr_img}, "
                  f"lambda_neutral_img={lambda_neutral_img})")
        else:
            self.fer_image = None
            print("AFSFERLoss: fer_image_ckpt not provided — "
                  "L_expr_img = L_neutral_img = 0 (設計乖離の修正が無効化された状態)")

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

    def _l_feat(
        self,
        w_tgt: torch.Tensor,
    ) -> torch.Tensor:
        """
        L_feat = MSE(feat32(G(w_new)), feat32(G(w_tgt)))

        feat32_gen は訓練ループ側で G(w_new) を呼んだ際にフックが捕捉済み。
        feat32_tgt はここで G(w_tgt) を no_grad 実行して取得する。
        generator が未設定の場合は 0 を返す。
        """
        feat_hook: Optional[_FeatureHook] = self.__dict__.get('_feat_hook')
        gen_ref = self.__dict__.get('_generator_ref')
        if feat_hook is None or gen_ref is None or feat_hook.feat is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        feat32_gen = feat_hook.feat  # G(w_new) の特徴（勾配あり）

        with torch.no_grad():
            gen_ref(
                [w_tgt],
                input_is_latent=True,
                randomize_noise=False,
                return_latents=False,
            )
            feat32_tgt = feat_hook.feat.detach()  # フックが rebind → 新テンソル

        return F.mse_loss(feat32_gen, feat32_tgt)

    def _l_expr_img(
        self,
        img_gen: Optional[torch.Tensor],
        label_tgt: torch.Tensor,
    ) -> torch.Tensor:
        """
        L_expr_img = CE( R_FER(G(w_new)), label_tgt )

        w_new = (w_src − h_src) + h_tgt にターゲットの表情コードを移植した結果が、
        画像ベース FER モデルから見てもターゲットの表情ラベルとして認識されるかを問う。
        L_id が「G(w_new) がソースの identity か」を ArcFace で確認するのと対称的に、
        こちらは「G(w_new) がターゲットの表情か」を画像ドメインで確認する。
        fer_image が未設定、または img_gen が渡されない場合は 0。
        """
        if self.fer_image is None or img_gen is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        logits = self.fer_image(img_gen)
        return self.ce(logits, label_tgt)

    def _l_neutral_img(
        self,
        img_id_src: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        L_neutral_img = CE( R_FER(G(w_src − h_src)), neutral=4 )

        表情成分を差し引いた残差（identity 側）を実際にデコードした画像が、
        画像ベース FER モデルから見て無表情に見えるかを問う。
        fer_image が未設定、または img_id_src が渡されない場合は 0。
        """
        if self.fer_image is None or img_id_src is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        B = img_id_src.size(0)
        neutral = torch.full((B,), self.NEUTRAL_LABEL, dtype=torch.long,
                              device=img_id_src.device)
        logits = self.fer_image(img_id_src)
        return self.ce(logits, neutral)

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
        img_gen:    Optional[torch.Tensor] = None,
        img_src:    Optional[torch.Tensor] = None,
        img_id_src: Optional[torch.Tensor] = None,
    ):
        l_expr        = self._l_expr(h_src, h_tgt, label_src, label_tgt)
        l_neutral     = self._l_neutral(h_src, h_tgt, w_src, w_tgt)
        l_id          = self._l_id(img_gen, img_src)
        l_sparse      = self._l_sparse(h_src, h_tgt)
        l_cons        = self._l_cons(h_new, h_tgt)
        l_feat        = self._l_feat(w_tgt)
        l_expr_img    = self._l_expr_img(img_gen, label_tgt)
        l_neutral_img = self._l_neutral_img(img_id_src)

        l_total = (self.lambda_expr        * l_expr
                   + self.lambda_id          * l_id
                   + self.lambda_neutral     * l_neutral
                   + self.lambda_sparse      * l_sparse
                   + self.lambda_cons        * l_cons
                   + self.lambda_feat        * l_feat
                   + self.lambda_expr_img    * l_expr_img
                   + self.lambda_neutral_img * l_neutral_img)

        metrics = {
            "expr":        l_expr.item(),
            "id":          l_id.item(),
            "neutral":     l_neutral.item(),
            "sparse":      l_sparse.item(),
            "cons":        l_cons.item(),
            "feat":        l_feat.item(),
            "expr_img":    l_expr_img.item(),
            "neutral_img": l_neutral_img.item(),
        }
        return l_total, metrics
