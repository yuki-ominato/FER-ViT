"""
AFS training losses.

Forward pass flow:
    w_sty_src = h(w_src)
    w_sty_tgt = h(w_tgt)
    w_new     = (w_src - w_sty_src) + w_sty_tgt
    w_sty_new = h(w_new)                           ← for L_cons
    img_gen   = G(w_new)                           ← StyleGAN2 (frozen)
    img_src, img_tgt ← ImageProvider (A or B)

Losses:
    L_id    = 1 - cosine( ArcFace(img_gen), ArcFace(img_src) )
    L_feat  = MSE( feat32(G(w_new)), feat32(G(w_tgt)) )   ← StyleGAN2 32×32 中間特徴
    L_lpips = LPIPS(img_gen, img_tgt)
    L_cons  = L1( h(w_new), stop_grad(h(w_tgt)) )
    L_total = λ_id * L_id + λ_feat * L_feat + λ_lpips * L_lpips + λ_cons * L_cons

StyleGAN2 の 32×32 特徴について
    Generator(1024, 512, 8) の convs はゼロ起算で:
        convs[0,1]: 8×8
        convs[2,3]: 16×16
        convs[4,5]: 32×32  ← ここを使う（convs[5] の出力 [B,512,32,32]）
        convs[6,7]: 64×64
        ...
    フォワードフック (_FeatureHook) を convs[5] に登録して捕捉する。
    feat32_gen は G(w_new) を計算するトレーニングループ側の呼び出しで捕捉され、
    feat32_tgt は criterion.forward 内で G(w_tgt) を no_grad で実行して捕捉する。
"""

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

from models.encoders.model_irse import Backbone      # ArcFace backbone
from criteria.lpips.lpips import LPIPS               # Perceptual loss


# ---------------------------------------------------------------------------
# Forward hook helper
# ---------------------------------------------------------------------------

class _FeatureHook:
    """
    Generator の特定レイヤーの出力テンソルを捕捉するフォワードフック。

    Python の参照セマンティクスにより、フックが再び発火して self.feat が
    新しいテンソルに rebind されても、それ以前に取得した参照は失われない。
    """

    def __init__(self, module: nn.Module) -> None:
        self.feat: Optional[torch.Tensor] = None
        self._handle = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inp, out) -> None:
        self.feat = out          # rebind（in-place ではない）

    def remove(self) -> None:
        self._handle.remove()


# ---------------------------------------------------------------------------
# ArcFace extractor
# ---------------------------------------------------------------------------

class ArcFaceExtractor(nn.Module):
    """
    Frozen ArcFace (IR-SE50) feature extractor.
    入力: [B, 3, 256, 256] in [-1, 1]
    出力: [B, 512] identity embedding
    """

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.net = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.net.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.net.eval()
        self.pool = nn.AdaptiveAvgPool2d((112, 112))
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :, 35:223, 32:220]   # crop to face region (256px input)
        x = self.pool(x)               # → [B, 3, 112, 112]
        return self.net(x)             # → [B, 512]


# ---------------------------------------------------------------------------
# Combined AFS loss
# ---------------------------------------------------------------------------

class AFSLoss(nn.Module):
    """
    AFS 論文の合計損失。

    Args
    ----
    arcface_path : model_ir_se50.pth へのパス。
    generator    : 凍結済み StyleGAN2 Generator。None の場合 L_feat = 0。
    lambda_feat  : L_feat の係数（論文: 3.5）。
    lambda_cons  : L_cons の係数（論文: 0.1）。

    Forward 引数
    ------------
    img_gen   [B,3,256,256]  G(w_new) の生成画像。
    img_src   [B,3,256,256]  ImageProvider が返す人物 A の参照画像。
    img_tgt   [B,3,256,256]  ImageProvider が返す人物 B の参照画像。
    w_sty_new [B,18,512]     h(w_new)。
    w_sty_tgt [B,18,512]     h(w_tgt)（L_cons のターゲット）。
    w_tgt     [B,18,512]     人物 B の W+ 潜在コード（L_feat 計算に使用）。
                              None の場合 L_feat = 0 としてスキップ。

    Returns
    -------
    l_total  : スカラー損失。
    metrics  : ログ用の各損失値 dict {"id", "feat", "lpips", "cons"}。
    """

    # generator は nn.Module ではなく外部から管理されるため、
    # nn.Module の __setattr__ を経由させずに保持する。
    # これにより criterion.to(device) や criterion.parameters() に影響しない。

    def __init__(
        self,
        arcface_path: str,
        generator: Optional[nn.Module] = None,
        lambda_feat: float = 3.5,
        lambda_cons: float = 0.1,
    ) -> None:
        super().__init__()
        self.arcface = ArcFaceExtractor(arcface_path)
        self.lpips   = LPIPS(net_type='alex')
        self.lambda_feat = lambda_feat
        self.lambda_cons = lambda_cons

        for p in self.lpips.parameters():
            p.requires_grad_(False)

        # generator と feature hook を __dict__ に直接格納し、
        # nn.Module のサブモジュール登録をバイパスする
        if generator is not None:
            # 32×32 特徴: Generator(1024,...) では convs[5] の出力
            object.__setattr__(self, '_generator_ref', generator)
            object.__setattr__(self, '_feat_hook', _FeatureHook(generator.convs[5]))
            print(f"AFSLoss: StyleGAN2 feature hook registered on convs[5] "
                  f"(lambda_feat={lambda_feat})")
        else:
            object.__setattr__(self, '_generator_ref', None)
            object.__setattr__(self, '_feat_hook',     None)
            print("AFSLoss: generator not provided, L_feat = 0")

    def forward(
        self,
        img_gen:   torch.Tensor,
        img_src:   torch.Tensor,
        img_tgt:   torch.Tensor,
        w_sty_new: torch.Tensor,
        w_sty_tgt: torch.Tensor,
        w_tgt:     Optional[torch.Tensor] = None,
    ):
        # --- L_id: Identity loss (ArcFace cosine) ---
        with torch.no_grad():
            feat_src = self.arcface(img_src)   # 固定参照; グラフ不要
        feat_gen = self.arcface(img_gen)       # G(w_new) 経由で h に勾配
        l_id = (1.0 - F.cosine_similarity(feat_gen, feat_src, dim=1)).mean()

        # --- L_feat: StyleGAN2 32×32 feature loss ---
        feat_hook: Optional[_FeatureHook] = self.__dict__.get('_feat_hook')
        gen_ref = self.__dict__.get('_generator_ref')

        if feat_hook is not None and gen_ref is not None and w_tgt is not None:
            # feat_hook.feat は直前のトレーニングループ側 G(w_new) 呼び出しで捕捉済み。
            # Python の rebind セマンティクスにより、次の呼び出しで feat が上書きされても
            # ここで取得した参照は影響を受けない。
            feat32_gen = feat_hook.feat               # [B, 512, 32, 32]、勾配あり

            with torch.no_grad():
                gen_ref(
                    [w_tgt],
                    input_is_latent=True,
                    randomize_noise=False,
                    return_latents=False,
                )
                feat32_tgt = feat_hook.feat.detach()  # フックが rebind → 新テンソル

            l_feat = F.mse_loss(feat32_gen, feat32_tgt)
        else:
            # ── DEBUG: L_feat = 0 の原因を特定したら以下3行を削除 ──
            if not hasattr(AFSLoss, '_feat_debug_printed'):
                AFSLoss._feat_debug_printed = True
                print(f"[DEBUG] L_feat=0: feat_hook={feat_hook!r}, "
                      f"gen_ref={gen_ref is not None}, "
                      f"w_tgt={w_tgt is not None}, "
                      f"hook.feat={feat_hook.feat if feat_hook else 'N/A'}")
            # ─────────────────────────────────────────────────────────
            l_feat = torch.tensor(0.0, device=img_gen.device)

        # --- L_lpips: Perceptual loss ---
        l_lpips = self.lpips(img_gen, img_tgt)

        # --- L_cons: Consistency loss ---
        l_cons = F.l1_loss(w_sty_new, w_sty_tgt.detach())

        # --- Total ---
        l_total = (l_id
                   + self.lambda_feat * l_feat
                   + l_lpips
                   + self.lambda_cons * l_cons)

        metrics = {
            "id":    l_id.item(),
            "feat":  l_feat.item(),
            "lpips": l_lpips.item(),
            "cons":  l_cons.item(),
        }
        return l_total, metrics
