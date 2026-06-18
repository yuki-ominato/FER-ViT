"""
AFS training losses.

Forward pass flow:
    w_sty_src = h(w_src)
    w_sty_tgt = h(w_tgt)
    w_new     = (w_src - w_sty_src) + w_sty_tgt
    w_sty_new = h(w_new)                           ← for L_cons
    img_gen   = G(w_new)                           ← StyleGAN2 (frozen)
    img_src, img_tgt ← ImageProvider (A or B)

Losses (all computed inside AFSLoss.forward):
    L_id    = 1 - cosine( ArcFace(img_gen), ArcFace(img_src) )
    L_lpips = LPIPS(img_gen, img_tgt)
    L_cons  = L1( h(w_new), stop_grad(h(w_tgt)) )
    L_total = L_id + L_lpips + λ_cons * L_cons

ArcFace(img_src) is computed under torch.no_grad() because img_src is a
fixed reference — gradients only need to flow through ArcFace(img_gen).
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Resolve pSp directory and add to path so we can reuse its modules.
_PSP_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'third_party', 'pixel2style2pixel')
)
if _PSP_ROOT not in sys.path:
    sys.path.insert(0, _PSP_ROOT)

from models.encoders.model_irse import Backbone      # ArcFace backbone
from criteria.lpips.lpips import LPIPS               # Perceptual loss


class ArcFaceExtractor(nn.Module):
    """
    Frozen ArcFace (IR-SE50) feature extractor.

    Replicates the preprocessing from pSp's IDLoss:
        1. Crop the face region from a 256×256 image.
        2. Pool to 112×112 (ArcFace input size).
        3. Extract the 512-d identity embedding.

    Parameters
    ----------
    model_path:
        Absolute path to model_ir_se50.pth.
        Typically: <repo>/third_party/pixel2style2pixel/pretrained_models/model_ir_se50.pth
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
        # x: [B, 3, 256, 256] in [-1, 1]
        x = x[:, :, 35:223, 32:220]   # crop face region (from 256-px input)
        x = self.pool(x)               # → [B, 3, 112, 112]
        return self.net(x)             # → [B, 512]


class AFSLoss(nn.Module):
    """
    Combined AFS loss.

    Args:
        arcface_path: Absolute path to model_ir_se50.pth.
        lambda_cons:  Weight for the consistency loss term (default 0.1).

    Forward
    -------
    Inputs:
        img_gen    [B, 3, 256, 256]  Generated image G(w_new).
        img_src    [B, 3, 256, 256]  Reference image of person A (from ImageProvider).
        img_tgt    [B, 3, 256, 256]  Reference image of person B (from ImageProvider).
        w_sty_new  [B, 18, 512]      h(w_new)  — style of the swapped latent.
        w_sty_tgt  [B, 18, 512]      h(w_tgt)  — style of the target (used as detached target).

    Returns:
        l_total  Scalar loss tensor (gradients flow through img_gen and w_sty_new).
        metrics  Dict with float values for logging: "id", "lpips", "cons".
    """

    def __init__(self, arcface_path: str, lambda_cons: float = 0.1) -> None:
        super().__init__()
        self.arcface = ArcFaceExtractor(arcface_path)
        self.lpips = LPIPS(net_type='alex')
        self.lambda_cons = lambda_cons

        # LPIPS parameters are also frozen.
        for p in self.lpips.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        img_gen: torch.Tensor,
        img_src: torch.Tensor,
        img_tgt: torch.Tensor,
        w_sty_new: torch.Tensor,
        w_sty_tgt: torch.Tensor,
    ):
        # --- Identity loss ---
        # img_src is a fixed reference; compute its features without building a graph.
        with torch.no_grad():
            feat_src = self.arcface(img_src)
        feat_gen = self.arcface(img_gen)   # gradient flows back to img_gen → w_new → h
        l_id = (1.0 - F.cosine_similarity(feat_gen, feat_src, dim=1)).mean()

        # --- LPIPS loss ---
        # img_tgt has no gradient; only img_gen contributes to the gradient.
        l_lpips = self.lpips(img_gen, img_tgt)

        # --- Consistency loss ---
        # h(w_swap) should match h(w_tgt).  Stop gradient on the target side
        # so that only h(w_swap) = h(w_new) is pushed toward h(w_tgt), not vice-versa.
        l_cons = F.l1_loss(w_sty_new, w_sty_tgt.detach())

        l_total = l_id + l_lpips + self.lambda_cons * l_cons

        metrics = {
            "id": l_id.item(),
            "lpips": l_lpips.item(),
            "cons": l_cons.item(),
        }
        return l_total, metrics
