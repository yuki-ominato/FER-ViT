"""
画像ベース FER 分類器のラッパー。

`document/AFS_FER.md` が当初提案していた設計

    L_expr = 1 - cos( R_FER(G(w_expr + w_rest^target)), R_FER(x_expr_source) )

は「生成画像を FER モデルに通して評価する」ことを前提にしていたが、実装
(`afs/fer_losses.py` の旧 `_l_expr`/`_l_neutral`) は Generator を経由せず
StyleExtractor の出力 h(w) に直接、正則化なしの小さな MLP をかけるだけになっていた。

このモジュールは `train/train_image_vit.py` で学習済みの画像ベース ImageViT
(または timm 事前学習モデル) を凍結して読み込み、StyleGAN2 の生成画像
(G(w) など、[-1, 1] 正規化) をその入力仕様（224×224 / ImageNet 正規化）に
変換したうえで分類する。これにより L_expr/L_neutral を「画像として実際に
その表情に見えるか」で評価できるようにする。
"""

import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models_fer_vit.image_vit import ImageViT, create_vit_small, create_vit_base, create_vit_tiny

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _build_model(model_cfg: dict) -> nn.Module:
    num_classes = model_cfg.get("num_classes", 7)
    img_size = model_cfg.get("img_size", 224)

    if model_cfg.get("use_pretrained"):
        import timm

        return timm.create_model(
            "vit_small_patch16_224", pretrained=False, num_classes=num_classes
        )

    model_size = model_cfg.get("model_size", "custom")
    if model_size == "tiny":
        return create_vit_tiny(num_classes=num_classes, img_size=img_size)
    if model_size == "small":
        return create_vit_small(num_classes=num_classes, img_size=img_size)
    if model_size == "base":
        return create_vit_base(num_classes=num_classes, img_size=img_size)
    return ImageViT(
        img_size=img_size,
        patch_size=model_cfg.get("patch_size", 16),
        embed_dim=model_cfg.get("embed_dim", 512),
        depth=model_cfg.get("depth", 6),
        heads=model_cfg.get("heads", 8),
        mlp_dim=model_cfg.get("mlp_dim", 2048),
        num_classes=num_classes,
        dropout=model_cfg.get("dropout", 0.1),
    )


class FERImageClassifier(nn.Module):
    """
    train_image_vit.py で学習した画像ベース FER モデルを凍結して読み込むラッパー。

    入力: [B, 3, H, W]  StyleGAN2 / face_pool の出力そのもの ([-1, 1] 正規化)
    出力: [B, num_classes]  ロジット

    Args:
        ckpt_path: `<out_dir>/<run_id>/checkpoints/best_model.pt` (または last_model.pt)。
                   同じ run_id ディレクトリ直下の config.json からアーキテクチャを復元する。
    """

    def __init__(self, ckpt_path: str) -> None:
        super().__init__()

        run_dir = os.path.dirname(os.path.dirname(ckpt_path))
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        model_cfg = config["model"]

        self.img_size = model_cfg.get("img_size", 224)
        self.net = _build_model(model_cfg)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        self.net.load_state_dict(state)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

        self.register_buffer("mean", _IMAGENET_MEAN.clone())
        self.register_buffer("std", _IMAGENET_STD.clone())

        print(f"FERImageClassifier: loaded {ckpt_path} "
              f"(model_size={model_cfg.get('model_size')}, img_size={self.img_size})")

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # [-1, 1] -> [0, 1]
        x = (img.clamp(-1.0, 1.0) + 1.0) / 2.0
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                               mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.net(x)
