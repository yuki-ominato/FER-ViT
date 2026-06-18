"""
Reference image providers for AFS training.

Two implementations expose different trade-offs for obtaining img_src / img_tgt:

    GeneratedImageProvider (案A)
        Decodes W+ latents through the frozen StyleGAN2 generator.
        ✓ Domain-consistent: img_gen and references share the StyleGAN2 output space.
        ✗ Requires two extra generator forward passes per training step.

    DiskImageProvider (案B)
        Loads the original image from the path stored in each .pt file.
        ✓ Only one generator call (G(w_new)) per training step.
        ✗ Minor domain gap: real photo vs StyleGAN2 output space.

Both return tensors of shape [B, 3, 256, 256] normalised to [-1, 1],
matching the input expectations of ArcFace and LPIPS.
"""

import abc
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

_TO_256 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class ImageProvider(abc.ABC):
    """Abstract base for reference image providers."""

    @abc.abstractmethod
    def get_images(
        self,
        w_batch: torch.Tensor,
        img_paths: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Return reference images as [B, 3, 256, 256] in [-1, 1].

        Args:
            w_batch:   W+ latents [B, 18, 512] (used by 案A, ignored by 案B).
            img_paths: Original image file paths (used by 案B, ignored by 案A).
            device:    Target device for the returned tensor.
        """


class GeneratedImageProvider(ImageProvider):
    """
    案A: decode W+ through the frozen StyleGAN2 generator.

    Args:
        generator: Frozen StyleGAN2 Generator module.
        face_pool: AdaptiveAvgPool2d((256, 256)) from pSp, used to downsample
                   the 1024-px generator output to 256 px.
    """

    def __init__(
        self,
        generator: nn.Module,
        face_pool: nn.Module,
    ) -> None:
        self.generator = generator
        self.face_pool = face_pool

    @torch.no_grad()
    def get_images(
        self,
        w_batch: torch.Tensor,
        img_paths: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        w = w_batch.to(device)
        imgs, _ = self.generator(
            [w],
            input_is_latent=True,
            randomize_noise=False,
            return_latents=False,
        )
        return self.face_pool(imgs)


class DiskImageProvider(ImageProvider):
    """
    案B: load the original image from the path stored in each .pt file.

    Args:
        img_root: 画像ディレクトリのルートパス（例: ``../dataset/fer2013/train``）。

            - 指定なし: .pt に保存されたパスを CWD 基準で絶対パスに解決して使用。
            - 指定あり: 保存パスの末尾2成分（``class/filename.jpg``）のみを取り出し、
              ``img_root / class / filename.jpg`` として再構成する。
              .pt 生成時のディレクトリ構造が現在の環境と異なる場合に使用する。
    """

    def __init__(self, img_root: Optional[str] = None) -> None:
        self.img_root = img_root

    def _resolve(self, p: str) -> str:
        if self.img_root is not None:
            parts = Path(p).parts
            # 末尾2成分: (class_dir, filename)
            return str(Path(self.img_root) / parts[-2] / parts[-1])
        return os.path.abspath(p)

    def get_images(
        self,
        w_batch: torch.Tensor,
        img_paths: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        imgs = torch.stack(
            [_TO_256(Image.open(self._resolve(p)).convert("RGB")) for p in img_paths],
            dim=0,
        )
        return imgs.to(device)
