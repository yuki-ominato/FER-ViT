"""
Dataset that returns random (src, tgt) pairs of W+ latents for AFS training.

Each .pt file in latent_dir is expected to contain:
    {
        "latent":   torch.Tensor [18, 512],
        "label":    int,
        "img_path": str,   # path to the original image (used by DiskImageProvider)
    }

Files without "img_path" are supported but incompatible with DiskImageProvider.

The target index is sampled uniformly at each __getitem__ call, excluding idx == tgt_idx
to avoid trivial self-swaps.  No identity-based pairing is enforced here; the loss
functions handle whatever pairs are provided.
"""

import os
import random
from typing import Tuple

import torch
from torch.utils.data import Dataset


class PairLatentDataset(Dataset):
    """
    Args:
        latent_dir: Directory containing .pt latent cache files.

    Returns per item:
        w_src      [18, 512]
        label_src  int
        path_src   str
        w_tgt      [18, 512]
        label_tgt  int
        path_tgt   str
    """

    def __init__(self, latent_dir: str) -> None:
        # 絶対パスに変換する。StyleGAN2 の CUDA コンパイル等で CWD が変わっても
        # self.files のパスが壊れないようにするため。
        latent_dir = os.path.abspath(latent_dir)

        if not os.path.isdir(latent_dir):
            raise FileNotFoundError(f"Latent directory not found: {latent_dir}")

        self.files = sorted(
            os.path.join(latent_dir, f)
            for f in os.listdir(latent_dir)
            if f.endswith('.pt')
        )

        if len(self.files) < 2:
            raise ValueError(
                f"Need at least 2 latent files for pairing, found {len(self.files)} in {latent_dir}"
            )

        print(f"PairLatentDataset: {len(self.files)} samples from {latent_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, int, str, torch.Tensor, int, str]:
        src = self._load(idx)

        # Sample a distinct target index.
        tgt_idx = random.randrange(len(self.files))
        while tgt_idx == idx:
            tgt_idx = random.randrange(len(self.files))
        tgt = self._load(tgt_idx)

        return (
            src['latent'],
            src['label'],
            src['img_path'],
            tgt['latent'],
            tgt['label'],
            tgt['img_path'],
        )

    def _load(self, idx: int) -> dict:
        data = torch.load(self.files[idx], map_location='cpu', weights_only=False)
        return {
            'latent': data['latent'],            # [18, 512]
            'label': int(data['label']),
            'img_path': data.get('img_path', ''),
        }
