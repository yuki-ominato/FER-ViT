import os
import sys
from types import SimpleNamespace
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_PSP_ROOT = os.path.join(_PROJECT_ROOT, 'third_party', 'pixel2style2pixel')
_E4E_ROOT = os.path.join(_PROJECT_ROOT, 'third_party', 'encoder4editing')

# モジュールロード時は sys.path を変更しない。
# pSp / e4e はそれぞれ同名の `models/` パッケージを持つため、
# 両方を同時に先頭に追加すると衝突が起きる。
# sys.path への追加は EncoderWrapper.__init__ でエンコーダータイプに応じて行う。


class EncoderWrapper:
    """
    pSp / e4e 事前学習済みエンコーダ共通ラッパ。

    サポートする encoder_type:
        "psp" : pixel2style2pixel (third_party/pixel2style2pixel)
        "e4e" : encoder4editing   (third_party/encoder4editing)

    どちらも同じインターフェース（encoder + latent_avg 加算）を持つため、
    encode_image / encode_batch は共通実装を使用する。
    """

    def __init__(self, model_path: str, device: str = "cuda", encoder_type: str = "psp") -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.encoder_type = encoder_type.lower()

        if self.encoder_type not in ("psp", "e4e"):
            raise ValueError(f"encoder_type は 'psp' または 'e4e' で指定してください: {encoder_type}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Encoder model not found: {model_path}")

        # エンコーダータイプに合わせて sys.path を設定してからロード
        self._setup_path()
        self.encoder = self._load_encoder()

    def _setup_path(self) -> None:
        """encoder_type に対応するリポジトリルートを sys.path 先頭に追加する。"""
        root = _PSP_ROOT if self.encoder_type == "psp" else _E4E_ROOT
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"{'pSp' if self.encoder_type == 'psp' else 'encoder4editing'} が見つかりません: {root}\n"
                f"third_party/ に配置してください。"
            )
        if root not in sys.path:
            sys.path.insert(0, root)

    # ------------------------------------------------------------------
    # Loader
    # ------------------------------------------------------------------

    def _load_encoder(self):
        if self.encoder_type == "psp":
            return self._load_psp()
        else:
            return self._load_e4e()

    def _load_psp(self):
        """
        pSp エンコーダをロード。
        _setup_path() により _PSP_ROOT が sys.path 先頭にある前提。
        """
        try:
            from models.psp import pSp  # pSp の models/psp.py
        except ImportError as e:
            raise RuntimeError(
                f"pSp のインポートに失敗しました: {e}\n"
                f"third_party/pixel2style2pixel が正しく配置されているか確認してください。"
            )

        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        opts_dict = dict(ckpt.get('opts', {}) or {})
        opts_dict['checkpoint_path']        = self.model_path
        opts_dict.setdefault('device',                  str(self.device))
        opts_dict.setdefault('encoder_type',            'GradualStyleEncoder')
        opts_dict.setdefault('start_from_latent_avg',   True)
        opts_dict.setdefault('learn_in_w',              False)
        opts_dict.setdefault('n_styles',                18)
        opts_dict.setdefault('n_style',                 18)
        opts_dict.setdefault('input_nc',                3)
        opts_dict.setdefault('output_size',             1024)

        model = pSp(SimpleNamespace(**opts_dict))
        model.to(self.device).eval()
        print(f"pSp loaded: {self.model_path}")
        return model

    def _load_e4e(self):
        """
        e4e (encoder4editing) エンコーダをロード。
        _setup_path() により _E4E_ROOT が sys.path 先頭にある前提。

        encoder4editing は pSp をベースにしており、モデルクラス名も pSp だが
        encoder_type が 'Encoder4Editing' である点が異なる。
        推論インターフェース（encoder + latent_avg）は pSp と同一。
        """
        try:
            from models.psp import pSp  # encoder4editing の models/psp.py
        except ImportError as e:
            raise RuntimeError(
                f"e4e のインポートに失敗しました: {e}\n"
                f"third_party/encoder4editing が正しく配置されているか確認してください。"
            )

        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        opts_dict = dict(ckpt.get('opts', {}) or {})
        opts_dict['checkpoint_path']        = self.model_path
        opts_dict.setdefault('device',                  str(self.device))
        opts_dict.setdefault('encoder_type',            'Encoder4Editing')
        opts_dict.setdefault('start_from_latent_avg',   True)
        opts_dict.setdefault('learn_in_w',              False)
        opts_dict.setdefault('n_styles',                18)
        opts_dict.setdefault('n_style',                 18)
        opts_dict.setdefault('input_nc',                3)
        opts_dict.setdefault('output_size',             1024)

        model = pSp(SimpleNamespace(**opts_dict))
        model.to(self.device).eval()
        print(f"e4e loaded: {self.model_path}")
        return model

    # ------------------------------------------------------------------
    # Encoding helpers (pSp / e4e 共通)
    # ------------------------------------------------------------------

    def _apply_latent_avg(self, codes: torch.Tensor) -> torch.Tensor:
        """start_from_latent_avg が True の場合に latent_avg を加算する。"""
        if not self.encoder.opts.start_from_latent_avg:
            return codes
        avg = self.encoder.latent_avg
        if self.encoder.opts.learn_in_w:
            return codes + avg.repeat(codes.shape[0], 1)
        else:
            return codes + avg.repeat(codes.shape[0], 1, 1)

    def preprocess(self, pil_image: Image.Image, resize: int = 256) -> torch.Tensor:
        """PIL → [C,H,W] tensor（[-1,1]、device 転送済み）"""
        tf = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return tf(pil_image).to(self.device)

    @torch.no_grad()
    def encode_image(self, pil_image: Image.Image) -> torch.Tensor:
        """1 枚の画像を W+ 潜在コード [1, 18, 512] にエンコード（CPU テンソル）"""
        x = self.preprocess(pil_image).unsqueeze(0)   # [1,C,H,W]
        codes = self.encoder.encoder(x)
        codes = self._apply_latent_avg(codes)
        return codes.detach().cpu()

    @torch.no_grad()
    def encode_batch(self, pil_images: list, batch_size: int = 4) -> torch.Tensor:
        """複数枚をバッチ処理して W+ 潜在コード [N, 18, 512] を返す（CPU テンソル）"""
        all_latents = []
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i + batch_size]
            x = torch.stack([self.preprocess(img) for img in batch], dim=0)  # [B,C,H,W]
            codes = self.encoder.encoder(x)
            codes = self._apply_latent_avg(codes)
            all_latents.append(codes.detach().cpu())
        return torch.cat(all_latents, dim=0)
