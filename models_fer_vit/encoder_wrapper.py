import os
import sys
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms

# pSp/e4e のパスを追加（third_party に配置した場合）
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'pixel2style2pixel'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'encoder4editing'))

# プロジェクト内に pixel2style2pixel を直接配置している場合の絶対パス（PYTHONPATH不要化）
_ABS_PSP = '/home/yuki/research2/fer-vit/third_party/pixel2style2pixel'
if os.path.exists(_ABS_PSP) and _ABS_PSP not in sys.path:
    sys.path.append(_ABS_PSP)


class EncoderWrapper:
    """
    pSp / e4e 等の事前学習済みエンコーダ呼び出し用ラッパ。
    
    使用方法:
    1. third_party/pixel2style2pixel または third_party/encoder4editing を配置
    2. 事前学習済み重みを pretrained_models/ に配置
    3. このクラスでエンコーダを初期化して使用
    """

    def __init__(self, model_path: str, device: str = "cuda", encoder_type: str = "psp") -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.encoder_type = encoder_type.lower()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Encoder model not found: {model_path}")

        self.encoder = self._load_encoder()

    def _load_encoder(self):
        """エンコーダをロード"""
        if self.encoder_type == "psp":
            return self._load_psp()
        elif self.encoder_type == "e4e":
            return self._load_e4e()
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

    def _load_psp(self):
        """pSpエンコーダをロード"""
        try:
            from third_party.pixel2style2pixel.models.psp import pSp
            
            print(f"[DEBUG] Loading checkpoint from: {self.model_path}")
            print(f"[DEBUG] Device: {self.device}")
            
            # チェックポイントから opts を読み込み
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
            print(f"[DEBUG] Checkpoint keys: {list(ckpt.keys())}")
            
            opts_dict = ckpt.get('opts', {}) or {}
            print(f"[DEBUG] Original opts keys: {list(opts_dict.keys())}")
            print(f"[DEBUG] Original checkpoint_path: {opts_dict.get('checkpoint_path', 'NOT_FOUND')}")
            
            # 重要な設定を確実に補完
            opts_dict['checkpoint_path'] = self.model_path  # Noneを上書き
            opts_dict.setdefault('device', str(self.device))
            opts_dict.setdefault('encoder_type', 'GradualStyleEncoder')
            opts_dict.setdefault('start_from_latent_avg', True)  # チェックポイントから初期化
            opts_dict.setdefault('learn_in_w', False)
            opts_dict.setdefault('n_styles', 18)
            opts_dict.setdefault('n_style', 18)
            opts_dict.setdefault('input_nc', 3)
            opts_dict.setdefault('output_size', 1024)
            
            print(f"[DEBUG] Final checkpoint_path: {opts_dict['checkpoint_path']}")
            print(f"[DEBUG] start_from_latent_avg: {opts_dict['start_from_latent_avg']}")
            print(f"[DEBUG] encoder_type: {opts_dict['encoder_type']}")
            
            from types import SimpleNamespace
            opts = SimpleNamespace(**opts_dict)
            
            # モデルをロード
            print(f"[DEBUG] Creating pSp model...")
            model = pSp(opts)
            model.eval()
            print(f"[DEBUG] pSp model loaded successfully")
            return model
            
        except ImportError as e:
            print(f"pSp import error: {e}")
            print("Please ensure pixel2style2pixel is in third_party/pixel2style2pixel")
            raise RuntimeError("pSp not available. Please install pixel2style2pixel.")
    # def _load_e4e(self):
    #     """e4eエンコーダをロード"""
    #     try:
    #         # e4eは別のリポジトリなので、pSpの構造を参考に実装
    #         # 実際のe4eリポジトリの構造に合わせて調整が必要
    #         # from models.psp import pSp
    #         from third_party.pixel2style2pixel.models.psp import pSp
    #         from utils.common import tensor2im
            
    #         # e4e設定（pSpと同様の構造を想定）
    #         class Opts:
    #             def __init__(self):
    #                 self.checkpoint_path = self.model_path
    #                 self.device = self.device
    #                 self.output_size = 1024
    #                 self.encoder_type = 'GradualStyleEncoder'
    #                 self.input_nc = 3
    #                 self.output_nc = 3
    #                 self.n_style = 18
    #                 self.start_from_latent_avg = True
    #                 self.learn_in_w = False
    #                 self.label_nc = 0
    #                 self.stylegan_weights = None
            
    #         opts = Opts()
    #         opts.checkpoint_path = self.model_path
    #         opts.device = self.device
            
    #         # モデルをロード（e4eの場合は別の実装が必要）
    #         model = pSp(opts)  # 暫定的にpSpを使用
    #         model.eval()
    #         return model
            
    #     except ImportError as e:
    #         print(f"e4e import error: {e}")
    #         print("Please ensure encoder4editing is in third_party/encoder4editing")
    #         raise RuntimeError("e4e not available. Please install encoder4editing.")

    def preprocess(self, pil_image: Image.Image, resize: int = 256) -> torch.Tensor:
        """画像の前処理"""
        tf = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return tf(pil_image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def encode_image(self, pil_image: Image.Image) -> torch.Tensor:
        """画像を潜在コードにエンコード"""
        if self.encoder is None:
            raise RuntimeError("Encoder not loaded.")
            
        x = self.preprocess(pil_image)
        
        if self.encoder_type == "psp":
            # pSpの推論（エンコーダ部分のみを使用）
            codes = self.encoder.encoder(x)
            # 平均潜在コードを加算
            if self.encoder.opts.start_from_latent_avg:
                if self.encoder.opts.learn_in_w:
                    codes = codes + self.encoder.latent_avg.repeat(codes.shape[0], 1)
                else:
                    codes = codes + self.encoder.latent_avg.repeat(codes.shape[0], 1, 1)
            # codes shape: (1, 18, 512) for w+
            return codes.detach().cpu()
            
        elif self.encoder_type == "e4e":
            # e4eの推論（pSpと同様の構造を想定）
            codes = self.encoder.encoder(x)
            if self.encoder.opts.start_from_latent_avg:
                if self.encoder.opts.learn_in_w:
                    codes = codes + self.encoder.latent_avg.repeat(codes.shape[0], 1)
                else:
                    codes = codes + self.encoder.latent_avg.repeat(codes.shape[0], 1, 1)
            return codes.detach().cpu()
        
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

    def encode_batch(self, pil_images: list, batch_size: int = 4) -> torch.Tensor:
        """バッチで画像をエンコード（メモリ効率化）"""
        all_latents = []
        
        for i in range(0, len(pil_images), batch_size):
            batch_images = pil_images[i:i + batch_size]
            batch_tensors = torch.stack([self.preprocess(img) for img in batch_images])
            
            with torch.no_grad():
                if self.encoder_type == "psp":
                    # エンコーダ部分のみを使用
                    codes = self.encoder.encoder(batch_tensors)
                    if self.encoder.opts.start_from_latent_avg:
                        if self.encoder.opts.learn_in_w:
                            codes = codes + self.encoder.latent_avg.repeat(codes.shape[0], 1)
                        else:
                            codes = codes + self.encoder.latent_avg.repeat(codes.shape[0], 1, 1)
                    latents = codes
                elif self.encoder_type == "e4e":
                    # e4eの推論（pSpと同様の構造を想定）
                    codes = self.encoder.encoder(batch_tensors)
                    if self.encoder.opts.start_from_latent_avg:
                        if self.encoder.opts.learn_in_w:
                            codes = codes + self.encoder.latent_avg.repeat(codes.shape[0], 1)
                        else:
                            codes = codes + self.encoder.latent_avg.repeat(codes.shape[0], 1, 1)
                    latents = codes
                else:
                    raise ValueError(f"Unsupported encoder type: {self.encoder_type}")
            
            all_latents.append(latents.detach().cpu())
        
        return torch.cat(all_latents, dim=0)


