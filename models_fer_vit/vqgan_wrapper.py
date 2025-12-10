import torch
import torch.nn as nn
from omegaconf import OmegaConf
# taming-transformers等のライブラリに依存します
# pip install taming-transformers-rom1504 等が必要
from taming.models.vqgan import VQModel

class VQGANWrapper(nn.Module):
    """
    事前学習済みVQ-GANを用いて画像を離散トークンに変換するラッパー
    """
    def __init__(self, config_path, checkpoint_path, device='cuda'):
        super().__init__()
        self.device = device
        
        # 設定と重みのロード
        config = OmegaConf.load(config_path)
        model = VQModel(**config.model.params)
        
        # 重みの読み込み (state_dictのキー調整が必要な場合があります)
        sd = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)
        
        self.model = model.to(device)
        self.model.eval()
        
        # コードブックサイズ (例: 1024 or 16384)
        self.vocab_size = config.model.params.n_embed

    @torch.no_grad()
    def get_codebook_indices(self, imgs):
        """
        画像バッチ (B, C, H, W) -> トークンID (B, h*w)
        """
        # VQ-GANのエンコーダに通す -> z_q (量子化前), emb_loss, info
        z, _, [_, _, indices] = self.model.encode(imgs)
        
        # indicesは (B, h, w) または (B*h*w) で返ってくるため整形
        # ここでは (B, Seq_Len) を返すように統一
        return indices.reshape(imgs.shape[0], -1)

    @torch.no_grad()
    def decode_indices(self, indices, shape):
        """
        トークンID -> 画像 (再構成の確認用)
        indices: (B, Seq_Len)
        shape: (B, h, w) の潜在空間サイズ
        """
        z_q = self.model.quantize.get_codebook_entry(indices, shape)
        imgs = self.model.decode(z_q)
        return imgs