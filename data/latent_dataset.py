import os
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Callable, Union

class LatentAugment:
    """
    潜在空間向けのデータ拡張クラス
    画像空間でのAugmentation（回転、色調補正など）に相当する効果を
    潜在空間上でのノイズ付加やスケーリングで模倣する。
    """
    def __init__(
        self,
        noise_std: float = 0.0,
        scale_range: Optional[Tuple[float, float]] = None,
        mask_prob: float = 0.0
    ):
        """
        Args:
            noise_std: ガウシアンノイズの標準偏差 (0.0で無効)
            scale_range: スケーリング係数の範囲 (min, max) (Noneで無効)
            mask_prob: 特徴量ドロップアウトの確率 (0.0で無効)
        """
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.mask_prob = mask_prob

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        # 元のデータを変更しないようにクローン
        out = latent.clone()
        
        with torch.no_grad():
            # 1. Additive Gaussian Noise
            if self.noise_std > 0:
                noise = torch.randn_like(out) * self.noise_std
                out = out + noise
            
            # 2. Scaling (Intensity Perturbation)
            if self.scale_range is not None:
                min_s, max_s = self.scale_range
                scale = torch.empty(1).uniform_(min_s, max_s).item()
                out = out * scale
                
            # 3. Channel Masking (Feature Dropout / Cutout equivalent)
            if self.mask_prob > 0:
                mask = torch.rand_like(out) > self.mask_prob
                out = out * mask.float()
                
        return out    

class LatentFERDataset(Dataset):
    """
    StyleGAN潜在コード（w+）を読み込むためのデータセットクラス
    """
    
    def __init__(
        self, 
        latent_dir: str, 
        transform: Optional[Callable] = None
    ):
        """
        Args:
            latent_dir: 潜在コードファイルが保存されているディレクトリ
            transform: 潜在コードへの変換処理 (LatentAugmentなど)
        """
        self.latent_dir = latent_dir
        self.transform = transform
        self.samples = self._load_samples()
        
    def _load_samples(self) -> list:
        """潜在コードファイルのリストを作成"""
        samples = []
        
        if not os.path.exists(self.latent_dir):
            raise FileNotFoundError(f"Latent directory not found: {self.latent_dir}")
        
        # .ptファイルをすべて取得
        for filename in sorted(os.listdir(self.latent_dir)):
            if filename.endswith('.pt'):
                filepath = os.path.join(self.latent_dir, filename)
                samples.append(filepath)
        
        if not samples:
            raise ValueError(f"No .pt files found in {self.latent_dir}")
        
        print(f"Loaded {len(samples)} latent samples from {self.latent_dir}")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            latent: 潜在コード [18, 512]
            label: 感情ラベル (0-6)
        """
        filepath = self.samples[idx]
        
        try:
            data = torch.load(filepath, map_location='cpu', weights_only=True)
            
            # 必須キーの確認等は省略せず実装（安全のため）
            latent = data['latent']
            label = data['label']
            
            # Transform (Augmentation) の適用
            if self.transform:
                latent = self.transform(latent)
            
            return latent, label
            
        except Exception as e:
            # 破損ファイル等はスキップまたはエラー送出（ここではエラー送出）
            raise RuntimeError(f"Error loading {filepath}: {e}")
    
    def get_class_counts(self) -> dict:
        """各クラスのサンプル数を取得 (簡易実装)"""
        # 注意: 全ファイルをロードするため時間がかかる場合がある
        # 必要に応じてキャッシュする仕組みを推奨
        class_counts = {}
        for filepath in self.samples:
            try:
                data = torch.load(filepath, map_location='cpu', weights_only=True)
                label = data['label']
                class_counts[label] = class_counts.get(label, 0) + 1
            except:
                continue
        return class_counts
    
    def get_class_names(self) -> dict:
        return {
            0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
            4: 'neutral', 5: 'sad', 6: 'surprise'
        }


def get_latent_train_transforms(
    noise_std: float = 0.1,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    mask_prob: float = 0.1
) -> LatentAugment:
    """
    学習用のデータ拡張を取得
    image_dataset.py の get_train_transforms に相当
    """
    return LatentAugment(
        noise_std=noise_std,
        scale_range=scale_range,
        mask_prob=mask_prob
    )

def get_latent_val_transforms() -> None:
    """
    検証用の変換
    image_dataset.py の get_val_transforms に相当。
    潜在空間では通常、検証時に正規化以外の変換は行わないためNoneを返すか、
    恒等変換を返す。
    """
    return None