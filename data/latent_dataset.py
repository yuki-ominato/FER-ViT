import os
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional


class LatentFERDataset(Dataset):
    """
    StyleGAN潜在コード（w+）を読み込むためのデータセットクラス
    
    潜在コードファイルは以下の形式で保存されている：
    {
        'latent': torch.Tensor,  # shape: [18, 512]
        'label': int,            # 0-6の感情クラス
        'img_path': str          # 元画像のパス
    }
    """
    
    def __init__(self, latent_dir: str):
        """
        Args:
            latent_dir: 潜在コードファイルが保存されているディレクトリ
        """
        self.latent_dir = latent_dir
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
        Args:
            idx: サンプルインデックス
            
        Returns:
            latent: 潜在コード [18, 512]
            label: 感情ラベル (0-6)
        """
        filepath = self.samples[idx]
        
        try:
            data = torch.load(filepath, map_location='cpu', weights_only=True)
            
            # データ構造の検証
            if not isinstance(data, dict):
                raise ValueError(f"Invalid data format in {filepath}")
            
            required_keys = ['latent', 'label']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing key '{key}' in {filepath}")
            
            latent = data['latent']
            label = data['label']
            
            # 形状の検証
            if not isinstance(latent, torch.Tensor):
                raise ValueError(f"Latent is not a tensor in {filepath}")
            
            if latent.shape != (18, 512):
                raise ValueError(f"Invalid latent shape {latent.shape} in {filepath}, expected (18, 512)")
            
            if not isinstance(label, int) or not (0 <= label <= 6):
                raise ValueError(f"Invalid label {label} in {filepath}, expected 0-6")
            
            return latent, label
            
        except Exception as e:
            raise RuntimeError(f"Error loading {filepath}: {e}")
    
    def get_class_counts(self) -> dict:
        """各クラスのサンプル数を取得"""
        class_counts = {}
        
        for filepath in self.samples:
            try:
                data = torch.load(filepath, map_location='cpu', weights_only=True)
                label = data['label']
                class_counts[label] = class_counts.get(label, 0) + 1
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
                continue
        
        return class_counts
    
    def get_class_names(self) -> dict:
        """ラベル番号からクラス名へのマッピング"""
        return {
            0: 'angry',
            1: 'disgust', 
            2: 'fear',
            3: 'happy',
            4: 'neutral',
            5: 'sad',
            6: 'surprise'
        }
