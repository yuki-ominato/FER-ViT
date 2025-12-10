import torch
from torch.utils.data import Dataset
import numpy as np

class TokenDataset(Dataset):
    """
    VQ-GANで生成されたトークンID列を読み込むデータセット
    """
    # RAF-DB / FER2013 共通のクラス定義（順番はgenerate_vq_tokens.pyの保存順に依存）
    CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def __init__(self, token_file_path, transform=None):
        """
        Args:
            token_file_path (str): 'rafdb_train_tokens.pt' へのパス
            transform (callable, optional): MIM用のマスキング変換など
        """
        self.token_file_path = token_file_path
        self.transform = transform

        # データのロード
        # generate_vq_tokens.py で保存した辞書キーに対応させます
        try:
            data = torch.load(token_file_path, map_location='cpu')
            self.tokens = data['tokens']  # shape: (N, Seq_Len)
            self.labels = data['labels']  # shape: (N,)
            
            # テンソル型チェック (int/longであることを保証)
            if self.tokens.dtype != torch.long:
                self.tokens = self.tokens.long()
                
        except KeyError as e:
            raise KeyError(f"指定されたキーがファイル内に見つかりません: {e}. "
                           "['tokens', 'labels'] が含まれているか確認してください。")
        except FileNotFoundError:
            raise FileNotFoundError(f"トークンファイルが見つかりません: {token_file_path}")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        # 1. データの取得
        token_seq = self.tokens[idx].clone() # 変更されないようコピー
        label = self.labels[idx]

        # 2. Transformの適用 (主にMIM用)
        if self.transform:
            token_seq = self.transform(token_seq)

        return token_seq, label

    def get_class_names(self):
        """
        データセット削減機能 (create_subset_dataset) との互換性用
        LatentDatasetと同様にクラス名のリストを返す
        """
        return self.CLASS_NAMES