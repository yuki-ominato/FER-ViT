"""
画像から直接学習するためのデータセット
FER2013 / RAF-DB 対応
"""

import csv
import os
from typing import Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ImageFERDataset(Dataset):
    """
    画像から直接感情認識を行うデータセット
    
    ディレクトリ構造:
    data_root/
        angry/
            img1.png
            img2.png
        happy/
            img1.png
            img2.png
        ...
    """
    
    CLASS_TO_LABEL = {
        "angry": 0,
        "disgust": 1,
        "fear": 2,
        "happy": 3,
        "neutral": 4,
        "sad": 5,
        "surprise": 6,
    }
    
    LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}
    
    def __init__(
        self,
        data_root: str,
        transform: Optional[Callable] = None,
        img_size: int = 224,
    ):
        """
        Args:
            data_root: データルートディレクトリ
            transform: カスタム変換（Noneの場合はデフォルト変換を使用）
            img_size: 画像サイズ（デフォルト224はViT標準）
        """
        self.data_root = data_root
        self.img_size = img_size
        
        # デフォルト変換
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet統計
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transform
        
        # 画像パスとラベルのリストを作成
        self.samples = []
        self._load_samples()
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {data_root}")
        
        print(f"Loaded {len(self.samples)} images from {data_root}")
        self._print_class_distribution()
    
    def _load_samples(self):
        """ディレクトリから画像とラベルを読み込み"""
        for class_name in sorted(os.listdir(self.data_root)):
            class_dir = os.path.join(self.data_root, class_name)
            
            if not os.path.isdir(class_dir):
                continue
            
            label = self.CLASS_TO_LABEL.get(class_name.lower())
            if label is None:
                print(f"Warning: Unknown class '{class_name}', skipping...")
                continue
            
            # クラスディレクトリ内の画像を読み込み
            for img_name in sorted(os.listdir(class_dir)):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, label))
    
    def _print_class_distribution(self):
        """クラス分布を表示"""
        from collections import Counter
        labels = [label for _, label in self.samples]
        counter = Counter(labels)
        
        print("\nClass distribution:")
        for label_id in sorted(counter.keys()):
            class_name = self.LABEL_TO_CLASS[label_id]
            count = counter[label_id]
            print(f"  {class_name:>8s} (id={label_id}): {count:>5d} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: (C, H, W) tensor
            label: 感情クラスID (0-6)
        """
        img_path, label = self.samples[idx]
        
        # 画像読み込み
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # エラー時は黒画像を返す
            image = Image.new('RGB', (self.img_size, self.img_size), color='black')
        
        # 変換適用
        if self.transform:
            image = self.transform(image)
        
        return image, label


class RAFDBDataset(Dataset):
    """
    RAF-DB データセット (CSV ラベルファイル + 番号付きサブディレクトリ形式)

    ディレクトリ構造:
    data_root/
        train/
            1/   (Surprise)
            2/   (Fear)
            ...
            7/   (Neutral)
        test/
            1/ ... 7/
        train_labels.csv
        test_labels.csv

    内部ラベル (0-6) は ImageFERDataset と共通。
    """

    # RAF-DB 公式ラベル (1-7) → 共通 0-6 へのマッピング
    RAFDB_TO_LABEL = {
        1: 6,  # Surprise
        2: 2,  # Fear
        3: 1,  # Disgust
        4: 3,  # Happiness
        5: 5,  # Sadness
        6: 0,  # Anger
        7: 4,  # Neutral
    }

    CLASS_TO_LABEL = ImageFERDataset.CLASS_TO_LABEL
    LABEL_TO_CLASS = ImageFERDataset.LABEL_TO_CLASS

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        csv_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        img_size: int = 224,
    ):
        """
        Args:
            data_root: RAF-DB のルートディレクトリ (train/, test/, *.csv を含む)
            split: 'train' または 'test'
            csv_path: ラベル CSV のパス。None の場合は data_root/{split}_labels.csv を使用
            transform: カスタム変換
            img_size: 画像サイズ
        """
        self.data_root = data_root
        self.split = split
        self.img_size = img_size

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transform

        if csv_path is None:
            csv_path = os.path.join(data_root, f'{split}_labels.csv')

        self.samples = []
        self._load_samples(csv_path)

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {data_root} (split={split})")

        print(f"Loaded {len(self.samples)} images from {data_root} [{split}]")
        self._print_class_distribution()

    def _load_samples(self, csv_path: str):
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row['image']
                raf_label = int(row['label'])
                label = self.RAFDB_TO_LABEL[raf_label]
                img_path = os.path.join(self.data_root, self.split, str(raf_label), img_name)
                self.samples.append((img_path, label))

    def _print_class_distribution(self):
        from collections import Counter
        labels = [label for _, label in self.samples]
        counter = Counter(labels)
        print("\nClass distribution:")
        for label_id in sorted(counter.keys()):
            class_name = self.LABEL_TO_CLASS[label_id]
            count = counter[label_id]
            print(f"  {class_name:>8s} (id={label_id}): {count:>5d} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (self.img_size, self.img_size), color='black')
        if self.transform:
            image = self.transform(image)
        return image, label


def create_image_dataset(
    dataset_type: str,
    path: str,
    split: str = 'train',
    transform: Optional[Callable] = None,
    img_size: int = 224,
) -> Dataset:
    """
    データセット種別に応じた Dataset を返すファクトリ関数。

    Args:
        dataset_type: 'fer2013' または 'raf-db'
        path: FER2013 の場合はクラスサブディレクトリを含むディレクトリ、
              RAF-DB の場合はルートディレクトリ (train_labels.csv 等を含む)
        split: RAF-DB のみ使用 ('train' / 'test')
        transform: カスタム変換
        img_size: 画像サイズ
    """
    if dataset_type == 'fer2013':
        return ImageFERDataset(path, transform=transform, img_size=img_size)
    elif dataset_type == 'raf-db':
        return RAFDBDataset(path, split=split, transform=transform, img_size=img_size)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}. Choose 'fer2013' or 'raf-db'.")


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """学習用のデータ拡張"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """検証用の変換（拡張なし）"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])