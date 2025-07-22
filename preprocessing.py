"""
FER2013データセット用前処理ライブラリ
表情認識用データセットの読み込み、前処理、分析機能を提供
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter

class FER2013Dataset(Dataset):
    """FER2013データセット用のカスタムDatasetクラス"""
    
    def __init__(self, root_dir, transform=None, split='train'):
        """
        Args:
            root_dir (str): データセットのルートディレクトリ
            transform: 適用する前処理変換
            split (str): 'train', 'val', 'test'のいずれか
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # 感情ラベルの定義
        self.emotion_labels = {
            'angry': 0,
            'disgust': 1,
            'fear': 2,
            'happy': 3,
            'neutral': 4,
            'sad': 5,
            'surprise': 6
        }
        
        # 感情名のリスト（推論時に使用）
        self.emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 
                             'Neutral', 'Sad', 'Surprise']
        
        # データパスとラベルのリストを作成
        self.data_list = []
        self._load_data_paths()
        
    def _load_data_paths(self):
        """データパスとラベルのリストを作成"""
        dataset_path = os.path.join(self.root_dir, self.split)
        
        for emotion_name, label in self.emotion_labels.items():
            emotion_dir = os.path.join(dataset_path, emotion_name)
            
            if os.path.exists(emotion_dir):
                for filename in os.listdir(emotion_dir):
                    if filename.endswith('.png'):
                        file_path = os.path.join(emotion_dir, filename)
                        self.data_list.append((file_path, label))
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        
        # 画像を読み込み
        image = Image.open(img_path).convert('RGB')
        
        # 前処理変換を適用
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """クラスの分布を取得"""
        labels = [item[1] for item in self.data_list]
        return Counter(labels)

def get_fer2013_transforms(input_size=224, augment=True):
    """FER2013用データ変換パイプラインを定義"""
    
    if augment:
        # 学習用データ拡張（FER2013に適した変換）
        train_transforms = transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.Grayscale(num_output_channels=3),  # グレースケールをRGBに変換
            transforms.RandomCrop((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # ViT用正規化
        ])
        
        # 検証・テスト用（データ拡張なし）
        val_transforms = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        return train_transforms, val_transforms
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def create_fer2013_data_loaders(root_dir, batch_size=32, input_size=224, 
                               num_workers=4, val_split=0.2):
    """FER2013データローダーを作成"""
    
    # データ変換を取得
    train_transforms, val_transforms = get_fer2013_transforms(
        input_size=input_size, augment=True
    )
    
    # 訓練データセットを作成
    train_dataset = FER2013Dataset(
        root_dir=root_dir, 
        transform=train_transforms, 
        split='train'
    )
    
    # 訓練データを訓練用と検証用に分割
    if val_split > 0:
        train_indices, val_indices = train_test_split(
            range(len(train_dataset)), 
            test_size=val_split, 
            random_state=42,
            stratify=[train_dataset.data_list[i][1] for i in range(len(train_dataset))]
        )
        
        # サブセットを作成
        train_subset = Subset(train_dataset, train_indices)
        
        # 検証用データセット（変換のみ変更）
        val_dataset = FER2013Dataset(
            root_dir=root_dir, 
            transform=val_transforms, 
            split='train'
        )
        val_subset = Subset(val_dataset, val_indices)
        
        # データローダーを作成
        train_loader = DataLoader(
            train_subset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        )
        val_loader = None
    
    # テストデータローダー（存在する場合）
    test_loader = None
    test_dir = os.path.join(root_dir, 'test')
    if os.path.exists(test_dir):
        test_dataset = FER2013Dataset(
            root_dir=root_dir, 
            transform=val_transforms, 
            split='test'
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader

def analyze_fer2013_dataset(root_dir):
    """FER2013データセットの統計情報を分析"""
    
    print("=== FER2013 データセット分析 ===\n")
    
    emotion_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # 各分割のデータ数を確認
    for split in ['train', 'test']:
        split_path = os.path.join(root_dir, split)
        if os.path.exists(split_path):
            print(f"{split.upper()} データ:")
            total_count = 0
            
            for emotion in emotion_names:
                emotion_path = os.path.join(split_path, emotion)
                if os.path.exists(emotion_path):
                    count = len([f for f in os.listdir(emotion_path) 
                               if f.endswith('.png')])
                    print(f"  {emotion.capitalize()}: {count}")
                    total_count += count
            
            print(f"  合計: {total_count}\n")

def visualize_fer2013_samples(dataset, num_samples=8, figsize=(12, 8)):
    """FER2013データセットのサンプルを可視化"""
    
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 
                    'Neutral', 'Sad', 'Surprise']
    
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.ravel()
    
    # ランダムなサンプルを選択
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        
        # テンソルを画像に変換
        if isinstance(image, torch.Tensor):
            # 正規化を逆変換 (ViT用の正規化に対応)
            image = image * 0.5 + 0.5  # [-1, 1] -> [0, 1]
            image = torch.clamp(image, 0, 1)
            if image.shape[0] == 3:  # RGB画像の場合
                image = image.permute(1, 2, 0).numpy()
            else:  # グレースケールの場合
                image = image.squeeze().numpy()
        
        axes[i].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        axes[i].set_title(f'{emotion_names[label]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def create_fer2013_inference_function(model_path, device=None):
    """FER2013用推論関数を作成"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデルの初期化
    model = models.vit_b_16()
    model.heads.head = nn.Linear(model.heads.head.in_features, 7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 前処理の定義
    transform = get_fer2013_transforms(input_size=224, augment=False)
    
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    def predict(image_path):
        """単一画像の感情を予測"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'emotion': emotion_names[predicted_class],
            'confidence': confidence,
            'probabilities': {emotion_names[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        }
    
    return predict

def plot_training_curves(train_losses, val_losses=None, val_accuracies=None, 
                        save_path=None, show=True):
    """学習曲線をプロット"""
    epochs = range(1, len(train_losses) + 1)
    
    if val_losses is not None and val_accuracies is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # 損失の比較
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 検証精度
        ax2.plot(epochs, val_accuracies, 'g-', label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # 総合進捗
        ax3.plot(epochs, train_losses, 'b-', label='Train Loss')
        normalized_acc = [acc/100 for acc in val_accuracies]
        ax3.plot(epochs, normalized_acc, 'g-', label='Val Accuracy (normalized)')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Normalized Value')
        ax3.set_title('Training Progress')
        ax3.legend()
        ax3.grid(True)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(epochs, train_losses, 'b-', label='Train Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'学習曲線を {save_path} として保存しました')
    
    if show:
        plt.show()

# モジュール情報
__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

if __name__ == "__main__":
    # ライブラリ単体でのテスト実行
    root_dir = "dataset/fer2013"
    
    print("FER2013前処理ライブラリのテスト実行")
    print(f"Version: {__version__}")
    
    if os.path.exists(root_dir):
        analyze_fer2013_dataset(root_dir)
        
        # サンプルデータローダーの作成
        train_loader, val_loader, test_loader = create_fer2013_data_loaders(
            root_dir=root_dir,
            batch_size=8,
            val_split=0.2
        )
        
        print(f"訓練データ: {len(train_loader)} バッチ")
        if val_loader:
            print(f"検証データ: {len(val_loader)} バッチ")
        if test_loader:
            print(f"テストデータ: {len(test_loader)} バッチ")
    else:
        print(f"データセットディレクトリが見つかりません: {root_dir}")