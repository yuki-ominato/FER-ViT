import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class RafDbDataset(Dataset):
    """RAF-DB データセットローダー"""
    def __init__(self, data_root, split='train', img_size=224, transform=None):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.img_folder = os.path.join(data_root, 'Image', 'aligned') # アライン済み画像推奨
        self.label_file = os.path.join(data_root, 'EmoLabel', 'list_patition_label.txt')
        
        # 基本的な変換
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                # VQGANは通常 [-1, 1] の入力を期待します
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

        self.samples = []
        self._load_annotations()

    def _load_annotations(self):
        with open(self.label_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # 例: train_00001.jpg 1
            fname, label = line.split(' ', 1)
            label = int(label) - 1 # RAF-DBは1-7なので0-6に変換
            
            # splitによるフィルタリング (RAF-DBのファイル名規則に基づく)
            if self.split == 'train' and fname.startswith('train'):
                self.samples.append((os.path.join(self.img_folder, fname), label))
            elif self.split == 'test' and fname.startswith('test'):
                self.samples.append((os.path.join(self.img_folder, fname), label))

        print(f"Loaded {len(self.samples)} images for {self.split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label