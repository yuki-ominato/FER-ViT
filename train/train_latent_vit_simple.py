import os
import argparse
import json
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np

from data.latent_dataset import LatentFERDataset
from models_fer_vit.latent_vit import LatentViT


def set_seed(seed: int = 42) -> None:
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_class_weights(dataset: LatentFERDataset) -> torch.Tensor:
    """クラス重みを計算"""
    class_counts = dataset.get_class_counts()
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    weights = torch.zeros(num_classes)
    for class_id, count in class_counts.items():
        weights[class_id] = total_samples / (num_classes * count)
    
    return weights


@torch.no_grad()
def evaluate(model, loader, device):
    """モデルを評価"""
    model.eval()
    all_preds = []
    all_labels = []
    
    for latents, labels in loader:
        latents = latents.to(device)
        labels = labels.to(device)
        
        outputs = model(latents)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels
    }


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # データセット読み込み
    print("Loading datasets...")
    train_ds = LatentFERDataset(args.latent_train_dir)
    val_ds = LatentFERDataset(args.latent_val_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # seq_len を自動推定（latentファイルから形状を取得）
    if args.seq_len <= 0:
        sample_latent, _ = train_ds[0]
        inferred_seq_len = int(sample_latent.shape[0])
        print(f"Inferred seq_len from latents: {inferred_seq_len}")
        args.seq_len = inferred_seq_len

    # モデル初期化
    print("Initializing model...")
    model = LatentViT(
        latent_dim=args.latent_dim,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        num_classes=args.num_classes,
        dropout=args.dropout,
    ).to(device)

    # クラス重み計算（不均衡データ対応）
    if args.use_class_weights:
        class_weights = calculate_class_weights(train_ds).to(device)
        print(f"Class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # オプティマイザーとスケジューラー
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    else:
        scheduler = None

    print(f"Starting training for {args.epochs} epochs...")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 学習ループ
    best_f1 = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # 学習
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (latents, labels) in enumerate(train_loader):
            latents = latents.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(latents)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # 検証
        val_metrics = evaluate(model, val_loader, device)
        val_acc = val_metrics['accuracy'] * 100
        val_f1 = val_metrics['f1']
        
        print(f'Epoch {epoch}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}')
        
        # スケジューラー更新
        if scheduler:
            if args.scheduler == 'plateau':
                scheduler.step(val_f1)
            else:
                scheduler.step()
        
        # ベストモデル保存
        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'args': args
            }, 'checkpoints/best_model.pt')
            print(f'  New best model saved! F1: {val_f1:.4f}')
        
        print()

    print(f"Training completed! Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LatentViT for FER")
    
    # データ関連
    parser.add_argument("--latent_train_dir", type=str, required=True, help="Training latent directory")
    parser.add_argument("--latent_val_dir", type=str, required=True, help="Validation latent directory")
    
    # モデル関連
    parser.add_argument("--latent_dim", type=int, default=512, help="Latent dimension")
    parser.add_argument("--seq_len", type=int, default=18, help="Sequence length (0 for auto-infer)")
    parser.add_argument("--embed_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=6, help="Transformer depth")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--mlp_dim", type=int, default=2048, help="MLP dimension")
    parser.add_argument("--num_classes", type=int, default=7, help="Number of classes")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # 学習関連
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--scheduler", type=str, default=None, choices=[None, 'cosine', 'plateau'], help="Learning rate scheduler")
    parser.add_argument("--use_class_weights", action="store_true", help="Use class weights for imbalanced data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    main(args)
