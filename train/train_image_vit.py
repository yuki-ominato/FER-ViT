"""
画像から直接ViTを学習するスクリプト
従来手法との比較用ベースライン
"""

import os
import sys
import argparse
import json
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.image_dataset import ImageFERDataset, get_train_transforms, get_val_transforms
from models_fer_vit.image_vit import ImageViT, create_vit_small, create_vit_base, create_vit_tiny
from utils.experiment_logger import ExperimentLogger, create_experiment_name


def set_seed(seed: int = 42) -> None:
    """再現性のためのシード設定"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def calculate_class_weights(dataset: ImageFERDataset) -> torch.Tensor:
    """クラス重みを計算（不均衡データ対応）"""
    labels = [label for _, label in dataset.samples]
    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    weights = []
    for i in range(num_classes):
        if i in class_counts:
            weight = total_samples / (num_classes * class_counts[i])
            weights.append(weight)
        else:
            weights.append(1.0)
    
    return torch.FloatTensor(weights)


def train_epoch(model, loader, optimizer, criterion, device, grad_clip=None):
    """1エポックの学習"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        
        # 勾配クリッピング（オプション）
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    
    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, device):
    """モデルの評価"""
    model.eval()
    all_preds = []
    all_labels = []
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        logits = model(images)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': all_preds,
        'labels': all_labels
    }


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # データセット準備
    print("\n" + "="*60)
    print("Loading datasets...")
    print("="*60)
    
    train_transform = get_train_transforms(args.img_size) if args.use_augmentation else get_val_transforms(args.img_size)
    val_transform = get_val_transforms(args.img_size)
    
    train_ds = ImageFERDataset(
        args.train_dir,
        transform=train_transform,
        img_size=args.img_size
    )
    
    val_ds = ImageFERDataset(
        args.val_dir,
        transform=val_transform,
        img_size=args.img_size
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # モデル作成
    print("\n" + "="*60)
    print("Creating model...")
    print("="*60)
    
    if args.model_size == 'tiny':
        model = create_vit_tiny(num_classes=args.num_classes, img_size=args.img_size)
    elif args.model_size == 'small':
        model = create_vit_small(num_classes=args.num_classes, img_size=args.img_size)
    elif args.model_size == 'base':
        model = create_vit_base(num_classes=args.num_classes, img_size=args.img_size)
    else:
        # カスタム設定
        model = ImageViT(
            img_size=args.img_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            heads=args.heads,
            mlp_dim=args.mlp_dim,
            num_classes=args.num_classes,
            dropout=args.dropout,
        )
    
    model = model.to(device)
    
    # パラメータ数を表示
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_params:,}")
    
    # 損失関数
    if args.use_class_weights:
        class_weights = calculate_class_weights(train_ds).to(device)
        print(f"Class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # オプティマイザー
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    # スケジューラー
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=5,
            factor=0.5,
            verbose=True
        )
    elif args.scheduler == 'warmup_cosine':
        # ウォームアップ + Cosine Annealing
        warmup_epochs = min(10, args.epochs // 10)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None
    
    # 実験設定
    model_config = {
        'model_size': args.model_size,
        'img_size': args.img_size,
        'patch_size': args.patch_size,
        'embed_dim': args.embed_dim,
        'depth': args.depth,
        'heads': args.heads,
        'mlp_dim': args.mlp_dim,
        'num_classes': args.num_classes,
        'dropout': args.dropout,
        'n_parameters': n_params,
    }
    
    training_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'use_augmentation': args.use_augmentation,
        'use_class_weights': args.use_class_weights,
        'label_smoothing': args.label_smoothing,
        'grad_clip': args.grad_clip,
        'seed': args.seed,
    }
    
    data_config = {
        'train_dir': args.train_dir,
        'val_dir': args.val_dir,
        'train_samples': len(train_ds),
        'val_samples': len(val_ds),
    }
    
    config = {
        'model': model_config,
        'training': training_config,
        'data': data_config,
    }
    
    # 実験ロガー初期化
    # experiment_name = create_experiment_name(model_config, training_config, prefix="image_vit")
    experiment_name = create_experiment_name(model_config, training_config)
    logger = ExperimentLogger(experiment_name, base_dir="experiments")
    logger.log_config(config)
    
    # 学習ループ
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_f1 = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # 学習
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip=args.grad_clip
        )
        
        # 評価
        val_results = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch}/{args.epochs}: "
              f"train_loss={train_loss:.4f} "
              f"val_acc={val_results['accuracy']:.4f} "
              f"val_f1={val_results['f1_macro']:.4f}")
        
        # ロギング
        logger.log_learning_curves(train_loss, val_results, epoch)
        logger.log_learning_rate(optimizer, epoch)
        
        if epoch % 10 == 0:
            logger.log_parameters(model, epoch)
            logger.log_gradients(model, epoch)
        
        # チェックポイント保存
        is_best = val_results['f1_macro'] > best_f1
        if is_best:
            best_f1 = val_results['f1_macro']
            print(f"  → New best model (F1: {best_f1:.4f})")
        
        logger.save_checkpoint(model, optimizer, epoch, val_results, is_best)
        
        # スケジューラー更新
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_results['f1_macro'])
            else:
                scheduler.step()
    
    # 最終評価
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Best F1 macro: {best_f1:.4f}")
    
    final_results = evaluate(model, val_loader, device)
    print(f"\nFinal validation results:")
    print(f"  Accuracy: {final_results['accuracy']:.4f}")
    print(f"  F1 Macro: {final_results['f1_macro']:.4f}")
    print(f"  F1 Weighted: {final_results['f1_weighted']:.4f}")
    
    # 分類レポート
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    print(f"\nClassification Report:")
    print(classification_report(
        final_results['labels'],
        final_results['predictions'],
        target_names=emotion_names
    ))
    
    # 混同行列
    logger.log_confusion_matrix(
        final_results['labels'],
        final_results['predictions'],
        emotion_names,
        args.epochs
    )
    
    # サマリー保存
    final_metrics = {
        'accuracy': final_results['accuracy'],
        'f1_macro': final_results['f1_macro'],
        'f1_weighted': final_results['f1_weighted'],
        'best_f1_macro': best_f1,
    }
    logger.log_experiment_summary(final_metrics)
    logger.close()
    
    print(f"\nExperiment saved to: {logger.get_experiment_path()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vision Transformer on image data")
    
    # データ関連
    parser.add_argument("--train_dir", required=True, help="Training data directory")
    parser.add_argument("--val_dir", required=True, help="Validation data directory")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--use_augmentation", action='store_true', help="Use data augmentation")
    
    # モデル関連
    parser.add_argument("--model_size", choices=['tiny', 'small', 'base', 'custom'],
                       default='small', help="Predefined model size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--embed_dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=12, help="Transformer depth")
    parser.add_argument("--heads", type=int, default=6, help="Number of heads")
    parser.add_argument("--mlp_dim", type=int, default=1536, help="MLP dimension")
    parser.add_argument("--num_classes", type=int, default=7, help="Number of classes")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # 学習設定
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--optimizer", choices=['adamw', 'sgd'], default='adamw')
    parser.add_argument("--scheduler", choices=['none', 'cosine', 'plateau', 'warmup_cosine'],
                       default='warmup_cosine', help="Learning rate scheduler")
    parser.add_argument("--grad_clip", type=float, default=None, help="Gradient clipping")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing")
    
    # その他
    parser.add_argument("--use_class_weights", action='store_true', help="Use class weights")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    main(args)