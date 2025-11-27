"""
潜在空間CNN学習スクリプト
ViTと同じ入力（StyleGAN潜在コード）でCNNを学習

train_latent_vit.pyのViT部分をCNNに置き換えたバージョン
"""

import os
import sys
import argparse
import json
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.latent_dataset import LatentFERDataset
from models_fer_vit.latent_cnn import create_latent_cnn
from utils.experiment_logger import ExperimentLogger


def set_seed(seed: int = 42) -> None:
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


def create_subset_dataset(dataset: LatentFERDataset, fraction: float, seed: int = 42):
    """データセットをクラスバランスを保ちながら削減"""
    if fraction >= 1.0:
        return dataset
    
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)
    
    class_indices = {}
    for idx, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    selected_indices = []
    print(f"\nデータセット削減: {fraction*100:.1f}% を使用")
    print("="*60)
    
    for class_id, indices in sorted(class_indices.items()):
        n_samples = len(indices)
        n_select = max(1, int(n_samples * fraction))
        
        np.random.seed(seed)
        selected = np.random.choice(indices, n_select, replace=False)
        selected_indices.extend(selected)
        
        emotion_name = dataset.get_class_names()[class_id]
        print(f"  {emotion_name:>8s}: {n_samples:>5d} → {n_select:>5d} ({n_select/n_samples*100:.1f}%)")
    
    print(f"  {'Total':>8s}: {len(labels):>5d} → {len(selected_indices):>5d}")
    print("="*60)
    
    return Subset(dataset, selected_indices)


def calculate_class_weights(dataset) -> torch.Tensor:
    """クラス重みを計算"""
    labels = []
    
    if isinstance(dataset, Subset):
        for idx in dataset.indices:
            _, label = dataset.dataset[idx]
            labels.append(label)
    else:
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
    
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


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for latents, labels in loader:
        latents = latents.to(device)
        labels = labels.to(device)
        # --- Mixup Implementation ---
        # 50%の確率、または常時適用するかは実験次第ですが、今回は常時適用例
        alpha = 1.0
        if alpha > 0:
            # Beta分布から混合比率lamをサンプリング
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        # バッチ内のデータをシャッフル
        index = torch.randperm(latents.size(0)).to(device)

        # 潜在ベクトルを混ぜる (StyleGANの空間では、これが中間の顔になる)
        mixed_latents = lam * latents + (1 - lam) * latents[index]

        # ラベルも混ぜる必要があるため、Loss計算時に処理する
        # ----------------------------

        optimizer.zero_grad()
        
        # 混ぜたベクトルを入力
        logits = model(mixed_latents)

        # Lossの計算（元のラベルと、混ぜた相手のラベルの加重平均）
        loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[index])
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * latents.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    for latents, labels in loader:
        latents = latents.to(device)
        labels = labels.to(device)
        logits = model(latents)
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

    # データセット読み込み
    print("\n" + "="*60)
    print("データセット読み込み中...")
    print("="*60)
    
    train_ds_full = LatentFERDataset(args.latent_train_dir)
    val_ds = LatentFERDataset(args.latent_val_dir)
    
    # データセット削減
    if args.data_fraction < 1.0:
        train_ds = create_subset_dataset(train_ds_full, args.data_fraction, args.seed)
        print(f"\n削減後の訓練データ: {len(train_ds)} サンプル")
    else:
        train_ds = train_ds_full
        print(f"\n全訓練データを使用: {len(train_ds)} サンプル")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)

    # seq_lenを自動推定
    if args.seq_len <= 0:
        if isinstance(train_ds, Subset):
            sample_latent, _ = train_ds.dataset[train_ds.indices[0]]
        else:
            sample_latent, _ = train_ds[0]
        inferred_seq_len = int(sample_latent.shape[0])
        print(f"\n潜在コードから推定された seq_len: {inferred_seq_len}")
        args.seq_len = inferred_seq_len

    # モデル初期化
    print("\n" + "="*60)
    print("モデル初期化中...")
    print("="*60)
    
    model = create_latent_cnn(
        model_type=args.model_type,
        latent_dim=args.latent_dim,
        seq_len=args.seq_len,
        num_classes=args.num_classes,
        dropout=args.dropout,
    ).to(device)
    
    # パラメータ数表示
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params:,}")
    print(f"Trainable parameters: {n_trainable:,}")
    print(f"Model size: {n_params * 4 / 1024 / 1024:.2f} MB")

    # クラス重み計算
    if args.use_class_weights:
        class_weights = calculate_class_weights(train_ds).to(device)
        print(f"\nクラス重み: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                         patience=5, factor=0.5)
    else:
        scheduler = None

    # 実験設定
    model_config = {
        'model_type': args.model_type,
        'latent_dim': args.latent_dim,
        'seq_len': args.seq_len,
        'num_classes': args.num_classes,
        'dropout': args.dropout,
        'n_parameters': n_params,
    }
    
    training_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'use_class_weights': args.use_class_weights,
        'scheduler': args.scheduler,
        'seed': args.seed,
        'data_fraction': args.data_fraction,
    }
    
    config = {
        'model': model_config,
        'training': training_config,
        'data': {
            'train_dir': args.latent_train_dir,
            'val_dir': args.latent_val_dir,
            'train_samples_total': len(train_ds_full),
            'train_samples_used': len(train_ds),
            'val_samples': len(val_ds),
        },
    }
    
    # 実験ロガー初期化
    base_name = f"latent_cnn_{args.model_type}_ep{args.epochs}_bs{args.batch_size}_lr{args.lr}"
    if args.data_fraction < 1.0:
        experiment_name = f"{base_name}_frac{int(args.data_fraction*100)}"
    else:
        experiment_name = base_name
    
    logger = ExperimentLogger(experiment_name, base_dir="experiments")
    logger.log_config(config)

    # 学習ループ
    print("\n" + "="*60)
    print("学習開始...")
    print("="*60)
    
    best_f1 = 0.0
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_results = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch}/{args.epochs}: "
              f"loss={train_loss:.4f} "
              f"val_acc={val_results['accuracy']:.4f} "
              f"val_f1={val_results['f1_macro']:.4f}")

        logger.log_learning_curves(train_loss, val_results, epoch)
        logger.log_learning_rate(optimizer, epoch)
        
        if epoch % 10 == 0:
            logger.log_parameters(model, epoch)
            logger.log_gradients(model, epoch)

        is_best = val_results['f1_macro'] > best_f1
        if is_best:
            best_f1 = val_results['f1_macro']
            print(f"  → Best model (F1: {best_f1:.4f})")
            logger.save_checkpoint(model, optimizer, epoch, val_results, is_best)

        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_results['f1_macro'])
            else:
                scheduler.step()

    # 最終結果
    print(f"\n{'='*60}")
    print("学習完了!")
    print(f"{'='*60}")
    print(f"使用データ割合: {args.data_fraction*100:.1f}%")
    print(f"Best F1 macro: {best_f1:.4f}")
    
    final_results = evaluate(model, val_loader, device)
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    print(f"\nClassification Report:")
    print(classification_report(final_results['labels'], final_results['predictions'], 
                              target_names=emotion_names))
    
    logger.log_confusion_matrix(final_results['labels'], final_results['predictions'], 
                               emotion_names, args.epochs)
    
    final_metrics = {
        'accuracy': final_results['accuracy'],
        'f1_macro': final_results['f1_macro'],
        'f1_weighted': final_results['f1_weighted'],
        'best_f1_macro': best_f1,
        'data_fraction': args.data_fraction,
    }
    logger.log_experiment_summary(final_metrics)
    logger.close()
    
    print(f"\n実験結果: {logger.get_experiment_path()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Latent CNN for FER (vs ViT comparison)")
    
    # データ関連
    parser.add_argument("--latent_train_dir", required=True)
    parser.add_argument("--latent_val_dir", required=True)
    parser.add_argument("--data_fraction", type=float, default=1.0,
                       help="使用する訓練データの割合 (0.0 < fraction <= 1.0)")
    
    # 学習設定
    parser.add_argument("--epochs", type=int, default=60, 
                       help="学習エポック数（ViTと同じ60を推奨）")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="学習率（ViTと比較するため同じ値を推奨）")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--scheduler", choices=['none', 'cosine', 'plateau'], default='plateau')
    parser.add_argument("--use_class_weights", action='store_true')
    
    # モデル設定
    parser.add_argument("--model_type", choices=['light', 'standard', 'deep', '2d'],
                       default='standard', help="CNN architecture type")
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=0,
                       help="Sequence length (0 for auto-detection)")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="ドロップアウト率（CNNは0.3が良い）")
    
    # その他
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    if args.data_fraction <= 0.0 or args.data_fraction > 1.0:
        raise ValueError(f"data_fraction must be in (0.0, 1.0], got {args.data_fraction}")
    
    main(args)