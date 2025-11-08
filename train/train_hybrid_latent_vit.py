"""
事前学習Transformer + StyleGAN潜在コードの学習スクリプト
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

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.latent_dataset import LatentFERDataset
from models_fer_vit.hybrid_latent_vit import create_hybrid_latent_vit, HybridLatentViT
from utils.experiment_logger import ExperimentLogger, create_experiment_name


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


def calculate_class_weights(dataset: LatentFERDataset) -> torch.Tensor:
    labels = []
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


def get_optimizer_groups(model: HybridLatentViT, lr: float, weight_decay: float):
    """
    レイヤーごとに異なる学習率を設定
    - Input projection: lr * 10
    - Transformer: lr
    - Head: lr * 10
    """
    param_groups = []
    
    # Input projection（高学習率）
    input_proj_params = list(model.input_proj.parameters())
    if len(input_proj_params) > 0:
        param_groups.append({
            'params': input_proj_params,
            'lr': lr * 10,
            'weight_decay': weight_decay,
        })
        print(f"Input projection: lr={lr*10:.2e}")
    
    # Transformer（基準学習率）
    transformer_params = [p for p in model.transformer.parameters() if p.requires_grad]
    if len(transformer_params) > 0:
        param_groups.append({
            'params': transformer_params,
            'lr': lr,
            'weight_decay': weight_decay,
        })
        print(f"Transformer: lr={lr:.2e}, params={len(transformer_params)}")
    
    # Adapter（ある場合）
    if model.use_adapter:
        adapter_params = [p for p in model.adapters.parameters() if p.requires_grad]
        if len(adapter_params) > 0:
            param_groups.append({
                'params': adapter_params,
                'lr': lr * 10,
                'weight_decay': weight_decay,
            })
            print(f"Adapters: lr={lr*10:.2e}")
    
    # Head（高学習率）
    head_params = list(model.head.parameters())
    if len(head_params) > 0:
        param_groups.append({
            'params': head_params,
            'lr': lr * 10,
            'weight_decay': weight_decay,
        })
        print(f"Head: lr={lr*10:.2e}")
    
    # Position embedding & CLS token
    other_params = [model.pos_embed, model.cls_token]
    param_groups.append({
        'params': other_params,
        'lr': lr * 5,
        'weight_decay': 0,  # 位置埋め込みには weight decay を適用しない
    })
    print(f"Position/CLS: lr={lr*5:.2e}")
    
    return param_groups


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    for latents, labels in loader:
        latents = latents.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(latents)
        loss = criterion(logits, labels)
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
    
    # データセット
    print("\n" + "="*60)
    print("Loading datasets...")
    print("="*60)
    
    train_ds = LatentFERDataset(args.latent_train_dir)
    val_ds = LatentFERDataset(args.latent_val_dir)
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    print(f"Train: {len(train_ds)} samples")
    print(f"Val: {len(val_ds)} samples")
    
    # seq_lenを自動推定
    sample_latent, _ = train_ds[0]
    seq_len = sample_latent.shape[0]
    latent_dim = sample_latent.shape[1]
    print(f"Detected latent shape: ({seq_len}, {latent_dim})")
    
    # モデル作成
    print("\n" + "="*60)
    print("Creating hybrid model...")
    print("="*60)
    
    model = create_hybrid_latent_vit(
        latent_dim=latent_dim,
        seq_len=seq_len,
        model_size=args.model_size,
        num_classes=args.num_classes,
        use_pretrained=args.use_pretrained,
        freeze_transformer=args.freeze_transformer,
        freeze_stages=args.freeze_stages if args.freeze_stages > 0 else None,
        use_adapter=args.use_adapter,
        adapter_dim=args.adapter_dim,
    )
    model = model.to(device)
    
    # 損失関数
    if args.use_class_weights:
        class_weights = calculate_class_weights(train_ds).to(device)
        print(f"Class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # オプティマイザー
    if args.use_layerwise_lr:
        param_groups = get_optimizer_groups(model, args.lr, args.weight_decay)
        optimizer = optim.AdamW(param_groups)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # スケジューラー
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5, verbose=True
        )
    else:
        scheduler = None
    
    # 実験設定
    model_config = {
        'latent_dim': latent_dim,
        'seq_len': seq_len,
        'model_size': args.model_size,
        'use_pretrained': args.use_pretrained,
        'freeze_transformer': args.freeze_transformer,
        'freeze_stages': args.freeze_stages if args.freeze_stages > 0 else None,
        'use_adapter': args.use_adapter,
        'adapter_dim': args.adapter_dim if args.use_adapter else None,
    }
    
    training_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'scheduler': args.scheduler,
        'use_class_weights': args.use_class_weights,
        'use_layerwise_lr': args.use_layerwise_lr,
        'seed': args.seed,
    }
    
    config = {
        'model': model_config,
        'training': training_config,
        'data': {
            'train_dir': args.latent_train_dir,
            'val_dir': args.latent_val_dir,
        },
    }
    
    # ロガー
    experiment_name = f"hybrid_vit_{create_experiment_name(model_config, training_config)}"
    logger = ExperimentLogger(experiment_name, base_dir="experiments")
    logger.log_config(config)
    
    # 学習ループ
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_f1 = 0.0
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_results = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch}/{args.epochs}: "
              f"loss={train_loss:.4f} "
              f"acc={val_results['accuracy']:.4f} "
              f"f1={val_results['f1_macro']:.4f}")
        
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
    
    # 最終評価
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Best F1: {best_f1:.4f}")
    
    final_results = evaluate(model, val_loader, device)
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    print("\nClassification Report:")
    print(classification_report(
        final_results['labels'],
        final_results['predictions'],
        target_names=emotion_names
    ))
    
    logger.log_confusion_matrix(
        final_results['labels'],
        final_results['predictions'],
        emotion_names,
        args.epochs
    )
    
    final_metrics = {
        'accuracy': final_results['accuracy'],
        'f1_macro': final_results['f1_macro'],
        'f1_weighted': final_results['f1_weighted'],
        'best_f1_macro': best_f1,
    }
    logger.log_experiment_summary(final_metrics)
    logger.close()
    
    print(f"\nResults saved to: {logger.get_experiment_path()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Hybrid ViT: Pretrained Transformer + StyleGAN Latents"
    )
    
    # データ
    parser.add_argument("--latent_train_dir", required=True)
    parser.add_argument("--latent_val_dir", required=True)
    
    # モデル
    parser.add_argument("--model_size", choices=['tiny', 'small', 'base'],
                       default='small', help="Pretrained ViT size")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--use_pretrained", action='store_true', default=True,
                       help="Use pretrained transformer weights")
    
    # ファインチューニング戦略
    parser.add_argument("--freeze_transformer", action='store_true',
                       help="Freeze entire transformer (linear probe)")
    parser.add_argument("--freeze_stages", type=int, default=0,
                       help="Freeze first N transformer blocks (0=no freeze)")
    parser.add_argument("--use_adapter", action='store_true',
                       help="Use adapter layers (parameter-efficient)")
    parser.add_argument("--adapter_dim", type=int, default=64,
                       help="Adapter bottleneck dimension")
    
    # 学習設定
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--scheduler", choices=['none', 'cosine', 'plateau'],
                       default='plateau')
    parser.add_argument("--use_class_weights", action='store_true')
    parser.add_argument("--use_layerwise_lr", action='store_true',
                       help="Use different learning rates for different layers")
    
    # その他
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)