import os
import sys
import argparse
import json
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.latent_dataset import LatentFERDataset
from models_fer_vit.latent_vit import LatentViT
from utils.experiment_logger import ExperimentLogger, create_experiment_name


def set_seed(seed: int = 42) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 再現性のための設定（一部の演算でエラーが出る場合がある）
    # CUDA環境では環境変数CUBLAS_WORKSPACE_CONFIGが必要
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def calculate_class_weights(dataset: LatentFERDataset) -> torch.Tensor:
    """クラス重みを計算（不均衡データ対応）"""
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)
    
    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    # 逆頻度重みを計算
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
        latents = latents.to(device)  # (B, L, D)
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
    """評価関数（精度、F1スコア、詳細メトリクス）"""
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
    
    # メトリクス計算
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

    # データセット読み込み
    train_ds = LatentFERDataset(args.latent_train_dir)
    val_ds = LatentFERDataset(args.latent_val_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # seq_len を自動推定（latentファイルから形状を取得）
    if args.seq_len <= 0:
        # 最初のサンプルを読み、(L, D) から L を推定
        sample_latent, _ = train_ds[0]
        inferred_seq_len = int(sample_latent.shape[0])
        print(f"Inferred seq_len from latents: {inferred_seq_len}")
        args.seq_len = inferred_seq_len

    # モデル初期化
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

    # 実験設定
    model_config = {
        'latent_dim': args.latent_dim,
        'seq_len': args.seq_len,
        'embed_dim': args.embed_dim,
        'depth': args.depth,
        'heads': args.heads,
        'mlp_dim': args.mlp_dim,
        'num_classes': args.num_classes,
        'dropout': args.dropout,
    }
    
    training_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'use_class_weights': args.use_class_weights,
        'scheduler': args.scheduler,
        'seed': args.seed,
    }
    
    data_config = {
        'train_dir': args.latent_train_dir,
        'val_dir': args.latent_val_dir,
    }
    
    config = {
        'model': model_config,
        'training': training_config,
        'data': data_config,
    }
    
    # 実験ロガー初期化
    experiment_name = create_experiment_name(model_config, training_config)
    logger = ExperimentLogger(experiment_name, base_dir="experiments")
    logger.log_config(config)
    
    # チェックポイントディレクトリを実験ロガーのディレクトリに設定
    args.checkpoint_dir = logger.get_experiment_path()

    # 学習ループ
    best_f1 = 0.0
    train_losses = []
    val_metrics = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_results = evaluate(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_metrics.append(val_results)
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} "
              f"val_acc={val_results['accuracy']:.4f} "
              f"val_f1_macro={val_results['f1_macro']:.4f}")

        # TensorBoardロギング
        logger.log_learning_curves(train_loss, val_results, epoch)
        logger.log_learning_rate(optimizer, epoch)
        
        # パラメータと勾配のログ（10エポックごと）
        if epoch % 10 == 0:
            logger.log_parameters(model, epoch)
            logger.log_gradients(model, epoch)

        # チェックポイント保存
        is_best = val_results['f1_macro'] > best_f1
        if is_best:
            best_f1 = val_results['f1_macro']
            print(f"New best model saved (F1: {best_f1:.4f})")
        
        logger.save_checkpoint(model, optimizer, epoch, val_results, is_best)

        # スケジューラー更新
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_results['f1_macro'])
            else:
                scheduler.step()

    # 最終結果
    print(f"\nTraining completed!")
    print(f"Best F1 macro: {best_f1:.4f}")
    
    # 最終評価（詳細レポート）
    final_results = evaluate(model, val_loader, device)
    print(f"\nFinal validation results:")
    print(f"Accuracy: {final_results['accuracy']:.4f}")
    print(f"F1 Macro: {final_results['f1_macro']:.4f}")
    print(f"F1 Weighted: {final_results['f1_weighted']:.4f}")
    
    # 分類レポート
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    print(f"\nClassification Report:")
    print(classification_report(final_results['labels'], final_results['predictions'], 
                              target_names=emotion_names))
    
    # 混同行列をログ
    logger.log_confusion_matrix(final_results['labels'], final_results['predictions'], 
                               emotion_names, args.epochs)
    
    # 実験サマリーをログ
    final_metrics = {
        'accuracy': final_results['accuracy'],
        'f1_macro': final_results['f1_macro'],
        'f1_weighted': final_results['f1_weighted'],
        'best_f1_macro': best_f1,
    }
    logger.log_experiment_summary(final_metrics)
    
    # ロガーを閉じる
    logger.close()
    
    print(f"\nExperiment completed. Results saved to: {logger.get_experiment_path()}")
    print(f"TensorBoard logs available at: {os.path.join(logger.get_experiment_path(), 'logs')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LatentViT for FER")
    
    # データ関連
    parser.add_argument("--latent_train_dir", required=True, help="Path to training latent files")
    parser.add_argument("--latent_val_dir", required=True, help="Path to validation latent files")
    
    # 学習設定
    parser.add_argument("--checkpoint_dir", default="checkpoints/latent_vit", help="Checkpoint save directory")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--scheduler", choices=['none', 'cosine', 'plateau'], default='plateau', 
                       help="Learning rate scheduler")
    parser.add_argument("--use_class_weights", action='store_true', 
                       help="Use class weights for imbalanced data")
    
    # モデル設定
    parser.add_argument("--latent_dim", type=int, default=512, help="Latent dimension")
    parser.add_argument("--seq_len", type=int, default=0, help="Sequence length (w+ layers). If 0 or less, infer from latents.")
    parser.add_argument("--embed_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=6, help="Transformer depth")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--mlp_dim", type=int, default=2048, help="MLP dimension")
    parser.add_argument("--num_classes", type=int, default=7, help="Number of emotion classes")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # 再現性
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    main(args)


