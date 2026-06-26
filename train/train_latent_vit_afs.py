"""
AFS Style Extractor で前処理した潜在コードによる LatentViT 学習スクリプト

各バッチで w_sty = h(w) を GPU 上でオンザフライ計算してから LatentViT に渡す。
StyleExtractor h はフリーズして推論専用として使用する。

Usage:
    python train/train_latent_vit_afs.py \\
        --latent_train_dir latents/rafdb/train \\
        --latent_val_dir   latents/rafdb/test \\
        --style_extractor_path experiments/afs_rafdb/<run_id>/checkpoints/best_model.pt \\
        --epochs 60 --batch_size 64
"""

import os
import sys
import argparse
import json
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.latent_dataset import (
    LatentFERDataset,
    get_latent_train_transforms,
    get_latent_val_transforms,
)
from models_fer_vit.latent_vit import LatentViT
from afs.style_extractor import StyleExtractor
from utils.experiment_logger import ExperimentLogger, create_experiment_name


# ------------------------------------------------------------------------------
# Seed / reproducibility
# ------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------
# Style Extractor loader (frozen)
# ------------------------------------------------------------------------------

def load_frozen_style_extractor(ckpt_path: str, device: torch.device) -> StyleExtractor:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    h = StyleExtractor()
    h.load_state_dict(ckpt['model_state'])
    h.eval()
    for p in h.parameters():
        p.requires_grad_(False)
    return h.to(device)


# ------------------------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------------------------

def create_subset_dataset(dataset: LatentFERDataset, fraction: float, seed: int = 42):
    if fraction >= 1.0:
        return dataset

    labels = [dataset[i][1] for i in range(len(dataset))]
    class_indices: dict = {}
    for idx, label in enumerate(labels):
        class_indices.setdefault(label, []).append(idx)

    selected = []
    print(f"\nデータセット削減: {fraction*100:.1f}% を使用")
    print("=" * 60)
    for cls_id, indices in sorted(class_indices.items()):
        n = max(1, int(len(indices) * fraction))
        rng = np.random.default_rng(seed)
        chosen = rng.choice(indices, n, replace=False)
        selected.extend(chosen)
        name = dataset.get_class_names()[cls_id]
        print(f"  {name:>8s}: {len(indices):>5d} → {n:>5d}")
    print(f"  {'Total':>8s}: {len(labels):>5d} → {len(selected):>5d}")
    print("=" * 60)
    return Subset(dataset, selected)


def calculate_class_weights(dataset) -> torch.Tensor:
    if isinstance(dataset, Subset):
        labels = [dataset.dataset[i][1] for i in dataset.indices]
    else:
        labels = [dataset[i][1] for i in range(len(dataset))]
    counts = Counter(labels)
    n_total = len(labels)
    n_cls = len(counts)
    weights = [n_total / (n_cls * counts.get(i, 1)) for i in range(n_cls)]
    return torch.FloatTensor(weights)


# ------------------------------------------------------------------------------
# Train / Eval
# ------------------------------------------------------------------------------

def train_epoch(model, h, loader, optimizer, criterion, device, mixup_alpha: float = 1.0):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for latents, labels in loader:
        latents = latents.to(device)
        labels  = labels.to(device)

        # AFS: w → w_sty (on GPU, frozen h)
        with torch.no_grad():
            latents = h(latents)

        # Mixup
        lam = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1.0
        idx = torch.randperm(latents.size(0), device=device)
        mixed = lam * latents + (1 - lam) * latents[idx]

        optimizer.zero_grad()
        logits = model(mixed)
        loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[idx])
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * latents.size(0)

        with torch.no_grad():
            preds = model(latents).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, h, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for latents, labels in loader:
        latents = latents.to(device)
        labels  = labels.to(device)

        # AFS: w → w_sty
        latents = h(latents)

        logits = model(latents)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(dim=1)

        total_loss += loss.item() * latents.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss    = total_loss / len(loader.dataset)
    acc         = accuracy_score(all_labels, all_preds)
    f1_macro    = f1_score(all_labels, all_preds, average='macro',    zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return {
        'loss': avg_loss, 'accuracy': acc,
        'f1_macro': f1_macro, 'f1_weighted': f1_weighted,
        'predictions': all_preds, 'labels': all_labels,
    }


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Style Extractor (frozen) ---
    print(f"Loading StyleExtractor from {args.style_extractor_path} ...")
    h = load_frozen_style_extractor(args.style_extractor_path, device)
    print("StyleExtractor loaded and frozen.")

    # --- Dataset ---
    print("\n" + "=" * 60)
    print("データセット読み込み中...")
    train_transform = (
        get_latent_train_transforms(
            noise_std=args.latent_noise,
            scale_range=(0.9, 1.1),
            mask_prob=args.latent_mask,
        )
        if args.use_augmentation else None
    )

    train_ds_full = LatentFERDataset(args.latent_train_dir, transform=train_transform)
    val_ds        = LatentFERDataset(args.latent_val_dir,   transform=get_latent_val_transforms())

    train_ds = create_subset_dataset(train_ds_full, args.data_fraction, args.seed)
    print(f"訓練: {len(train_ds)} サンプル  /  検証: {len(val_ds)} サンプル")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # seq_len を自動推定
    if args.seq_len <= 0:
        base_ds = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
        sample, _ = base_ds[0]
        args.seq_len = int(sample.shape[0])
        print(f"seq_len を自動推定: {args.seq_len}")

    # --- Model ---
    print("\n" + "=" * 60)
    print("モデル初期化中...")
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
    print(f"LatentViT params: {sum(p.numel() for p in model.parameters()):,}")

    # --- Loss / Optimizer ---
    if args.use_class_weights:
        cw = calculate_class_weights(train_ds).to(device)
        print(f"クラス重み: {cw.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                          patience=5, factor=0.5)
    else:
        scheduler = None

    # --- Logger ---
    model_config = dict(
        latent_dim=args.latent_dim, seq_len=args.seq_len,
        embed_dim=args.embed_dim, depth=args.depth, heads=args.heads,
        mlp_dim=args.mlp_dim, num_classes=args.num_classes, dropout=args.dropout,
    )
    training_config = dict(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, scheduler=args.scheduler,
        mixup=args.mixup, data_fraction=args.data_fraction, seed=args.seed,
        style_extractor_path=args.style_extractor_path,
    )
    config = dict(
        model=model_config, training=training_config,
        data=dict(
            train_dir=args.latent_train_dir, val_dir=args.latent_val_dir,
            train_samples=len(train_ds), val_samples=len(val_ds),
        ),
    )
    base_name = create_experiment_name(model_config, training_config, is_latent=True)
    logger = ExperimentLogger(f"{base_name}_afs_frac{int(args.data_fraction*100)}",
                              base_dir="experiments")
    logger.log_config(config)

    # --- Training loop ---
    print("\n" + "=" * 60)
    print("学習開始...")
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_epoch(
            model, h, train_loader, optimizer, criterion, device, args.mixup
        )
        val_results = evaluate(model, h, val_loader, criterion, device)
        val_loss    = val_results['loss']
        val_acc     = val_results['accuracy']
        val_f1      = val_results['f1_macro']

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  train_f1={train_f1:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

        metrics = dict(
            train_loss=train_loss, train_acc=train_acc, train_f1=train_f1,
            val_loss=val_loss,     val_acc=val_acc,     val_f1=val_f1,
        )
        logger.log_metrics(metrics, epoch)
        logger.log_learning_rate(optimizer, epoch)

        if epoch % 10 == 0:
            logger.log_parameters(model, epoch)
            logger.log_gradients(model, epoch)

        if val_f1 > best_f1:
            best_f1 = val_f1
            print(f"  → best model saved (val_f1={best_f1:.4f})")
            logger.save_checkpoint(model, optimizer, epoch, val_results, is_best=True)

        if scheduler is not None:
            scheduler.step(val_f1) if args.scheduler == 'plateau' else scheduler.step()

    # --- Final evaluation ---
    print("\n" + "=" * 60)
    print(f"学習完了  /  Best val F1 macro: {best_f1:.4f}")
    final = evaluate(model, h, val_loader, criterion, device)
    names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    print("\nClassification Report:")
    print(classification_report(final['labels'], final['predictions'], target_names=names))
    logger.log_confusion_matrix(final['labels'], final['predictions'], names, args.epochs)
    logger.log_experiment_summary(dict(
        accuracy=final['accuracy'], f1_macro=final['f1_macro'],
        f1_weighted=final['f1_weighted'], best_f1_macro=best_f1,
        data_fraction=args.data_fraction,
    ))
    logger.close()
    print(f"実験結果: {logger.get_experiment_path()}")


# ------------------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LatentViT on AFS style-only latents (w_sty = h(w))"
    )

    # データ
    parser.add_argument("--latent_train_dir",      required=True)
    parser.add_argument("--latent_val_dir",        required=True)
    parser.add_argument("--style_extractor_path",  required=True,
                        help="訓練済み StyleExtractor チェックポイント (best_model.pt)")
    parser.add_argument("--data_fraction",   type=float, default=1.0)
    parser.add_argument("--use_augmentation", action='store_true')
    parser.add_argument("--latent_noise",    type=float, default=0.1)
    parser.add_argument("--latent_mask",     type=float, default=0.1)

    # 学習
    parser.add_argument("--epochs",          type=int,   default=60)
    parser.add_argument("--batch_size",      type=int,   default=64)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--weight_decay",    type=float, default=1e-2)
    parser.add_argument("--scheduler",       choices=['none', 'cosine', 'plateau'],
                        default='plateau')
    parser.add_argument("--use_class_weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--mixup",           type=float, default=0.0)

    # モデル
    parser.add_argument("--latent_dim",  type=int,   default=512)
    parser.add_argument("--seq_len",     type=int,   default=18)
    parser.add_argument("--embed_dim",   type=int,   default=512)
    parser.add_argument("--depth",       type=int,   default=6)
    parser.add_argument("--heads",       type=int,   default=8)
    parser.add_argument("--mlp_dim",     type=int,   default=2048)
    parser.add_argument("--num_classes", type=int,   default=7)
    parser.add_argument("--dropout",     type=float, default=0.1)

    parser.add_argument("--seed",        type=int,   default=42)

    args = parser.parse_args()
    if not (0.0 < args.data_fraction <= 1.0):
        raise ValueError(f"data_fraction は (0, 1] の範囲で指定してください: {args.data_fraction}")
    main(args)
