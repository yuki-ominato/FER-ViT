"""
ExpressionAwareViT の学習スクリプト。

【ワークフロー】
  Step 1 (一度だけ):
      python latent_analysis/compute_expression_directions.py \\
          --latent_dir /path/to/latents/train \\
          --output_dir ./latent_analysis/directions

  Step 2 (学習):
      python train/train_expression_aware_vit.py \\
          --latent_train_dir /path/to/latents/train \\
          --latent_val_dir   /path/to/latents/val \\
          --directions_path  ./latent_analysis/directions/binary_directions.pt \\
          --output_mode expr_only \\
          --model_size small \\
          --use_pretrained \\
          --use_class_weights \\
          --epochs 60
"""

import os
import sys
import argparse
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
from models_fer_vit.expression_aware_vit import ExpressionAwareViT
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
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = Counter(labels)
    total = len(labels)
    num_cls = max(class_counts) + 1
    weights = [total / (num_cls * class_counts.get(i, 1)) for i in range(num_cls)]
    return torch.FloatTensor(weights)


def get_optimizer_groups(model: ExpressionAwareViT, lr: float, weight_decay: float):
    """ViT 内の各コンポーネントにレイヤーごとの学習率を設定"""
    vit = model.vit
    param_groups = []

    input_proj_params = list(vit.input_proj.parameters())
    if input_proj_params:
        param_groups.append({'params': input_proj_params, 'lr': lr * 10, 'weight_decay': weight_decay})
        print(f"  input_proj   : lr={lr*10:.2e}")

    transformer_params = [p for p in vit.transformer.parameters() if p.requires_grad]
    if transformer_params:
        param_groups.append({'params': transformer_params, 'lr': lr, 'weight_decay': weight_decay})
        print(f"  transformer  : lr={lr:.2e}, params={len(transformer_params)}")

    if vit.use_adapter:
        adapter_params = [p for p in vit.adapters.parameters() if p.requires_grad]
        if adapter_params:
            param_groups.append({'params': adapter_params, 'lr': lr * 10, 'weight_decay': weight_decay})
            print(f"  adapters     : lr={lr*10:.2e}")

    head_params = list(vit.head.parameters())
    if head_params:
        param_groups.append({'params': head_params, 'lr': lr * 10, 'weight_decay': weight_decay})
        print(f"  head         : lr={lr*10:.2e}")

    other_params = [vit.pos_embed, vit.cls_token]
    param_groups.append({'params': other_params, 'lr': lr * 5, 'weight_decay': 0})
    print(f"  pos/cls      : lr={lr*5:.2e}")

    return param_groups


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for latents, labels in loader:
        latents = latents.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(latents)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * latents.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for latents, labels in loader:
        latents = latents.to(device)
        labels = labels.to(device)

        logits = model(latents)
        loss = criterion(logits, labels)
        total_loss += loss.item() * latents.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    return {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'predictions': all_preds,
        'labels': all_labels,
    }


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\n" + "=" * 60)
    print("Loading datasets ...")
    print("=" * 60)

    train_ds = LatentFERDataset(args.latent_train_dir)
    val_ds = LatentFERDataset(args.latent_val_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    print("\n" + "=" * 60)
    print("Creating ExpressionAwareViT ...")
    print("=" * 60)

    model = ExpressionAwareViT.from_config(
        directions_path=args.directions_path,
        model_size=args.model_size,
        num_classes=args.num_classes,
        use_pretrained=args.use_pretrained,
        freeze_transformer=args.freeze_transformer,
        freeze_stages=args.freeze_stages if args.freeze_stages > 0 else None,
        use_adapter=args.use_adapter,
        adapter_dim=args.adapter_dim,
        output_mode=args.output_mode,
        enhance_alpha=args.enhance_alpha,
        decompose_mode=args.decompose_mode,
    )
    model = model.to(device)
    model.print_info()

    if args.use_class_weights:
        weights = calculate_class_weights(train_ds).to(device)
        print(f"\nClass weights: {weights.cpu().numpy().round(3)}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    print("\nOptimizer parameter groups:")
    if args.use_layerwise_lr:
        param_groups = get_optimizer_groups(model, args.lr, args.weight_decay)
        optimizer = optim.AdamW(param_groups)
    else:
        optimizer = optim.AdamW(model.get_trainable_params(),
                                lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5, verbose=True)
    else:
        scheduler = None

    model_cfg = {
        'type': 'ExpressionAwareViT',
        'directions_path': args.directions_path,
        'output_mode': args.output_mode,
        'decompose_mode': args.decompose_mode,
        'enhance_alpha': args.enhance_alpha,
        'model_size': args.model_size,
        'use_pretrained': args.use_pretrained,
        'freeze_transformer': args.freeze_transformer,
        'freeze_stages': args.freeze_stages if args.freeze_stages > 0 else None,
        'use_adapter': args.use_adapter,
        'adapter_dim': args.adapter_dim if args.use_adapter else None,
    }
    training_cfg = {
        'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr,
        'weight_decay': args.weight_decay, 'scheduler': args.scheduler,
        'use_class_weights': args.use_class_weights,
        'use_layerwise_lr': args.use_layerwise_lr, 'seed': args.seed,
    }
    config = {
        'model': model_cfg, 'training': training_cfg,
        'data': {'train_dir': args.latent_train_dir, 'val_dir': args.latent_val_dir},
    }

    exp_name = f"expr_aware_vit_{create_experiment_name(model_cfg, training_cfg)}"
    logger = ExperimentLogger(exp_name, base_dir='experiments')
    logger.log_config(config)

    print("\n" + "=" * 60)
    print("Training ...")
    print("=" * 60)

    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device)
        val_res = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | "
            f"val loss={val_res['loss']:.4f} acc={val_res['accuracy']:.4f} "
            f"f1={val_res['f1_macro']:.4f}"
        )

        logger.log_metrics({
            'train_loss': train_loss, 'train_acc': train_acc, 'train_f1': train_f1,
            'val_loss': val_res['loss'], 'val_acc': val_res['accuracy'],
            'val_f1': val_res['f1_macro'],
        }, epoch)
        logger.log_learning_rate(optimizer, epoch)

        if epoch % 10 == 0:
            logger.log_parameters(model, epoch)
            logger.log_gradients(model, epoch)

        is_best = val_res['f1_macro'] > best_f1
        if is_best:
            best_f1 = val_res['f1_macro']
            print(f"  -> Best model (F1: {best_f1:.4f})")

        logger.save_checkpoint(model, optimizer, epoch, val_res, is_best)

        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_res['f1_macro'])
            else:
                scheduler.step()

    print("\n" + "=" * 60)
    print(f"Training done! Best val F1: {best_f1:.4f}")
    print("=" * 60)

    final = evaluate(model, val_loader, criterion, device)
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    print(classification_report(final['labels'], final['predictions'],
                                target_names=emotion_names))

    logger.log_confusion_matrix(final['labels'], final['predictions'],
                                emotion_names, args.epochs)
    logger.log_experiment_summary({
        'accuracy': final['accuracy'], 'f1_macro': final['f1_macro'],
        'f1_weighted': final['f1_weighted'], 'best_f1_macro': best_f1,
    })
    logger.close()
    print(f"\nResults saved to: {logger.get_experiment_path()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train ExpressionAwareViT "
                    "(InterFaceGAN-style decomposition + HybridLatentViT)"
    )
    parser.add_argument('--latent_train_dir', required=True)
    parser.add_argument('--latent_val_dir', required=True)
    parser.add_argument('--directions_path', required=True)
    parser.add_argument('--output_mode',
                        choices=['expr_only', 'id_only', 'enhanced', 'concat'],
                        default='expr_only')
    parser.add_argument('--decompose_mode',
                        choices=['all_classes', 'max_class'], default='all_classes')
    parser.add_argument('--enhance_alpha', type=float, default=2.0)
    parser.add_argument('--model_size', choices=['tiny', 'small', 'base'], default='small')
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--use_pretrained', action='store_true', default=False)
    parser.add_argument('--freeze_transformer', action='store_true')
    parser.add_argument('--freeze_stages', type=int, default=0)
    parser.add_argument('--use_adapter', action='store_true')
    parser.add_argument('--adapter_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--scheduler',
                        choices=['none', 'cosine', 'plateau'], default='plateau')
    parser.add_argument('--use_class_weights', action='store_true')
    parser.add_argument('--use_layerwise_lr', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)