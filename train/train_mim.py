"""
MIM (Masked Image Modeling) 事前学習スクリプト
VQ-GANトークンの一部をマスクし、ViTに元のトークンを予測させることで、
ラベルなしで顔の構造的特徴を学習する。
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# プロジェクトルートへのパス追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 自作モジュールのインポート
from data.token_dataset import TokenDataset
from models_fer_vit.beit_vit import TokenViT
from utils.experiment_logger import ExperimentLogger

def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def apply_random_mask(tokens, mask_token_id, mask_ratio=0.4, device='cuda'):
    """
    トークン列にランダムマスクを適用する
    Args:
        tokens: (B, L) 入力トークン列
        mask_token_id: MASKとして使うID (Vocab Sizeと同じ値)
        mask_ratio: マスクする割合 (0.0 ~ 1.0)
    Returns:
        masked_tokens: マスク適用後のトークン列
        mask_bool: マスク位置を示すブール行列 (Loss計算用)
    """
    B, L = tokens.shape
    
    # 乱数を生成して、mask_ratioより小さい部分をマスク対象とする
    # (B, L) のランダム値
    rand = torch.rand(B, L, device=device)
    
    # MASKする位置をTrueにする (CLSトークン等は考慮せず、単純に全体から選ぶ)
    # 必要なら CLSトークンの位置を除外する処理を追加してもよい
    mask_bool = rand < mask_ratio
    
    # 元のトークンをコピーしてマスク適用
    masked_tokens = tokens.clone()
    masked_tokens[mask_bool] = mask_token_id
    
    return masked_tokens, mask_bool

def train_epoch(model, loader, optimizer, criterion, device, args):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    
    # tqdmでプログレスバー表示
    pbar = tqdm(loader, desc="Training")
    
    for tokens, _ in pbar:
        # ラベル(_)はMIMでは使わないので捨てる
        tokens = tokens.to(device) # これが「正解データ (Target)」
        
        # 1. ランダムマスキング適用
        masked_tokens, mask_bool = apply_random_mask(
            tokens, 
            mask_token_id=args.vocab_size, # vocab_size番目をMASK IDとする
            mask_ratio=args.mask_ratio,
            device=device
        )
        
        optimizer.zero_grad()
        
        # 2. モデルに入力 (MIMモード)
        # 出力: (B, L, Vocab_Size)
        outputs = model(masked_tokens)
        
        # 3. 損失計算
        # PyTorchのCrossEntropyは (N, C) と (N) を期待する
        # 全トークンではなく、「マスクされた部分」だけでLossを計算するのが一般的
        
        # マスクされた部分の予測ログジット: (N_masked, Vocab_Size)
        pred_masked = outputs[mask_bool]
        # マスクされた部分の正解トークン: (N_masked,)
        target_masked = tokens[mask_bool]
        
        if len(target_masked) > 0:
            loss = criterion(pred_masked, target_masked)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 精度計算 (Reconstruction Accuracy)
            pred_ids = pred_masked.argmax(dim=1)
            correct = (pred_ids == target_masked).sum().item()
            total_correct += correct
            total_masked += len(target_masked)
            
            # プログレスバー更新
            pbar.set_postfix({'loss': loss.item(), 'acc': correct / len(target_masked)})
        
    avg_loss = total_loss / len(loader)
    avg_acc = total_correct / total_masked if total_masked > 0 else 0
    
    return avg_loss, avg_acc

@torch.no_grad()
def evaluate(model, loader, criterion, device, args):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    
    for tokens, _ in loader:
        tokens = tokens.to(device)
        
        # 検証時もランダムマスクを行い、復元できるか試す
        masked_tokens, mask_bool = apply_random_mask(
            tokens, 
            mask_token_id=args.vocab_size,
            mask_ratio=args.mask_ratio,
            device=device
        )
        
        outputs = model(masked_tokens)
        
        pred_masked = outputs[mask_bool]
        target_masked = tokens[mask_bool]
        
        if len(target_masked) > 0:
            loss = criterion(pred_masked, target_masked)
            total_loss += loss.item()
            
            pred_ids = pred_masked.argmax(dim=1)
            total_correct += (pred_ids == target_masked).sum().item()
            total_masked += len(target_masked)
            
    avg_loss = total_loss / len(loader)
    avg_acc = total_correct / total_masked if total_masked > 0 else 0
    
    return avg_loss, avg_acc

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. データセット準備
    print(f"\nLoading Token Dataset from: {args.token_dir}")
    # 実際には train/val を分けて保存しているファイルを想定
    train_path = os.path.join(args.token_dir, 'rafdb_train_tokens.pt')
    val_path = os.path.join(args.token_dir, 'rafdb_test_tokens.pt')
    
    train_ds = TokenDataset(train_path)
    val_ds = TokenDataset(val_path)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    
    # シーケンス長を取得
    sample_token, _ = train_ds[0]
    seq_len = len(sample_token)
    print(f"Sequence length: {seq_len}")
    
    # 2. モデル作成 (MIMモード)
    print(f"\nCreating TokenViT (MIM Mode)...")
    model = TokenViT(
        vocab_size=args.vocab_size,
        seq_len=seq_len,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        num_classes=args.num_classes, # MIMでは使われないが引数として必要なら渡す
        dropout=args.dropout,
        use_mim=True # 重要: MIMモードを有効化
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {n_params:,}")
    
    # 3. 学習設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # ロガー設定
    experiment_name = f"MIM_vit_d{args.depth}_h{args.heads}_mask{int(args.mask_ratio*100)}"
    logger = ExperimentLogger(experiment_name, base_dir="experiments_mim")
    
    config = vars(args)
    logger.log_config(config)
    
    # 4. 学習ループ
    print("\nStarting MIM Pre-training...")
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, args)
        
        # Val
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, args)
        
        # Log
        logger.log_metrics({
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc
        }, epoch)
        logger.log_learning_rate(optimizer, epoch)
        
        scheduler.step()
        
        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Checkpoint Save
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            print(" -> Best Model Saved!")
            
        logger.save_checkpoint(
            model, optimizer, epoch, 
            {'loss': val_loss, 'acc': val_acc}, 
            is_best
        )

    logger.close()
    print("\nMIM Pre-training Completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MIM (Masked Image Modeling) for FER ViT")
    
    # データ設定
    parser.add_argument("--token_dir", type=str, required=True, help="Directory containing .pt token files")
    parser.add_argument("--vocab_size", type=int, default=16384, help="VQ-GAN Codebook size")
    
    # マスク設定
    parser.add_argument("--mask_ratio", type=float, default=0.4, help="Ratio of tokens to mask (e.g. 0.4)")
    
    # モデル設定
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=7) # モデル初期化用
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # 学習設定
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.5e-4) # MIMは少し高めのLRが良い場合が多い
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)