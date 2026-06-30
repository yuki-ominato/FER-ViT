"""
FER特化StyleExtractor 学習スクリプト

train_style_extractor.py を FER 損失に対応させたバリアント。
AFSFERLoss を使い、StyleExtractor h と ExprClassifier を共同学習する。

主な変更点（対 train_style_extractor.py）:
    ・損失関数を AFSFERLoss に変更
        L_expr   : h(w) が感情ラベルに識別可能か（ジェネレータ不要）
        L_neutral: w − h(w) が無表情ラベル(=4)に識別可能か（ジェネレータ不要）
        L_id     : ArcFace によるアイデンティティ保存（ジェネレータ必要）
        L_sparse : 非表情 W+ 層のスパース性（ジェネレータ不要）
        L_cons   : h の一貫性（ジェネレータ不要）
    ・ImageProvider が不要（L_expr / L_neutral は潜在空間で完結）
    ・L_id 用に G(w_new) と G(w_src) を内部で生成
    ・optimizer が h と criterion.classifier を両方最適化する

Usage:
    # generator あり（L_id 有効）
    python train/train_fer_extractor.py \\
        --latent_dir     latents/fer2013/train \\
        --val_latent_dir latents/fer2013/val \\
        --psp_path       pretrained_models/e4e_ffhq_encode.pt \\
        --arcface_path   pretrained_models/model_ir_se50.pth \\
        --out_dir        outputs/afs_fer \\
        --epochs         10 --batch_size 4

    # generator なし（L_id = 0、高速）
    python train/train_fer_extractor.py \\
        --latent_dir     latents/fer2013/train \\
        --val_latent_dir latents/fer2013/val \\
        --psp_path       pretrained_models/e4e_ffhq_encode.pt \\
        --arcface_path   pretrained_models/model_ir_se50.pth \\
        --out_dir        outputs/afs_fer \\
        --no_generator \\
        --epochs         10 --batch_size 16
"""

import os
import sys
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_PSP_ROOT = os.path.join(_PROJECT_ROOT, 'third_party', 'pixel2style2pixel')
if _PSP_ROOT not in sys.path:
    sys.path.insert(0, _PSP_ROOT)

from afs import StyleExtractor, PairLatentDataset
from afs.fer_losses import AFSFERLoss


# ---------------------------------------------------------------------------
# Generator loader (train_style_extractor.py から流用)
# ---------------------------------------------------------------------------

def load_generator(psp_path: str, device: torch.device):
    """
    pSp / e4e チェックポイントから凍結済み StyleGAN2 Generator を抽出する。
    pSp と e4e はどちらも state_dict['decoder.*'] にジェネレータ重みを保持する。

    Returns: (generator, face_pool) or (None, None)
    """
    from models.stylegan2.model import Generator

    print(f"Loading StyleGAN2 generator from {psp_path} ...")
    ckpt = torch.load(psp_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)

    decoder_state = {
        k[len('decoder.'):]: v
        for k, v in state.items()
        if k.startswith('decoder.')
    }

    generator = Generator(1024, 512, 8)
    generator.load_state_dict(decoder_state)
    generator.eval()
    for p in generator.parameters():
        p.requires_grad_(False)

    face_pool = nn.AdaptiveAvgPool2d((256, 256))

    return generator.to(device), face_pool.to(device)


# ---------------------------------------------------------------------------
# 1 エポック処理
# ---------------------------------------------------------------------------

def run_epoch(
    h: nn.Module,
    criterion: AFSFERLoss,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    generator=None,
    face_pool=None,
) -> dict:
    """
    Train / Validation の 1 エポック。optimizer が None の場合は validation モード。
    """
    is_train = optimizer is not None
    h.train() if is_train else h.eval()
    criterion.train() if is_train else criterion.eval()

    totals = {"loss": 0.0, "expr": 0.0, "id": 0.0,
              "neutral": 0.0, "sparse": 0.0, "cons": 0.0}
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for w_src, label_src, _path_src, w_tgt, label_tgt, _path_tgt in loader:
            w_src     = w_src.to(device)
            w_tgt     = w_tgt.to(device)
            label_src = label_src.to(device)
            label_tgt = label_tgt.to(device)

            # ---- StyleExtractor の適用 ----
            h_src = h(w_src)                              # (B, 18, 512)
            h_tgt = h(w_tgt)
            w_new = (w_src - h_src) + h_tgt              # 表情転送後の潜在コード
            h_new = h(w_new)                              # 一貫性損失用

            # ---- L_id 用画像生成（generator が利用可能な場合のみ）----
            img_gen = img_src_gen = None
            if generator is not None:
                with torch.no_grad() if not is_train else torch.enable_grad():
                    # 勾配は w_new を経由して h に流れるため、img_gen は勾配あり
                    pass
                img_gen_raw, _ = generator(
                    [w_new],
                    input_is_latent=True,
                    randomize_noise=False,
                    return_latents=False,
                )
                img_gen = face_pool(img_gen_raw)          # (B, 3, 256, 256)

                with torch.no_grad():
                    img_src_raw, _ = generator(
                        [w_src],
                        input_is_latent=True,
                        randomize_noise=False,
                        return_latents=False,
                    )
                    img_src_gen = face_pool(img_src_raw)  # 固定参照; 勾配不要

            # ---- 損失計算 ----
            loss, metrics = criterion(
                h_src, h_tgt, h_new,
                w_src, w_tgt,
                label_src, label_tgt,
                img_gen=img_gen,
                img_src=img_src_gen,
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(h.parameters()) + list(criterion.classifier.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()

            totals["loss"] += loss.item()
            for k in metrics:
                totals[k] += metrics[k]
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FER-focused Style Extractor")

    # データ
    p.add_argument("--latent_dir",      required=True,
                   help="訓練用潜在コードディレクトリ")
    p.add_argument("--val_latent_dir",  default=None,
                   help="検証用潜在コードディレクトリ（省略時は train loss で best 判定）")

    # モデル
    p.add_argument("--psp_path",        required=True,
                   help="pSp / e4e チェックポイント（StyleGAN2 decoder を抽出）")
    p.add_argument("--arcface_path",    required=True,
                   help="model_ir_se50.pth へのパス")
    p.add_argument("--no_generator",    action="store_true",
                   help="指定すると generator をロードせず L_id = 0 で学習（高速）")

    # 出力
    p.add_argument("--out_dir",         default="outputs/afs_fer")

    # 学習
    p.add_argument("--epochs",          type=int,   default=10)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--lr",              type=float, default=1e-4)

    # 損失係数
    p.add_argument("--lambda_expr",     type=float, default=1.0,
                   help="L_expr の係数（表情識別可能性）")
    p.add_argument("--lambda_id",       type=float, default=1.0,
                   help="L_id の係数（アイデンティティ保存; generator が必要）")
    p.add_argument("--lambda_neutral",  type=float, default=0.5,
                   help="L_neutral の係数（残差コードの無表情化）")
    p.add_argument("--lambda_sparse",   type=float, default=0.02,
                   help="L_sparse の係数（非表情 W+ 層のスパース性）")
    p.add_argument("--lambda_cons",     type=float, default=0.1,
                   help="L_cons の係数（スタイル抽出の一貫性）")

    p.add_argument("--device",          default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, run_id)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- Generator（オプション）---
    generator = face_pool = None
    if not args.no_generator:
        try:
            generator, face_pool = load_generator(args.psp_path, device)
        except Exception as e:
            print(f"Warning: generator のロードに失敗しました ({e})。L_id = 0 で続行します。")

    # --- StyleExtractor & 損失関数 ---
    h = StyleExtractor().to(device)
    print(f"StyleExtractor parameters: {sum(p.numel() for p in h.parameters()):,}")

    criterion = AFSFERLoss(
        arcface_path  = args.arcface_path,
        generator     = generator,
        lambda_expr   = args.lambda_expr,
        lambda_id     = args.lambda_id,
        lambda_neutral= args.lambda_neutral,
        lambda_sparse = args.lambda_sparse,
        lambda_cons   = args.lambda_cons,
    ).to(device)
    print(f"ExprClassifier parameters: {sum(p.numel() for p in criterion.classifier.parameters()):,}")

    # --- Optimizer: h と ExprClassifier を共同最適化 ---
    optimizer = torch.optim.Adam(
        list(h.parameters()) + list(criterion.classifier.parameters()),
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # --- Dataset ---
    train_ds = PairLatentDataset(args.latent_dir)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Train: {len(train_ds)} samples")

    val_loader = None
    if args.val_latent_dir is not None:
        val_ds = PairLatentDataset(args.val_latent_dir)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        print(f"Val:   {len(val_ds)} samples")

    # --- 学習ループ ---
    monitor_key = "val_loss" if val_loader is not None else "train_loss"
    print(f"Best model criterion: {monitor_key}")
    print(f"Loss weights: expr={args.lambda_expr} id={args.lambda_id} "
          f"neutral={args.lambda_neutral} sparse={args.lambda_sparse} "
          f"cons={args.lambda_cons}")

    log = []
    best_loss = float("inf")
    best_model_path = os.path.join(ckpt_dir, "best_model.pt")
    last_model_path = os.path.join(ckpt_dir, "last_model.pt")

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(h, criterion, train_loader, optimizer, device,
                            generator, face_pool)
        scheduler.step()

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()}}

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_m['loss']:.4f}  "
            f"expr={train_m['expr']:.4f}  "
            f"id={train_m['id']:.4f}  "
            f"neutral={train_m['neutral']:.4f}  "
            f"sparse={train_m['sparse']:.4f}  "
            f"cons={train_m['cons']:.4f}",
            end="",
        )

        if val_loader is not None:
            val_m = run_epoch(h, criterion, val_loader, None, device,
                              generator, face_pool)
            row.update({f"val_{k}": v for k, v in val_m.items()})
            print(f"  val_loss={val_m['loss']:.4f}", end="")
            monitor_loss = val_m["loss"]
        else:
            monitor_loss = train_m["loss"]

        print()
        log.append(row)

        ckpt = {
            "epoch":            epoch,
            "model_state":      h.state_dict(),
            "classifier_state": criterion.classifier.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            monitor_key:        monitor_loss,
        }
        torch.save(ckpt, last_model_path)

        if monitor_loss < best_loss:
            best_loss = monitor_loss
            torch.save(ckpt, best_model_path)
            print(f"  → best_model saved ({monitor_key}={best_loss:.4f})")

    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n学習完了。チェックポイント: {ckpt_dir}")
    print(f"ExprClassifier の重みは best_model.pt の 'classifier_state' に保存されています。")


if __name__ == "__main__":
    main()
