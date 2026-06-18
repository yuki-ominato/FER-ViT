"""
AFS Style Extractor 学習スクリプト

Usage example:
    python train/train_style_extractor.py \\
        --latent_dir  /path/to/cache_latent \\
        --psp_path    /path/to/psp_ffhq_encode.pt \\
        --arcface_path third_party/pixel2style2pixel/pretrained_models/model_ir_se50.pth \\
        --out_dir      outputs/afs \\
        --epochs       50 \\
        --batch_size   8

Checkpoints saved to <out_dir>/<run_id>/checkpoints/:
    best_model.pt  — training loss が改善したエポックを保存（上書き）
    last_model.pt  — 毎エポック上書き（学習再開用）

Image provider selection:
    --provider b   (default) DiskImageProvider: reads img_path stored in .pt files.
    --provider a            GeneratedImageProvider: decodes G(w_src) / G(w_tgt) each step.
"""

import os
import sys
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# -- project root on sys.path --------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# pSp must also be on the path (for Generator and other imports used by afs/).
_PSP_ROOT = os.path.join(_PROJECT_ROOT, 'third_party', 'pixel2style2pixel')
if _PSP_ROOT not in sys.path:
    sys.path.insert(0, _PSP_ROOT)

from afs import StyleExtractor, AFSLoss, PairLatentDataset
from afs.image_provider import DiskImageProvider, GeneratedImageProvider


# ------------------------------------------------------------------------------
# Generator loader
# ------------------------------------------------------------------------------

def load_generator(psp_path: str, device: torch.device) -> tuple:
    """
    Extract the frozen StyleGAN2 generator and face_pool from a pSp checkpoint.

    Returns
    -------
    generator : Generator(1024, 512, 8)
    face_pool : AdaptiveAvgPool2d((256, 256))
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


# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

def train_one_epoch(
    h: nn.Module,
    generator: nn.Module,
    face_pool: nn.Module,
    criterion: AFSLoss,
    provider,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    h.train()
    totals = {"loss": 0.0, "id": 0.0, "lpips": 0.0, "cons": 0.0}
    n_batches = 0

    for w_src, _, paths_src, w_tgt, _, paths_tgt in loader:
        w_src = w_src.to(device)   # [B, 18, 512]
        w_tgt = w_tgt.to(device)

        # --- Forward through StyleExtractor ---
        w_sty_src = h(w_src)                          # [B, 18, 512]
        w_sty_tgt = h(w_tgt)
        w_new = (w_src - w_sty_src) + w_sty_tgt       # swapped latent
        w_sty_new = h(w_new)                           # for consistency loss

        # --- Generate swapped image (gradient flows through generator) ---
        img_gen, _ = generator(
            [w_new],
            input_is_latent=True,
            randomize_noise=False,
            return_latents=False,
        )
        img_gen = face_pool(img_gen)   # [B, 3, 256, 256]

        # --- Reference images (no gradient) ---
        paths_src = list(paths_src)
        paths_tgt = list(paths_tgt)
        img_src = provider.get_images(w_src, paths_src, device)
        img_tgt = provider.get_images(w_tgt, paths_tgt, device)

        # --- Loss ---
        loss, metrics = criterion(img_gen, img_src, img_tgt, w_sty_new, w_sty_tgt)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(h.parameters(), max_norm=1.0)
        optimizer.step()

        totals["loss"] += loss.item()
        for k in ("id", "lpips", "cons"):
            totals[k] += metrics[k]
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AFS Style Extractor")
    p.add_argument("--latent_dir",   required=True,
                   help="Directory with cached .pt latent files")
    p.add_argument("--psp_path",     required=True,
                   help="Path to pSp checkpoint (.pt) containing StyleGAN2 decoder")
    p.add_argument("--arcface_path", required=True,
                   help="Path to model_ir_se50.pth")
    p.add_argument("--out_dir",      default="outputs/afs",
                   help="Output directory for checkpoints and logs")
    p.add_argument("--provider",     choices=["a", "b"], default="b",
                   help="Image provider: a=GeneratedImageProvider, b=DiskImageProvider")
    p.add_argument("--epochs",       type=int, default=50)
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--lambda_cons",  type=float, default=0.1,
                   help="Weight for consistency loss")
    p.add_argument("--device",       default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, run_id)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save config
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- Models ---
    generator, face_pool = load_generator(args.psp_path, device)

    h = StyleExtractor().to(device)
    print(f"StyleExtractor parameters: {sum(p.numel() for p in h.parameters()):,}")

    criterion = AFSLoss(
        arcface_path=args.arcface_path,
        lambda_cons=args.lambda_cons,
    ).to(device)

    # --- Image provider ---
    if args.provider == "b":
        print("Using DiskImageProvider (案B)")
        provider = DiskImageProvider()
    else:
        print("Using GeneratedImageProvider (案A)")
        provider = GeneratedImageProvider(generator, face_pool)

    # --- Dataset ---
    dataset = PairLatentDataset(args.latent_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,   # random pairing in __getitem__; keep single process for safety
        pin_memory=(device.type == "cuda"),
    )

    # --- Optimiser ---
    optimizer = torch.optim.Adam(h.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # --- Training loop ---
    log = []
    best_loss = float("inf")
    best_model_path = os.path.join(ckpt_dir, "best_model.pt")
    last_model_path = os.path.join(ckpt_dir, "last_model.pt")

    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(
            h, generator, face_pool, criterion, provider, loader, optimizer, device
        )
        scheduler.step()

        log.append({"epoch": epoch, **metrics})
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={metrics['loss']:.4f}  "
            f"id={metrics['id']:.4f}  "
            f"lpips={metrics['lpips']:.4f}  "
            f"cons={metrics['cons']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": h.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": metrics["loss"],
        }

        # last_model は毎エポック上書き
        torch.save(ckpt, last_model_path)

        # best_model は損失が改善したときのみ保存
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(ckpt, best_model_path)
            print(f"  → best_model saved (loss={best_loss:.4f})")

    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print("Training complete.")


if __name__ == "__main__":
    main()
