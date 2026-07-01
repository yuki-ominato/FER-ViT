"""
W+ 潜在コードを w / w_expr / w_id に分解して StyleGAN2 で画像化し、
横並びグリッドとして保存する視覚的評価スクリプト。

出力グリッドのレイアウト:
    列: G(w)  |  G(w_expr)  |  G(w_id)
    行: サンプルごと（左端に感情ラベル）

w_expr = h(w)       ← StyleExtractor の出力（表情成分）
w_id   = w - h(w)   ← 残差（アイデンティティ成分）

Usage:
    # ランダムに n_samples 枚選ぶ
    python eval/visualize_decomposition.py \\
        --latent_dir  latents/fer2013_e4e/val \\
        --extractor   experiments/extractor/.../best_model.pt \\
        --psp_path    pretrained_models/e4e_ffhq_encode.pt \\
        --out_dir     eval_output/decomposition \\
        --n_samples   8

    # 感情クラスごとに 1 枚ずつ（計 7 行）選ぶ
    python eval/visualize_decomposition.py \\
        --latent_dir  latents/fer2013_e4e/val \\
        --extractor   experiments/extractor/.../best_model.pt \\
        --psp_path    pretrained_models/e4e_ffhq_encode.pt \\
        --out_dir     eval_output/decomposition \\
        --per_class

    # 特定の感情ラベルのみ（0=angry, 3=happy, 4=neutral）
    python eval/visualize_decomposition.py \\
        --latent_dir  latents/fer2013_e4e/val \\
        --extractor   experiments/extractor/.../best_model.pt \\
        --psp_path    pretrained_models/e4e_ffhq_encode.pt \\
        --out_dir     eval_output/decomposition \\
        --labels      0 3 4 \\
        --n_samples   3
"""

import os
import sys
import argparse
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_PSP_ROOT = os.path.join(_PROJECT_ROOT, 'third_party', 'pixel2style2pixel')
if _PSP_ROOT not in sys.path:
    sys.path.insert(0, _PSP_ROOT)

from afs import StyleExtractor

LABEL_NAMES = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

COL_TITLES = ["G(w)\noriginal", "G(w_expr)\nexpression", "G(w_id)\nidentity"]


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_generator(psp_path: str, device: torch.device):
    from models.stylegan2.model import Generator

    print(f"Loading StyleGAN2 generator from {psp_path} ...")
    ckpt  = torch.load(psp_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    decoder_state = {k[len("decoder."):]: v for k, v in state.items()
                     if k.startswith("decoder.")}
    gen = Generator(1024, 512, 8)
    gen.load_state_dict(decoder_state)
    gen.eval()
    for p in gen.parameters():
        p.requires_grad_(False)
    face_pool = nn.AdaptiveAvgPool2d((256, 256))
    return gen.to(device), face_pool.to(device)


def load_extractor(extractor_path: str, device: torch.device) -> StyleExtractor:
    print(f"Loading StyleExtractor from {extractor_path} ...")
    ckpt  = torch.load(extractor_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    h = StyleExtractor()
    h.load_state_dict(state)
    h.eval()
    for p in h.parameters():
        p.requires_grad_(False)
    return h.to(device)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def generate_image(generator, face_pool, w: torch.Tensor) -> torch.Tensor:
    """W+ コード w (1,18,512) から [3,256,256] テンソル（[-1,1]）を返す。"""
    with torch.no_grad():
        img, _ = generator(
            [w],
            input_is_latent=True,
            randomize_noise=False,
            return_latents=False,
        )
    return face_pool(img)[0]  # (3, 256, 256)


def tensor_to_pil(t: torch.Tensor, size: int) -> Image.Image:
    """[-1, 1] の (C, H, W) テンソルを PIL Image (RGB) に変換する。"""
    t = t.clamp(-1.0, 1.0).cpu()
    arr = ((t + 1.0) / 2.0 * 255.0).permute(1, 2, 0).numpy().astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

LABEL_COL_W = 80   # 左端の感情ラベル列の幅 (px)
HEADER_H    = 48   # 列タイトル行の高さ (px)
PAD         = 4    # 画像間の余白 (px)
BG_COLOR    = (30, 30, 30)
TEXT_COLOR  = (220, 220, 220)
TITLE_COLOR = (255, 215, 0)


def build_grid(
    rows: list[list[Image.Image]],
    row_labels: list[str],
    col_titles: list[str],
    img_size: int,
) -> Image.Image:
    """
    rows        : n_samples × n_cols の PIL Image リスト
    row_labels  : 各行の感情ラベル文字列
    col_titles  : 各列のタイトル文字列
    img_size    : 各セル画像の一辺 (px)
    """
    n_rows = len(rows)
    n_cols = len(col_titles)

    total_w = LABEL_COL_W + n_cols * (img_size + PAD) + PAD
    total_h = HEADER_H   + n_rows * (img_size + PAD) + PAD

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)

    # --- 列タイトル ---
    for c, title in enumerate(col_titles):
        cx = LABEL_COL_W + PAD + c * (img_size + PAD) + img_size // 2
        for i, line in enumerate(title.split("\n")):
            draw.text((cx, 6 + i * 18), line, fill=TITLE_COLOR, anchor="mt")

    # --- 各行 ---
    for r, (imgs, label) in enumerate(zip(rows, row_labels)):
        y = HEADER_H + PAD + r * (img_size + PAD)

        # 感情ラベル（左端）
        draw.text(
            (LABEL_COL_W // 2, y + img_size // 2),
            label,
            fill=TEXT_COLOR,
            anchor="mm",
        )

        # 画像3枚
        for c, img in enumerate(imgs):
            x = LABEL_COL_W + PAD + c * (img_size + PAD)
            canvas.paste(img, (x, y))

    return canvas


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def load_files(latent_dir: str) -> list[Path]:
    return sorted(Path(latent_dir).glob("*.pt"))


def select_samples(
    files: list[Path],
    n_samples: int,
    per_class: bool,
    labels: list[int] | None,
    seed: int,
) -> list[Path]:
    rng = random.Random(seed)

    if per_class:
        by_label: dict[int, list[Path]] = defaultdict(list)
        for f in files:
            data = torch.load(f, map_location="cpu", weights_only=False)
            by_label[int(data["label"])].append(f)
        target_labels = labels if labels is not None else sorted(by_label.keys())
        selected = []
        for lbl in target_labels:
            pool = by_label.get(lbl, [])
            if pool:
                selected.append(rng.choice(pool))
        return selected

    if labels is not None:
        filtered = []
        for f in files:
            data = torch.load(f, map_location="cpu", weights_only=False)
            if int(data["label"]) in labels:
                filtered.append(f)
        files = filtered

    if len(files) > n_samples:
        files = rng.sample(files, n_samples)
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize W+ decomposition via StyleGAN2")
    p.add_argument("--latent_dir", required=True,
                   help="潜在コード .pt ファイルのディレクトリ")
    p.add_argument("--extractor",  required=True,
                   help="StyleExtractor チェックポイント (.pt)")
    p.add_argument("--psp_path",   required=True,
                   help="e4e / pSp チェックポイント（StyleGAN2 decoder の抽出に使用）")
    p.add_argument("--out_dir",    default="eval_output/decomposition",
                   help="出力ディレクトリ")
    p.add_argument("--out_name",   default="decomposition.png",
                   help="出力ファイル名")
    p.add_argument("--n_samples",  type=int, default=8,
                   help="可視化するサンプル数（--per_class 非使用時）")
    p.add_argument("--per_class",  action="store_true",
                   help="感情クラスごとに 1 サンプル選ぶ（計最大 7 行）")
    p.add_argument("--labels",     type=int, nargs="+", default=None,
                   help="可視化する感情ラベルを指定（例: --labels 0 3 4）")
    p.add_argument("--img_size",   type=int, default=256,
                   help="出力画像の一辺サイズ (px)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     default="cuda")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    generator, face_pool = load_generator(args.psp_path, device)
    h = load_extractor(args.extractor, device)

    all_files = load_files(args.latent_dir)
    if not all_files:
        raise FileNotFoundError(f"No .pt files found in {args.latent_dir}")

    selected = select_samples(
        all_files,
        n_samples=args.n_samples,
        per_class=args.per_class,
        labels=args.labels,
        seed=args.seed,
    )
    print(f"Selected {len(selected)} samples")

    rows       = []
    row_labels = []

    for fpath in selected:
        data  = torch.load(fpath, map_location="cpu", weights_only=False)
        w     = data["latent"].unsqueeze(0).to(device)  # (1, 18, 512)
        label = int(data["label"])

        with torch.no_grad():
            w_expr = h(w)        # (1, 18, 512)
            w_id   = w - w_expr  # (1, 18, 512)

        img_w    = tensor_to_pil(generate_image(generator, face_pool, w),      args.img_size)
        img_expr = tensor_to_pil(generate_image(generator, face_pool, w_expr), args.img_size)
        img_id   = tensor_to_pil(generate_image(generator, face_pool, w_id),   args.img_size)

        rows.append([img_w, img_expr, img_id])
        row_labels.append(LABEL_NAMES.get(label, str(label)))
        print(f"  {fpath.name}  label={row_labels[-1]}")

    grid     = build_grid(rows, row_labels, COL_TITLES, args.img_size)
    out_path = os.path.join(args.out_dir, args.out_name)
    grid.save(out_path)
    print(f"\nSaved → {out_path}  ({grid.size[0]}×{grid.size[1]} px)")


if __name__ == "__main__":
    main()
