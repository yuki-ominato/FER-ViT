"""
RAF-DB 向け AFS Style Extractor 評価スクリプト

定量評価
    id_sim  : ArcFace cosine similarity（G(w_new), G(w_src)）   高いほど良い
    cons    : L1( h(w_new), h(w_tgt) )                         低いほど良い
    ※ 全体 + クラス別の両方を出力

定性評価
    1. comparison_grid.png  : src | tgt | gen の比較グリッド
    2. cross_class_grid.png : 7×7 クロスクラス転写グリッド
       行 = src 感情（identity 元）、列 = tgt 感情（expression 元）

Usage:
    python eval/evaluate_afs_rafdb.py \\
        --ckpt_path experiments/afs_rafdb/20260625_120000/checkpoints/best_model.pt \\
        --psp_path  pretrained_models/psp_ffhq_encode.pt \\
        --arcface_path pretrained_models/model_ir_se50.pth \\
        --latent_dir latents/rafdb/test
"""

import os
import sys
import argparse
import json
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image
from PIL import Image, ImageDraw, ImageFont
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_PSP_ROOT     = os.path.join(_PROJECT_ROOT, 'third_party', 'pixel2style2pixel')
for _p in (_PROJECT_ROOT, _PSP_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from afs.style_extractor import StyleExtractor
from afs.losses import ArcFaceExtractor
from models.stylegan2.model import Generator

# RAF-DB 共通ラベル（generate_latents.py の RAFDB_TO_LABEL 変換後）
EMOTION_NAMES = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise',
}
N_CLASSES = 7


# ------------------------------------------------------------------------------
# Model loaders
# ------------------------------------------------------------------------------

def load_generator(psp_path: str, device: torch.device):
    ckpt = torch.load(psp_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    decoder_state = {k[len('decoder.'):]: v
                     for k, v in state.items() if k.startswith('decoder.')}
    gen = Generator(1024, 512, 8)
    gen.load_state_dict(decoder_state)
    gen.eval()
    for p in gen.parameters():
        p.requires_grad_(False)
    face_pool = nn.AdaptiveAvgPool2d((256, 256))
    return gen.to(device), face_pool.to(device)


def load_style_extractor(ckpt_path: str, device: torch.device) -> StyleExtractor:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    h = StyleExtractor()
    h.load_state_dict(ckpt['model_state'])
    h.eval()
    return h.to(device)


@torch.no_grad()
def decode(w: torch.Tensor, generator, face_pool) -> torch.Tensor:
    """W+ [B,18,512] → 256px 画像 [B,3,256,256] in [-1,1]"""
    img, _ = generator([w], input_is_latent=True,
                       randomize_noise=False, return_latents=False)
    return face_pool(img)


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------

class PairedEvalDataset(Dataset):
    """
    テスト latent をシード固定でランダムペアリング。
    全サンプルを src として使い、tgt はシード固定 shuffle で割り当て（自己ペア除外）。
    """

    def __init__(self, latent_dir: str, seed: int = 0) -> None:
        latent_dir = os.path.abspath(latent_dir)
        files = sorted(
            os.path.join(latent_dir, f)
            for f in os.listdir(latent_dir) if f.endswith('.pt')
        )
        rng = random.Random(seed)
        tgt_indices = list(range(len(files)))
        rng.shuffle(tgt_indices)
        for i, ti in enumerate(tgt_indices):
            if ti == i:
                swap = (i + 1) % len(files)
                tgt_indices[i], tgt_indices[swap] = tgt_indices[swap], tgt_indices[i]

        self.src_files = files
        self.tgt_files = [files[i] for i in tgt_indices]

    def __len__(self):
        return len(self.src_files)

    def __getitem__(self, idx):
        src = torch.load(self.src_files[idx], map_location='cpu', weights_only=False)
        tgt = torch.load(self.tgt_files[idx], map_location='cpu', weights_only=False)
        return (src['latent'], int(src['label']),
                tgt['latent'], int(tgt['label']))


# ------------------------------------------------------------------------------
# Quantitative evaluation
# ------------------------------------------------------------------------------

@torch.no_grad()
def evaluate_metrics(h, generator, face_pool, arcface, loader, device):
    """全体および感情クラス別の id_sim / cons を返す。"""
    id_sims_all, cons_all = [], []
    id_sims_cls  = defaultdict(list)
    cons_cls     = defaultdict(list)

    for w_src, labels_src, w_tgt, labels_tgt in loader:
        w_src = w_src.to(device)
        w_tgt = w_tgt.to(device)

        w_sty_src = h(w_src)
        w_sty_tgt = h(w_tgt)
        w_new     = (w_src - w_sty_src) + w_sty_tgt
        w_sty_new = h(w_new)

        img_src = decode(w_src, generator, face_pool)
        img_gen = decode(w_new, generator, face_pool)

        feat_src = arcface(img_src)
        feat_gen = arcface(img_gen)
        sim  = F.cosine_similarity(feat_gen, feat_src, dim=1).cpu()
        cons = F.l1_loss(w_sty_new, w_sty_tgt, reduction='none').mean(dim=(1, 2)).cpu()

        id_sims_all.append(sim)
        cons_all.append(cons)

        for i, lbl in enumerate(labels_src.tolist()):
            id_sims_cls[lbl].append(sim[i:i+1])
            cons_cls[lbl].append(cons[i:i+1])

    id_sims_all = torch.cat(id_sims_all)
    cons_all    = torch.cat(cons_all)

    result = {
        "overall": {
            "id_sim_mean": id_sims_all.mean().item(),
            "id_sim_std":  id_sims_all.std().item(),
            "cons_mean":   cons_all.mean().item(),
            "cons_std":    cons_all.std().item(),
            "n_samples":   len(id_sims_all),
        },
        "per_class": {},
    }
    for cls_id in range(N_CLASSES):
        if cls_id not in id_sims_cls:
            continue
        s = torch.cat(id_sims_cls[cls_id])
        c = torch.cat(cons_cls[cls_id])
        result["per_class"][EMOTION_NAMES[cls_id]] = {
            "id_sim_mean": s.mean().item(),
            "id_sim_std":  s.std().item(),
            "cons_mean":   c.mean().item(),
            "cons_std":    c.std().item(),
            "n_samples":   len(s),
        }
    return result


# ------------------------------------------------------------------------------
# Qualitative: comparison grid (src | tgt | gen)
# ------------------------------------------------------------------------------

@torch.no_grad()
def save_comparison_grid(h, generator, face_pool, dataset, device, out_path, n_vis=16):
    images = []
    for i in range(min(n_vis, len(dataset))):
        w_src, _, w_tgt, _ = dataset[i]
        w_src = w_src.unsqueeze(0).to(device)
        w_tgt = w_tgt.unsqueeze(0).to(device)

        w_sty_src = h(w_src)
        w_sty_tgt = h(w_tgt)
        w_new     = (w_src - w_sty_src) + w_sty_tgt

        img_src = decode(w_src, generator, face_pool)[0]
        img_tgt = decode(w_tgt, generator, face_pool)[0]
        img_gen = decode(w_new, generator, face_pool)[0]
        images.extend([img_src, img_tgt, img_gen])

    grid = make_grid(torch.stack(images), nrow=3, normalize=True, value_range=(-1, 1))
    pil = Image.fromarray(
        (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    )
    pil.save(out_path)
    print(f"  → comparison_grid saved: {out_path}")


# ------------------------------------------------------------------------------
# Qualitative: 7×7 cross-class transfer grid
# ------------------------------------------------------------------------------

def _pick_representatives(latent_dir: str, n_per_class: int = 1, seed: int = 42):
    """各感情クラスから n_per_class 件の latent をランダムに選ぶ。"""
    latent_dir = os.path.abspath(latent_dir)
    by_class = defaultdict(list)
    for f in sorted(os.listdir(latent_dir)):
        if not f.endswith('.pt'):
            continue
        path = os.path.join(latent_dir, f)
        data = torch.load(path, map_location='cpu', weights_only=False)
        lbl = int(data['label'])
        by_class[lbl].append(data['latent'])

    rng = random.Random(seed)
    reps = {}
    for cls_id in range(N_CLASSES):
        items = by_class.get(cls_id, [])
        if not items:
            continue
        chosen = rng.sample(items, min(n_per_class, len(items)))
        reps[cls_id] = torch.stack(chosen)  # [n, 18, 512]
    return reps


def _add_labels(grid_pil: Image.Image, cell_size: int, header_h: int = 28) -> Image.Image:
    """グリッド左端と上端に感情ラベルテキストを追加する。"""
    n = N_CLASSES
    label_w = 68
    img_w, img_h = grid_pil.size
    canvas = Image.new("RGB", (label_w + img_w, header_h + img_h), color=(30, 30, 30))
    canvas.paste(grid_pil, (label_w, header_h))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except Exception:
        font = ImageFont.load_default()
        small = font

    # 列ヘッダ（上）— expression source
    for j in range(n):
        cls_id = j
        name = EMOTION_NAMES.get(cls_id, str(cls_id))
        x = label_w + j * cell_size + cell_size // 2
        draw.text((x, 4), name, fill=(220, 220, 220), font=small, anchor="mt")

    # 行ヘッダ（左）— identity source
    for i in range(n):
        cls_id = i
        name = EMOTION_NAMES.get(cls_id, str(cls_id))
        y = header_h + i * cell_size + cell_size // 2
        draw.text((label_w - 4, y), name, fill=(220, 220, 220), font=small, anchor="rm")

    # 凡例
    draw.text((2, 2), "ID↓ Exp→", fill=(160, 160, 160), font=small)
    return canvas


@torch.no_grad()
def save_cross_class_grid(h, generator, face_pool, latent_dir, device, out_path, seed=42):
    """
    7×7 クロスクラス転写グリッドを保存する。

    行 = src 感情（identity 元）
    列 = tgt 感情（expression 元）
    各セルは G( w_id_src + w_sty_tgt )
    """
    reps = _pick_representatives(latent_dir, n_per_class=1, seed=seed)
    available = sorted(reps.keys())

    if len(available) < 2:
        print("  [WARN] cross_class_grid: クラス数が不足しているためスキップします。")
        return

    # 行×列ごとにレンダリング
    cells = []
    cell_cls_order = available
    for src_cls in cell_cls_order:
        row = []
        w_src = reps[src_cls][0:1].to(device)          # [1,18,512]
        w_sty_src = h(w_src)
        w_id_src  = w_src - w_sty_src

        for tgt_cls in cell_cls_order:
            w_tgt = reps[tgt_cls][0:1].to(device)
            w_sty_tgt = h(w_tgt)
            w_new = w_id_src + w_sty_tgt
            img   = decode(w_new, generator, face_pool)[0]  # [3,256,256]
            row.append(img)
        cells.extend(row)

    n_cols = len(cell_cls_order)
    grid = make_grid(
        torch.stack(cells),
        nrow=n_cols,
        normalize=True,
        value_range=(-1, 1),
        padding=2,
    )
    pil = Image.fromarray(
        (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    )

    cell_size = pil.width // n_cols
    pil = _add_labels(pil, cell_size=cell_size)
    pil.save(out_path)
    print(f"  → cross_class_grid saved: {out_path}")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="RAF-DB AFS Style Extractor 評価")
    p.add_argument("--ckpt_path",    required=True,  help="best_model.pt へのパス")
    p.add_argument("--psp_path",     required=True,  help="pSp チェックポイント")
    p.add_argument("--arcface_path", required=True,  help="model_ir_se50.pth へのパス")
    p.add_argument("--latent_dir",   required=True,  help="RAF-DB テスト潜在コードディレクトリ")
    p.add_argument("--out_dir",      default=None,
                   help="出力先（省略時は ckpt_path/../eval/）")
    p.add_argument("--n_vis",        type=int, default=16,
                   help="comparison_grid に使うペア数")
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--device",       default="cuda")
    return p.parse_args()


def _print_metrics(metrics: dict):
    ov = metrics["overall"]
    print(f"\n[全体]")
    print(f"  id_sim : {ov['id_sim_mean']:.4f} ± {ov['id_sim_std']:.4f}  (高いほど同一人物性が保持)")
    print(f"  cons   : {ov['cons_mean']:.4f} ± {ov['cons_std']:.4f}  (低いほどスタイル抽出が安定)")
    print(f"  n      : {ov['n_samples']}")

    print(f"\n[クラス別 id_sim / cons]")
    print(f"  {'Emotion':<12} {'id_sim':>10} {'cons':>10}  n")
    for name, m in metrics["per_class"].items():
        print(f"  {name:<12} {m['id_sim_mean']:>6.4f}±{m['id_sim_std']:.3f}"
              f"  {m['cons_mean']:>6.4f}±{m['cons_std']:.3f}  {m['n_samples']}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(args.ckpt_path)), "eval"
    )
    os.makedirs(out_dir, exist_ok=True)

    # --- Load models ---
    print("Loading models ...")
    generator, face_pool = load_generator(args.psp_path, device)
    h = load_style_extractor(args.ckpt_path, device)
    arcface = ArcFaceExtractor(args.arcface_path).to(device)

    # --- Dataset ---
    dataset = PairedEvalDataset(args.latent_dir, seed=0)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=0)
    print(f"Eval samples: {len(dataset)}")

    # --- Quantitative ---
    print("\n[定量評価]")
    metrics = evaluate_metrics(h, generator, face_pool, arcface, loader, device)
    _print_metrics(metrics)

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n  → metrics.json saved: {metrics_path}")

    # --- Qualitative: comparison grid ---
    print("\n[定性評価 1: comparison grid]")
    save_comparison_grid(
        h, generator, face_pool, dataset, device,
        out_path=os.path.join(out_dir, "comparison_grid.png"),
        n_vis=args.n_vis,
    )
    print("  列順: src（identity元） | tgt（expression元） | gen（転写結果）")

    # --- Qualitative: cross-class grid ---
    print("\n[定性評価 2: 7×7 cross-class transfer grid]")
    save_cross_class_grid(
        h, generator, face_pool, args.latent_dir, device,
        out_path=os.path.join(out_dir, "cross_class_grid.png"),
    )
    print("  行 = identity 元感情、列 = expression 元感情")


if __name__ == "__main__":
    main()
