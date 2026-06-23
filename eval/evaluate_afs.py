"""
AFS Style Extractor 評価スクリプト（Phase 3: Identity/Style 分離確認）

定量評価
    id_sim  : ArcFace cosine similarity（G(w_new), G(w_src)）  高いほど良い
    cons    : L1( h(w_new), h(w_tgt) )                        低いほど良い

定性評価
    src | tgt | gen の比較グリッド画像を保存

Usage:
    python eval/evaluate_afs.py \\
        --ckpt_path experiments/afs/20260619_090532/checkpoints/best_model.pt \\
        --psp_path  pretrained_models/psp_ffhq_encode.pt \\
        --arcface_path pretrained_models/model_ir_se50.pth \\
        --latent_dir latents/test \\
        --out_dir experiments/afs/20260619_090532/eval
"""

import os
import sys
import argparse
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from PIL import Image
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_PSP_ROOT     = os.path.join(_PROJECT_ROOT, 'third_party', 'pixel2style2pixel')
for p in (_PROJECT_ROOT, _PSP_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from afs.style_extractor import StyleExtractor
from afs.losses import ArcFaceExtractor
from models.stylegan2.model import Generator


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

EMOTION_NAMES = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                 4: 'neutral', 5: 'sad', 6: 'surprise'}


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
def decode(w, generator, face_pool):
    """W+ → 256px 画像 [B,3,256,256] in [-1,1]"""
    img, _ = generator([w], input_is_latent=True,
                       randomize_noise=False, return_latents=False)
    return face_pool(img)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """[-1,1] テンソル [3,H,W] → PIL"""
    t = (t.clamp(-1, 1) + 1) / 2          # [0,1]
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ------------------------------------------------------------------------------
# Eval dataset（決定論的ペアリング）
# ------------------------------------------------------------------------------

class PairedEvalDataset(Dataset):
    """
    テスト latent をシード固定でランダムペアリングする。
    全サンプルを src として使い、tgt はシード固定の shuffle で割り当てる。
    """

    def __init__(self, latent_dir: str, seed: int = 0) -> None:
        files = sorted(
            os.path.join(latent_dir, f)
            for f in os.listdir(latent_dir) if f.endswith('.pt')
        )
        rng = random.Random(seed)
        tgt_indices = list(range(len(files)))
        rng.shuffle(tgt_indices)
        # 自己ペアは隣に移す
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
        return (src['latent'], src['label'],
                tgt['latent'], tgt['label'])


# ------------------------------------------------------------------------------
# Quantitative evaluation
# ------------------------------------------------------------------------------

@torch.no_grad()
def evaluate_metrics(h, generator, face_pool, arcface, loader, device, max_batches=None):
    id_sims, cons_vals = [], []

    for i, (w_src, _, w_tgt, _) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

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
        sim = F.cosine_similarity(feat_gen, feat_src, dim=1)
        id_sims.append(sim.cpu())

        cons = F.l1_loss(w_sty_new, w_sty_tgt, reduction='none').mean(dim=(1, 2))
        cons_vals.append(cons.cpu())

    id_sims  = torch.cat(id_sims)
    cons_vals = torch.cat(cons_vals)
    return {
        "id_sim_mean":  id_sims.mean().item(),
        "id_sim_std":   id_sims.std().item(),
        "cons_mean":    cons_vals.mean().item(),
        "cons_std":     cons_vals.std().item(),
        "n_samples":    len(id_sims),
    }


# ------------------------------------------------------------------------------
# Qualitative visualization
# ------------------------------------------------------------------------------

@torch.no_grad()
def save_grid(h, generator, face_pool, dataset, device, out_path, n_vis=16):
    """
    n_vis 件のペアについて [src | tgt | gen] の比較グリッドを保存する。
    """
    images = []
    for i in range(min(n_vis, len(dataset))):
        w_src, label_src, w_tgt, label_tgt = dataset[i]
        w_src = w_src.unsqueeze(0).to(device)
        w_tgt = w_tgt.unsqueeze(0).to(device)

        w_sty_src = h(w_src)
        w_sty_tgt = h(w_tgt)
        w_new     = (w_src - w_sty_src) + w_sty_tgt

        img_src = decode(w_src, generator, face_pool)[0]   # [3,256,256]
        img_tgt = decode(w_tgt, generator, face_pool)[0]
        img_gen = decode(w_new, generator, face_pool)[0]

        images.extend([img_src, img_tgt, img_gen])

    # make_grid: nrow=3 で (src, tgt, gen) を1行に並べる
    grid = make_grid(
        torch.stack(images),
        nrow=3,
        normalize=True,
        value_range=(-1, 1),
    )
    pil = Image.fromarray(
        (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    )
    pil.save(out_path)
    print(f"  → Saved grid ({n_vis} pairs): {out_path}")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="AFS Style Extractor 評価")
    p.add_argument("--ckpt_path",    required=True,
                   help="best_model.pt へのパス")
    p.add_argument("--psp_path",     required=True,
                   help="pSp チェックポイント（StyleGAN2 generator 取得用）")
    p.add_argument("--arcface_path", required=True,
                   help="model_ir_se50.pth へのパス")
    p.add_argument("--latent_dir",   default="latents/test",
                   help="評価に使う潜在コードのディレクトリ")
    p.add_argument("--out_dir",      default=None,
                   help="出力ディレクトリ。省略時は ckpt_path の親ディレクトリ直下の eval/")
    p.add_argument("--n_vis",        type=int, default=16,
                   help="可視化するペア数")
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--device",       default="cuda")
    return p.parse_args()


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
    print(f"Eval samples: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=0)

    # --- Quantitative ---
    print("\n[定量評価]")
    metrics = evaluate_metrics(h, generator, face_pool, arcface, loader, device)
    print(f"  id_sim : {metrics['id_sim_mean']:.4f} ± {metrics['id_sim_std']:.4f}  "
          f"(高いほど同一人物性が保持されている)")
    print(f"  cons   : {metrics['cons_mean']:.4f} ± {metrics['cons_std']:.4f}  "
          f"(低いほどスタイル抽出が安定している)")
    print(f"  n      : {metrics['n_samples']}")

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  → Saved: {metrics_path}")

    # --- Qualitative ---
    print("\n[定性評価]")
    grid_path = os.path.join(out_dir, "comparison_grid.png")
    save_grid(h, generator, face_pool, dataset, device, grid_path, n_vis=args.n_vis)
    print("  グリッド列順: src（identity元） | tgt（expression元） | gen（転写結果）")


if __name__ == "__main__":
    main()
