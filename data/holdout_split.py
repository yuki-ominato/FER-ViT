"""
train latent からクラス比を保って検証用（holdout）を切り出し、ファイル名の分割を JSON に保存する。

RAF-DB には val split が無く、これまで test を best 選択・LR 調整・層重要度分析に使っていた。
この分割を使えば、モデル選択や「どの層を使うか」の判断を test を見ずに行える。

分割はファイル名だけで持つので、同じ画像から作った latent（pSp / e4e / AFS 抽出後）には
同じ分割をそのまま適用できる（ファイル名は generate_latents.py / extract_style_latents.py で共通）。

使い方:
    python data/holdout_split.py --latent_dir latents/raf-db/train \
        --out splits/raf-db_holdout10_seed0.json --val_fraction 0.1 --seed 0
"""

import os
import sys
import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from afs.provenance import now


def load_split(path: str) -> dict:
    """holdout_split.py が保存した分割を読み込む"""
    with open(path) as f:
        return json.load(f)


def main(args):
    files = sorted(f for f in os.listdir(args.latent_dir) if f.endswith('.pt'))
    if not files:
        raise ValueError(f"No .pt files found in {args.latent_dir}")

    by_class = defaultdict(list)
    for fname in tqdm(files, desc='reading labels'):
        data = torch.load(os.path.join(args.latent_dir, fname), map_location='cpu', weights_only=False)
        by_class[int(data['label'])].append(fname)

    rng = np.random.RandomState(args.seed)
    train_files, val_files = [], []
    print(f"{'class':>5s} {'total':>6s} {'train':>6s} {'val':>5s}")
    for label in sorted(by_class):
        names = by_class[label]
        n_val = max(1, int(round(len(names) * args.val_fraction)))
        perm = rng.permutation(len(names))
        val_files.extend(names[i] for i in perm[:n_val])
        train_files.extend(names[i] for i in perm[n_val:])
        print(f"{label:>5d} {len(names):>6d} {len(names) - n_val:>6d} {n_val:>5d}")
    print(f"{'total':>5s} {len(files):>6d} {len(train_files):>6d} {len(val_files):>5d}")

    split = {
        'source_latent_dir': os.path.abspath(args.latent_dir),
        'val_fraction': args.val_fraction,
        'seed': args.seed,
        'created_at': now(),
        'train_files': sorted(train_files),
        'val_files': sorted(val_files),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(split, f, indent=1)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train latent からクラス層化の holdout 検証分割を作る")
    parser.add_argument("--latent_dir", required=True, help="分割する train latent のディレクトリ")
    parser.add_argument("--out", required=True, help="分割 JSON の保存先（例: splits/raf-db_holdout10_seed0.json）")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0,
                        help="分割の seed（学習の seed とは独立に固定しておく）")
    main(parser.parse_args())
