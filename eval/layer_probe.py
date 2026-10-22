"""
W+ の各層（18層）に線形分類器（ロジスティック回帰）を当て、層ごとの表情情報量を測る。

analyze_layer_importance.py は「学習済みモデルがどの層に頼っているか」を測るが、
複数の層が同じ情報を持つ場合は各層の重要度が低く出る。こちらは各層単体で
どれだけ表情を当てられるかを直接測るので、ViT の学習なしで数分で済む。

層単体のほか、層グループ（粗い 0-3 / 中間 4-11 / 細かい 12-17 / 全層）を連結した場合も評価する。
層番号は 0 始まり。

学習は holdout 分割の train 部分、評価は holdout 部分で行い、test は使わない:
    python eval/layer_probe.py --latent_dir latents/raf-db/train \
        --holdout_split splits/raf-db_holdout10_seed0.json \
        --output_dir eval_results/layer_probe/raf-db_psp

データを減らした条件（例: 10%）も --data_fraction で試せる（クラス比を保って train 部分を削減）。
"""

import os
import sys
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.latent_dataset import LatentFERDataset
from data.holdout_split import load_split
from eval.analyze_layer_importance import LAYER_GROUP_COLORS


LAYER_GROUPS = {
    'coarse_0-3': list(range(0, 4)),
    'medium_4-11': list(range(4, 12)),
    'fine_12-17': list(range(12, 18)),
    'all_0-17': list(range(0, 18)),
}


def load_all(latent_dir: str, files: list):
    dataset = LatentFERDataset(latent_dir, files=files)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=4)
    latents, labels = next(iter(loader))
    return latents.float().numpy(), labels.numpy()


def subsample(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """クラス比を保って fraction だけ残すインデックス"""
    if fraction >= 1.0:
        return np.arange(len(labels))
    rng = np.random.RandomState(seed)
    keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        n = max(1, int(len(idx) * fraction))
        keep.extend(rng.choice(idx, n, replace=False))
    return np.sort(np.array(keep))


def fit_probe(x_train, y_train, x_val, y_val, C: float, seed: int) -> dict:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, class_weight='balanced', random_state=seed),
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_val)
    return {
        'accuracy': float(accuracy_score(y_val, pred)),
        'f1_macro': float(f1_score(y_val, pred, average='macro')),
    }


def plot(results: dict, save_path: str):
    layer = results['per_layer']
    seq_len = len(layer)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(range(seq_len), [r['accuracy'] for r in layer], color=LAYER_GROUP_COLORS[:seq_len])
    ax.axhline(results['chance_accuracy'], color='gray', linestyle='--', linewidth=1,
               label=f"majority class ({results['chance_accuracy']:.3f})")
    ax.axhline(results['groups']['all_0-17']['accuracy'], color='black', linewidth=1,
               label=f"all layers ({results['groups']['all_0-17']['accuracy']:.3f})")
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels([str(i) for i in range(seq_len)])
    ax.set_xlabel('W+ layer index (0-based)')
    ax.set_ylabel('Linear probe accuracy (holdout)')
    ax.set_title(f"Per-layer linear probe (train fraction={results['data_fraction']})")
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="W+ の層ごとの線形分類で表情情報量を測る")
    parser.add_argument("--latent_dir", required=True, help="train latent のディレクトリ")
    parser.add_argument("--holdout_split", required=True, help="data/holdout_split.py の分割 JSON")
    parser.add_argument("--output_dir", default="eval_results/layer_probe")
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="train 部分のうち学習に使う割合（クラス比を保って削減）")
    parser.add_argument("--C", type=float, default=0.1, help="ロジスティック回帰の正則化の逆数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    split = load_split(args.holdout_split)
    x_train, y_train = load_all(args.latent_dir, split['train_files'])
    x_val, y_val = load_all(args.latent_dir, split['val_files'])

    keep = subsample(y_train, args.data_fraction, args.seed)
    x_train, y_train = x_train[keep], y_train[keep]
    print(f"train={len(y_train)} holdout={len(y_val)} layers={x_train.shape[1]}")

    per_layer = []
    for layer in range(x_train.shape[1]):
        r = fit_probe(x_train[:, layer], y_train, x_val[:, layer], y_val, args.C, args.seed)
        per_layer.append(r)
        print(f"layer {layer:2d}: acc={r['accuracy']:.4f} f1={r['f1_macro']:.4f}")

    groups = {}
    for name, layers in LAYER_GROUPS.items():
        layers = [l for l in layers if l < x_train.shape[1]]
        r = fit_probe(x_train[:, layers].reshape(len(y_train), -1), y_train,
                      x_val[:, layers].reshape(len(y_val), -1), y_val, args.C, args.seed)
        groups[name] = r
        print(f"{name:>12s}: acc={r['accuracy']:.4f} f1={r['f1_macro']:.4f}")

    results = {
        'latent_dir': args.latent_dir,
        'holdout_split': os.path.abspath(args.holdout_split),
        'data_fraction': args.data_fraction,
        'n_train': int(len(y_train)),
        'n_holdout': int(len(y_val)),
        'C': args.C,
        'seed': args.seed,
        'chance_accuracy': float(np.bincount(y_val).max() / len(y_val)),
        'per_layer': per_layer,
        'groups': groups,
    }
    tag = f"frac{int(round(args.data_fraction * 100))}"
    with open(os.path.join(args.output_dir, f"layer_probe_{tag}.json"), 'w') as f:
        json.dump(results, f, indent=2)
    plot(results, os.path.join(args.output_dir, f"layer_probe_{tag}.png"))


if __name__ == "__main__":
    main()
