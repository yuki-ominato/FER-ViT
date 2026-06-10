"""
InterFaceGAN の手法を FER2013 の w+ 潜在空間に適用し、
表情方向ベクトルを LinearSVC で計算して保存するスクリプト。

理論的背景 (Shen et al., CVPR 2020):
    GAN の潜在空間は線形分離可能であり、二クラス LinearSVM の
    決定境界の法線ベクトルが各意味的属性（表情など）の方向に対応する。

    n_expr を SVM の係数ベクトルとすると:
        w_expr = (w · n_expr) * n_expr   ← 表情成分
        w_id   = w - w_expr              ← アイデンティティ成分

使い方:
    python latent_analysis/compute_expression_directions.py \\
        --latent_dir /path/to/latents/train \\
        --output_dir ./latent_analysis/directions \\
        --method both
"""

import os
import sys
import argparse
import numpy as np
import torch
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EMOTION_NAMES = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
    4: 'neutral', 5: 'sad', 6: 'surprise',
}
NUM_CLASSES = 7


def load_all_latents(latent_dir: str):
    """潜在コードファイルを全件ロードして numpy 配列として返す"""
    files = sorted([f for f in os.listdir(latent_dir) if f.endswith('.pt')])
    if not files:
        raise ValueError(f"No .pt files found in {latent_dir}")

    all_w, all_labels = [], []
    print(f"Loading {len(files)} latent files from {latent_dir} ...")
    for fname in tqdm(files):
        data = torch.load(os.path.join(latent_dir, fname),
                          map_location='cpu', weights_only=True)
        all_w.append(data['latent'].numpy())  # (18, 512)
        all_labels.append(int(data['label']))

    all_w = np.stack(all_w, axis=0)      # (N, 18, 512)
    all_labels = np.array(all_labels)     # (N,)
    print(f"Loaded: all_w={all_w.shape}, labels={all_labels.shape}")
    return all_w, all_labels


def compute_binary_directions(all_w_flat: np.ndarray, all_labels: np.ndarray):
    """
    各クラス vs 残り全クラス (one-vs-rest) で LinearSVC を訓練し
    表情方向ベクトルを返す。

    InterFaceGAN と同じ手順: 二値 SVM の係数ベクトルが属性方向に対応。

    Returns:
        directions : dict {class_id: ndarray (18*512,)} 正規化済み
        svms       : dict {class_id: LinearSVC}
    """
    directions, svms = {}, {}
    for cls_id in range(NUM_CLASSES):
        binary_labels = (all_labels == cls_id).astype(int)
        pos = binary_labels.sum()
        neg = len(binary_labels) - pos
        print(f"\n  [{EMOTION_NAMES[cls_id]}] pos={pos}, neg={neg}")

        svm = LinearSVC(max_iter=10000, C=0.1, class_weight='balanced')
        svm.fit(all_w_flat, binary_labels)

        acc = accuracy_score(binary_labels, svm.predict(all_w_flat))
        print(f"    train accuracy: {acc:.4f}")

        n = svm.coef_[0]                       # (18*512,)
        n = n / (np.linalg.norm(n) + 1e-12)    # L2 正規化
        directions[cls_id] = n
        svms[cls_id] = svm

    return directions, svms


def compute_multiclass_directions(all_w_flat: np.ndarray, all_labels: np.ndarray):
    """
    7 クラス LinearSVC (OvR) を訓練し、各クラスの係数ベクトルを方向として返す。

    Returns:
        directions : dict {class_id: ndarray (18*512,)} 正規化済み
        svm        : LinearSVC
    """
    print("\n  Training 7-class LinearSVC (OvR) ...")
    svm = LinearSVC(max_iter=10000, C=0.1, class_weight='balanced',
                    multi_class='ovr')
    svm.fit(all_w_flat, all_labels)

    acc = accuracy_score(all_labels, svm.predict(all_w_flat))
    print(f"  7-class train accuracy: {acc:.4f}")
    print(classification_report(
        all_labels, svm.predict(all_w_flat),
        target_names=list(EMOTION_NAMES.values())
    ))

    directions = {}
    for cls_id in range(NUM_CLASSES):
        n = svm.coef_[cls_id]                  # (18*512,)
        n = n / (np.linalg.norm(n) + 1e-12)
        directions[cls_id] = n

    return directions, svm


def save_directions(
    directions: dict,
    output_dir: str,
    prefix: str,
    seq_len: int = 18,
    latent_dim: int = 512,
) -> str:
    """方向ベクトルを .pt ファイルとして保存"""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{prefix}_directions.pt")

    save_dict = {
        'directions': {
            cls_id: torch.tensor(n, dtype=torch.float32).reshape(seq_len, latent_dim)
            for cls_id, n in directions.items()
        },
        'emotion_names': EMOTION_NAMES,
        'seq_len': seq_len,
        'latent_dim': latent_dim,
        'method': prefix,
    }
    torch.save(save_dict, out_path)
    print(f"\nSaved {prefix} directions -> {out_path}")
    return out_path


def print_class_distribution(all_labels: np.ndarray) -> None:
    N = len(all_labels)
    print("\nClass distribution:")
    for cls_id in range(NUM_CLASSES):
        count = (all_labels == cls_id).sum()
        print(f"  {EMOTION_NAMES[cls_id]:>8}: {count:5d} ({count/N*100:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Compute expression direction vectors via InterFaceGAN-style LinearSVC"
    )
    parser.add_argument('--latent_dir', type=str, required=True,
                        help='Directory containing .pt latent files (train split)')
    parser.add_argument('--output_dir', type=str,
                        default='./latent_analysis/directions',
                        help='Output directory for direction vectors')
    parser.add_argument('--method', type=str, default='both',
                        choices=['binary', 'multiclass', 'both'],
                        help='binary: per-class OvR SVM | multiclass: single 7-class SVM')
    parser.add_argument('--seq_len', type=int, default=18)
    parser.add_argument('--latent_dim', type=int, default=512)
    args = parser.parse_args()

    all_w, all_labels = load_all_latents(args.latent_dir)
    N = len(all_w)
    all_w_flat = all_w.reshape(N, -1)  # (N, 18*512)

    print_class_distribution(all_labels)

    if args.method in ('binary', 'both'):
        print("\n" + "=" * 60)
        print("Computing BINARY direction vectors (one-vs-rest) ...")
        print("=" * 60)
        dirs, _ = compute_binary_directions(all_w_flat, all_labels)
        save_directions(dirs, args.output_dir, 'binary', args.seq_len, args.latent_dim)

    if args.method in ('multiclass', 'both'):
        print("\n" + "=" * 60)
        print("Computing MULTICLASS direction vectors (7-class SVM) ...")
        print("=" * 60)
        dirs, _ = compute_multiclass_directions(all_w_flat, all_labels)
        save_directions(dirs, args.output_dir, 'multiclass', args.seq_len, args.latent_dim)

    print("\nDone!")


if __name__ == '__main__':
    main()