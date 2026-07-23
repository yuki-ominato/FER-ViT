"""
ランダム部分空間の基底行列 N を構築して保存するスクリプト（SVM版・PCA版の対照実験用）。

build_svm_projection.py / build_pca_projection.py と同一フォーマットの emotion_basis を
出力するので、後段の project_latents_svm.py（=project_latents_pca.py と同一）と
evaluate_svm_subspace.py（=evaluate_pca_subspace.py と同一）にそのまま渡せる。

基底の作り方だけがSVM/PCAと異なる:
    - ラベルもデータ分散も一切使わず、乱数で張った k 次元部分空間を使う。
    - (9216, k) のガウス乱数行列を QR 分解して正規直交基底 Q を取る（PCA基底と同じく直交）。
    - --seed で再現可能。

これにより「SVM/PCAで選んだ k 次元部分空間が、同じ次元のランダム部分空間より
どれだけ感情情報を集約できているか」を、射影・評価を完全に共通化した上で比較できる。

使い方:
    python latent_analysis/build_random_projection.py \
        --output_dir latent_analysis/random_output_fer2013 \
        --k 7 --seed 42
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EMOTION_NAMES = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
    4: 'neutral', 5: 'sad', 6: 'surprise',
}
SEQ_LEN = 18
LATENT_DIM = 512
FLAT_DIM = SEQ_LEN * LATENT_DIM   # 9216


def build_random_N(k: int, seed: int, dim: int = FLAT_DIM) -> np.ndarray:
    """
    (dim, k) のガウス乱数を QR 分解し、正規直交な k 次元部分空間基底を返す。

    Returns
    -------
    N : (dim, k) float32   各列が正規直交（PCA基底と同じ直交条件）
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, k)).astype(np.float64)
    Q, _ = np.linalg.qr(A)            # Q: (dim, k), 列は正規直交
    return Q[:, :k].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Build a random orthonormal subspace basis N (control for SVM/PCA)"
    )
    parser.add_argument('--output_dir', type=str,
                        default='latent_analysis/random_output',
                        help='Directory to save the basis N')
    parser.add_argument('--k', type=int, default=7,
                        help='部分空間の次元（SVM/PCA版と揃える, デフォルト7）')
    parser.add_argument('--seed', type=int, default=42,
                        help='再現用の乱数シード（デフォルト42）')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Building random orthonormal basis: dim={FLAT_DIM}, k={args.k}, seed={args.seed}")
    N = build_random_N(args.k, args.seed)
    print(f"N.shape: {N.shape}")   # (9216, k)

    # 直交性の確認（QR 由来なので単位行列になるはず）
    gram = N.T @ N
    print("\nGram matrix N^T N (ideally identity):")
    for i in range(N.shape[1]):
        row = '  '.join(f"{gram[i, j]:+.3f}" for j in range(N.shape[1]))
        print(f"  [{i}]  {row}")

    out_path = os.path.join(args.output_dir, 'emotion_basis_random.pt')
    torch.save({
        'N': torch.tensor(N, dtype=torch.float32),   # (9216, k)
        'emotion_names': EMOTION_NAMES,
        'seq_len': SEQ_LEN,
        'latent_dim': LATENT_DIM,
        'seed': args.seed,
        'k': args.k,
    }, out_path)
    print(f"\nSaved random subspace basis N -> {out_path}")


if __name__ == '__main__':
    main()
