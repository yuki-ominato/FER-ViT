"""
学習済み LinearSVC から感情部分空間の基底行列 N を構築して保存するスクリプト。

clf.coef_.T を N (shape: 9216×7) として取り出し、各列を L2 正規化する。
射影行列 P = N(N^T N)^-1 N^T は大きいため保存せず、
project_latents_svm.py で右から順に計算する:
    w_emotion = N @ (pinv(N) @ w)   (メモリ効率版)

使い方:
    python latent_analysis/build_svm_projection.py \
        --svm_model latent_analysis/svm_output/svm_model.joblib \
        --output_dir latent_analysis/svm_output
"""

import os
import sys
import argparse
import numpy as np
import torch
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EMOTION_NAMES = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
    4: 'neutral', 5: 'sad', 6: 'surprise',
}
SEQ_LEN = 18
LATENT_DIM = 512
FLAT_DIM = SEQ_LEN * LATENT_DIM   # 9216


def build_N(clf) -> np.ndarray:
    """
    SVM の係数行列から感情方向基底 N を構築する。

    clf.coef_ : (7, 9216)
    N = clf.coef_.T : (9216, 7)  各列が各感情クラスの方向ベクトル

    各列を L2 正規化して返す。
    """
    coef = clf.coef_                # (7, 9216)
    assert coef.shape == (7, FLAT_DIM), \
        f"Unexpected coef shape: {coef.shape}, expected (7, {FLAT_DIM})"

    N = coef.T                      # (9216, 7)

    # 各列を L2 正規化
    norms = np.linalg.norm(N, axis=0, keepdims=True) + 1e-12   # (1, 7)
    N = N / norms

    return N.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Build emotion subspace basis N from trained LinearSVC"
    )
    parser.add_argument('--svm_model', type=str, required=True,
                        help='Path to svm_model.joblib')
    parser.add_argument('--output_dir', type=str,
                        default='latent_analysis/svm_output',
                        help='Directory to save the basis N')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading SVM model from {args.svm_model} ...")
    clf = joblib.load(args.svm_model)

    N = build_N(clf)
    print(f"N.shape: {N.shape}")   # (9216, 7)

    # 列ベクトル間のコサイン類似度（直交性の確認）
    gram = N.T @ N   # (7, 7)
    print("\nGram matrix N^T N (ideally close to identity):")
    for i in range(7):
        row = '  '.join(f"{gram[i, j]:+.3f}" for j in range(7))
        print(f"  [{EMOTION_NAMES[i]:>8}]  {row}")

    out_path = os.path.join(args.output_dir, 'emotion_basis_N.pt')
    torch.save({
        'N': torch.tensor(N, dtype=torch.float32),   # (9216, 7)
        'emotion_names': EMOTION_NAMES,
        'seq_len': SEQ_LEN,
        'latent_dim': LATENT_DIM,
    }, out_path)
    print(f"\nSaved emotion basis N -> {out_path}")


if __name__ == '__main__':
    main()
