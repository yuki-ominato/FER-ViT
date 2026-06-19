"""
感情基底 N を使って W+ 潜在コードを感情成分と非感情成分に射影するスクリプト。

射影の計算（メモリ効率版）:
    N      : (9216, 7)   感情部分空間の基底（列 = 各感情方向）
    pinv_N : (7, 9216)   N の疑似逆行列

    w          : (9216,)   元の潜在コード (flatten)
    w_emotion  = N @ (pinv_N @ w)   感情成分
    w_residual = w - w_emotion       非感情成分

出力 .pt ファイルのフォーマット:
    {
        "latent"         : Tensor (18, 512)   元の潜在コード
        "emotion_latent" : Tensor (18, 512)   感情成分
        "residual_latent": Tensor (18, 512)   非感情成分
        "label"          : int
    }

使い方:
    # 全スプリットを一括処理
    python latent_analysis/project_latents_svm.py \
        --basis latent_analysis/svm_output/emotion_basis_N.pt \
        --latent_root latents \
        --output_root latents_svm \
        --splits train val test
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEQ_LEN = 18
LATENT_DIM = 512
FLAT_DIM = SEQ_LEN * LATENT_DIM   # 9216


def load_basis(basis_path: str):
    """emotion_basis_N.pt から N と pinv(N) を返す"""
    data = torch.load(basis_path, map_location='cpu', weights_only=True)
    N = data['N'].numpy()   # (9216, 7)  float32

    # 疑似逆行列を事前計算しておく（射影時に再利用）
    pinv_N = np.linalg.pinv(N)   # (7, 9216)

    print(f"Loaded basis N: {N.shape}")
    print(f"Computed pinv(N): {pinv_N.shape}")
    return N, pinv_N


def project_single(w_flat: np.ndarray, N: np.ndarray, pinv_N: np.ndarray):
    """
    w_flat : (9216,)
    returns: w_emotion (9216,), w_residual (9216,)
    """
    coords = pinv_N @ w_flat       # (7,)   感情部分空間上の座標
    w_emotion = N @ coords         # (9216,)
    w_residual = w_flat - w_emotion
    return w_emotion, w_residual


def process_split(
    split: str,
    latent_dir: str,
    output_dir: str,
    N: np.ndarray,
    pinv_N: np.ndarray,
) -> None:
    files = sorted([f for f in os.listdir(latent_dir) if f.endswith('.pt')])
    if not files:
        raise ValueError(f"No .pt files found in {latent_dir}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[{split}] Processing {len(files)} files -> {output_dir}")

    for fname in tqdm(files):
        src = os.path.join(latent_dir, fname)
        data = torch.load(src, map_location='cpu', weights_only=True)

        latent = data['latent']          # Tensor (18, 512)
        label = data['label']

        w_flat = latent.numpy().reshape(-1).astype(np.float32)   # (9216,)
        w_emotion, w_residual = project_single(w_flat, N, pinv_N)

        out = {
            'latent': latent,
            'emotion_latent': torch.tensor(
                w_emotion.reshape(SEQ_LEN, LATENT_DIM), dtype=torch.float32
            ),
            'residual_latent': torch.tensor(
                w_residual.reshape(SEQ_LEN, LATENT_DIM), dtype=torch.float32
            ),
            'label': label,
        }
        torch.save(out, os.path.join(output_dir, fname))


def main():
    parser = argparse.ArgumentParser(
        description="Project W+ latents into emotion / residual subspaces via SVM basis"
    )
    parser.add_argument('--basis', type=str, required=True,
                        help='Path to emotion_basis_N.pt')
    parser.add_argument('--latent_root', type=str, default='latents',
                        help='Root directory containing split subdirectories')
    parser.add_argument('--output_root', type=str, default='latents_svm',
                        help='Root directory for output split subdirectories')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='Split names to process')
    args = parser.parse_args()

    N, pinv_N = load_basis(args.basis)

    for split in args.splits:
        latent_dir = os.path.join(args.latent_root, split)
        output_dir = os.path.join(args.output_root, split)

        if not os.path.isdir(latent_dir):
            print(f"[WARN] {latent_dir} not found, skipping {split}")
            continue

        process_split(split, latent_dir, output_dir, N, pinv_N)

    print("\nDone.")


if __name__ == '__main__':
    main()
