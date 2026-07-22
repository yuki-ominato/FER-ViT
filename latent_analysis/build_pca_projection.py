"""
PCA + ラベル相関に基づいて感情部分空間の基底行列 N を構築するスクリプト。

train_svm.py + build_svm_projection.py の PCA 版。
SVM のように分類器を学習してから係数を取り出すのではなく、

    1. train latents 全体に PCA をかけて上位 n_components 個の主成分を得る
    2. 各主成分のスコア(サンプルごとの射影値)と感情ラベルとの相関を
       ANOVA F 値 (sklearn.feature_selection.f_classif) で評価する
    3. F 値が高い上位 k 個（デフォルト 7、SVM 版と揃える）を
       「感情部分空間」の基底として採用する

保存形式・射影方法は build_svm_projection.py が作る emotion_basis_N.pt と同一
（N: (9216, k)、w_emotion = N @ pinv(N) @ w、w_residual = w - w_emotion）。
そのため project_latents_svm.py / evaluate_svm_subspace.py はこの基底に対しても
そのまま使用できる。本リポジトリでは分かりやすさのため、PCA 版として
project_latents_pca.py / evaluate_pca_subspace.py という薄い専用スクリプトも用意している。

使い方:
    python latent_analysis/build_pca_projection.py \
        --latent_dir latents/train \
        --output_dir latent_analysis/pca_output
"""

import os
import sys
import argparse
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EMOTION_NAMES = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
    4: 'neutral', 5: 'sad', 6: 'surprise',
}
SEQ_LEN = 18
LATENT_DIM = 512
FLAT_DIM = SEQ_LEN * LATENT_DIM   # 9216


def load_latents(latent_dir: str):
    """ディレクトリ内の .pt ファイルを全件ロードして numpy 配列で返す"""
    files = sorted([f for f in os.listdir(latent_dir) if f.endswith('.pt')])
    if not files:
        raise ValueError(f"No .pt files found in {latent_dir}")

    all_w, all_labels = [], []
    print(f"Loading {len(files)} latent files from {latent_dir} ...")
    for fname in tqdm(files):
        data = torch.load(os.path.join(latent_dir, fname),
                          map_location='cpu', weights_only=True)
        all_w.append(data['latent'].numpy())   # (18, 512)
        all_labels.append(int(data['label']))

    X = np.stack(all_w, axis=0)            # (N, 18, 512)
    n = X.shape[0]
    X = X.reshape(n, -1)                   # (N, 9216)
    y = np.array(all_labels)               # (N,)
    print(f"  X={X.shape}, y={y.shape}")
    return X, y


def build_N(X: np.ndarray, y: np.ndarray, n_components: int, k: int):
    """
    X に PCA をかけ、y との ANOVA F 値が高い上位 k 個の主成分を
    感情部分空間の基底 N として返す。

    Returns
    -------
    N    : (FLAT_DIM, k) float32  各列が選ばれた主成分方向（PCA成分なのですでに単位ベクトル）
    info : 診断用メタデータの dict（選ばれた主成分の元インデックス、F値、p値、寄与率）
    """
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X)   # (N_samples, n_components)

    f_scores, p_values = f_classif(coords, y)   # (n_components,)
    order = np.argsort(f_scores)[::-1]          # F値降順
    top_k = order[:k]

    N = pca.components_[top_k].T.astype(np.float32)   # (FLAT_DIM, k)

    info = {
        'selected_components': top_k.tolist(),
        'f_scores': f_scores[top_k].tolist(),
        'p_values': p_values[top_k].tolist(),
        'explained_variance_ratio': pca.explained_variance_ratio_[top_k].tolist(),
        'n_components_searched': n_components,
    }
    return N, info


def main():
    parser = argparse.ArgumentParser(
        description="Build emotion subspace basis N via PCA + label-correlation selection"
    )
    parser.add_argument('--latent_dir', type=str, required=True,
                        help='Directory containing train .pt latent files')
    parser.add_argument('--output_dir', type=str,
                        default='latent_analysis/pca_output',
                        help='Directory to save the basis N')
    parser.add_argument('--n_components', type=int, default=100,
                        help='PCA で計算する主成分の探索プール数（この中から上位 k 個を選ぶ）')
    parser.add_argument('--k', type=int, default=7,
                        help='感情部分空間として採用する主成分の数（デフォルト: SVM版と同じ7）')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    X, y = load_latents(args.latent_dir)

    print(f"\nFitting PCA (n_components={args.n_components}) and ranking components by "
          f"ANOVA F-value against emotion labels ...")
    N, info = build_N(X, y, args.n_components, args.k)
    print(f"N.shape: {N.shape}")

    print(f"\nSelected principal components (by F-score, descending):")
    print(f"  {'PC index':>10}  {'F-score':>12}  {'p-value':>12}  {'expl. var.':>12}")
    for idx, f, p, ev in zip(info['selected_components'], info['f_scores'],
                              info['p_values'], info['explained_variance_ratio']):
        print(f"  {idx:>10}  {f:>12.2f}  {p:>12.2e}  {ev:>12.4f}")

    # 直交性の確認（PCA成分同士なので理論上はほぼ単位行列になるはず）
    gram = N.T @ N
    print("\nGram matrix N^T N (ideally close to identity):")
    for i in range(N.shape[1]):
        row = '  '.join(f"{gram[i, j]:+.3f}" for j in range(N.shape[1]))
        print(f"  [{i}]  {row}")

    out_path = os.path.join(args.output_dir, 'emotion_basis_pca.pt')
    torch.save({
        'N': torch.tensor(N, dtype=torch.float32),   # (9216, k)
        'emotion_names': EMOTION_NAMES,
        'seq_len': SEQ_LEN,
        'latent_dim': LATENT_DIM,
        **info,
    }, out_path)
    print(f"\nSaved PCA emotion basis N -> {out_path}")


if __name__ == '__main__':
    main()
