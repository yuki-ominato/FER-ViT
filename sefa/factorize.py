import torch
import numpy as np
from typing import Dict, List


def factorize_stylegan_weights(
    stylegan_pkl_path: str,
    layer_idx: List[int] = None,
    num_semantics: int = 10,
) -> Dict[str, np.ndarray]:
    """
    SeFaによるStyleGAN重みの因子分解。
    StyleGANの最初の全結合層（mapping network直後）の
    重み行列 A に対して固有値分解を行い、意味的方向を取り出す。

    Args:
        stylegan_pkl_path: StyleGAN2の事前学習済み重みのパス（.pkl）
        layer_idx: 因子分解する層のインデックスリスト
                   例: [0, 1] → Coarse層、[5, 11] → Medium層
                   Noneの場合は全層を使用
        num_semantics: 取り出す方向の数（固有値の大きい順）

    Returns:
        {
            "directions": np.ndarray, shape=(num_semantics, latent_dim),  # 方向ベクトル群
            "eigenvalues": np.ndarray, shape=(num_semantics,),            # 寄与度
        }
    """
    import pickle
    import sys
    sys.path.insert(0, "stylegan2-ada-pytorch")  # StyleGAN2のコードパス

    with open(stylegan_pkl_path, "rb") as f:
        G = pickle.load(f)["G_ema"].to("cpu")

    # StyleGAN2の最初のfc層の重みを取得
    # G.mapping.fc0.weight: shape (latent_dim, latent_dim)
    weight = G.mapping.fc0.weight.detach().cpu().numpy()  # (D, D)

    if layer_idx is not None:
        # 特定層のみを対象とする場合はスライス
        weight = weight[layer_idx, :]

    # A^T A の固有値分解（SeFaのアルゴリズム）
    ATA = weight.T @ weight                          # (D, D)
    eigenvalues, eigenvectors = np.linalg.eigh(ATA)

    # 固有値の大きい順に並べ替え
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 上位 num_semantics 個の固有ベクトルを方向ベクトルとして返す
    directions = eigenvectors[:, :num_semantics].T  # (K, D)

    return {
        "directions": directions,
        "eigenvalues": eigenvalues[:num_semantics],
    }
