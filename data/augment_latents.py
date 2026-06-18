import os
import torch
import numpy as np
from tqdm import tqdm
from typing import List


def augment_latents_with_directions(
    latent_dir: str,
    output_dir: str,
    directions: np.ndarray,
    direction_indices: List[int],
    step_sizes: List[float] = [-2.0, -1.0, 1.0, 2.0],
):
    """
    既存の潜在コード（.ptファイル）に対して、
    指定した非表情方向に沿って摂動を加えた拡張サンプルを生成する。

    生成される拡張サンプル数：
        元サンプル数 × len(direction_indices) × len(step_sizes)

    Args:
        latent_dir: 元の潜在コードが格納されたディレクトリ
        output_dir: 拡張済み潜在コードの出力先
        directions: shape=(K, latent_dim), 全方向ベクトル
        direction_indices: 採用する方向のインデックスリスト
                           （verify_directionsで低change_rateのものを選ぶ）
        step_sizes: 各方向への移動量リスト
    """
    os.makedirs(output_dir, exist_ok=True)

    # 採用する方向ベクトルを抽出
    selected_dirs = [
        torch.tensor(directions[i], dtype=torch.float32)
        for i in direction_indices
    ]

    pt_files = [f for f in os.listdir(latent_dir) if f.endswith(".pt")]

    for fname in tqdm(pt_files, desc="Augmenting latents"):
        data = torch.load(
            os.path.join(latent_dir, fname), weights_only=False
        )
        w = data["latent"]    # (18, 512)
        label = data["label"]

        # 元のサンプルをそのまま出力先にコピー
        out_path = os.path.join(output_dir, fname)
        if not os.path.exists(out_path):
            torch.save(data, out_path)

        # 各方向・各移動量で拡張
        base = os.path.splitext(fname)[0]
        for d_i, direction in zip(direction_indices, selected_dirs):
            for step in step_sizes:
                w_aug = w + step * direction.unsqueeze(0)   # (18, 512)

                aug_fname = f"{base}_dir{d_i}_step{step:.1f}.pt"
                aug_path = os.path.join(output_dir, aug_fname)

                if not os.path.exists(aug_path):
                    torch.save(
                        {
                            "latent": w_aug,
                            "label": label,
                            "augmented": True,
                            "direction_idx": d_i,
                            "step": step,
                        },
                        aug_path,
                    )

    original_count = len(pt_files)
    aug_count = original_count * len(direction_indices) * len(step_sizes)
    total = original_count + aug_count
    print(f"\n完了: 元サンプル {original_count} + 拡張 {aug_count} = 合計 {total} サンプル")
    print(f"出力先: {output_dir}")
