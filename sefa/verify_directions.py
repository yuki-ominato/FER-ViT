import torch
import numpy as np
from typing import List


def verify_non_expression_directions(
    directions: np.ndarray,
    sample_latents: torch.Tensor,
    fer_model,
    stylegan_generator,
    step_sizes: List[float] = [-3.0, -1.5, 0.0, 1.5, 3.0],
    device: str = "cuda",
) -> dict:
    """
    発見した方向ベクトルに沿って潜在コードを移動させた際に、
    FERモデルの予測ラベルが変化しないかを確認する。

    ラベルが変化しない方向 = 「非表情方向」として採用可能。
    ラベルが変化する方向   = 表情成分が含まれるため採用不可。

    Args:
        directions: shape=(K, latent_dim), SeFaで発見した方向群
        sample_latents: shape=(N, 18, 512), 検証用の潜在コードサンプル
        fer_model: 学習済みのLatentViTモデル
        stylegan_generator: StyleGAN2のジェネレータ（画像再生成用）
        step_sizes: 移動量のリスト（0.0が元の潜在コード）
        device: 使用デバイス

    Returns:
        {
            "direction_idx": int,
            "label_change_rate": float,  # ラベルが変化した割合（低いほど非表情方向）
        } のリスト
    """
    fer_model.eval()
    results = []

    for d_idx, direction in enumerate(directions):
        dir_tensor = torch.tensor(
            direction, dtype=torch.float32, device=device
        )  # (latent_dim,)

        label_changes = []

        for w in sample_latents[:50]:  # 計算コスト削減のため50サンプル
            w = w.to(device)  # (18, 512)

            # 元の予測ラベル
            with torch.no_grad():
                original_pred = fer_model(w.unsqueeze(0)).argmax(dim=1).item()

            changed = False
            for step in step_sizes:
                if step == 0.0:
                    continue
                # 方向に沿って移動
                w_perturbed = w.clone()
                w_perturbed += step * dir_tensor.unsqueeze(0)  # (18, 512) にブロードキャスト

                with torch.no_grad():
                    perturbed_pred = fer_model(
                        w_perturbed.unsqueeze(0)
                    ).argmax(dim=1).item()

                if perturbed_pred != original_pred:
                    changed = True
                    break

            label_changes.append(changed)

        change_rate = np.mean(label_changes)
        results.append({
            "direction_idx": d_idx,
            "label_change_rate": change_rate,
        })
        print(f"Direction {d_idx:02d}: label change rate = {change_rate:.3f}")

    return results
