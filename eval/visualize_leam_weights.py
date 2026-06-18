"""
学習済みチェックポイントからLEAM重みを取り出し、棒グラフで可視化する。
"""

import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def visualize_leam_weights(checkpoint_path: str, save_path: str = None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # state_dict から LEAM 重みを抽出
    state = checkpoint["model_state_dict"]
    raw_weights = state["leam.layer_weights"]           # (18,)
    weights = torch.sigmoid(raw_weights).numpy()

    fig, ax = plt.subplots(figsize=(12, 5))

    # グループごとに色を変える
    colors = (
        ["#e74c3c"] * 4 +    # Coarse（赤）
        ["#2ecc71"] * 8 +    # Medium（緑）
        ["#3498db"] * 6      # Fine（青）
    )
    ax.bar(range(18), weights, color=colors)

    # グループ境界線
    ax.axvline(x=3.5, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(x=11.5, color="black", linestyle="--", linewidth=0.8)

    # 凡例
    patches = [
        mpatches.Patch(color="#e74c3c", label="Coarse (layers 1-4: structure)"),
        mpatches.Patch(color="#2ecc71", label="Medium (layers 5-12: expression)"),
        mpatches.Patch(color="#3498db", label="Fine (layers 13-18: texture)"),
    ]
    ax.legend(handles=patches, loc="upper right")

    ax.set_xlabel("StyleGAN Layer Index")
    ax.set_ylabel("LEAM Weight (after sigmoid)")
    ax.set_title("LEAM: Learned Layer Importance Weights")
    ax.set_xticks(range(18))
    ax.set_xticklabels([str(i + 1) for i in range(18)])
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize LEAM weights from a checkpoint")
    parser.add_argument("checkpoint", help="学習済みチェックポイントのパス")
    parser.add_argument("--save_path", default=None, help="グラフの保存先パス（省略時は保存しない）")
    args = parser.parse_args()

    visualize_leam_weights(args.checkpoint, args.save_path)
