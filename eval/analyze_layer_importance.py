"""
w+ の18層を1層ずつアブレーション（ゼロ化 or 平均値化）し、
精度・正解クラス確信度の低下量から学習済みモデルが重要視している層を推定するスクリプト。

visualize_leam_weights.py（LEAM前提）や evaluate_model.py の visualize_attention()
（CLSトークンとのコサイン類似度、Transformerの表現崩壊で無情報になりやすい）とは異なり、
モデル内部構造に依存せず「その層を潰すと実際に予測がどれだけ悪化するか」を直接測る。
LatentViT / LatentViTv2 / HybridLatentViT のいずれの checkpoint にも対応する
（load_model を eval.evaluate_model から流用）。
"""

import os
import sys
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.latent_dataset import LatentFERDataset
from eval.evaluate_model import load_model


# LEAM可視化と同じ層グルーピング（0-3: 構造, 4-11: 表情, 12-17: テクスチャ）
LAYER_GROUP_COLORS = (
    ["#e74c3c"] * 4 +
    ["#2ecc71"] * 8 +
    ["#3498db"] * 6
)


@torch.no_grad()
def _predict(model, latents: torch.Tensor, labels: torch.Tensor, device, batch_size: int):
    """(N, L, D) の潜在テンソルをバッチ推論し、精度と正解クラス確信度の平均を返す"""
    n = latents.size(0)
    correct = 0
    true_class_prob_sum = 0.0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = latents[start:end].to(device)
        batch_labels = labels[start:end].to(device)

        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        correct += (preds == batch_labels).sum().item()
        true_class_prob_sum += probs.gather(1, batch_labels.unsqueeze(1)).sum().item()

    return correct / n, true_class_prob_sum / n


def analyze_layer_importance(model, latents, labels, device, batch_size, ablation):
    seq_len = latents.size(1)

    baseline_acc, baseline_conf = _predict(model, latents, labels, device, batch_size)

    if ablation == "mean":
        # 層ごとの全データ平均で置換（ゼロ化よりも分布外になりにくい）
        ablation_values = latents.mean(dim=0)  # (L, D)
    else:
        ablation_values = torch.zeros(seq_len, latents.size(2))

    layer_acc, layer_conf = [], []
    for layer_idx in range(seq_len):
        ablated = latents.clone()
        ablated[:, layer_idx, :] = ablation_values[layer_idx]

        acc, conf = _predict(model, ablated, labels, device, batch_size)
        layer_acc.append(acc)
        layer_conf.append(conf)

        print(
            f"Layer {layer_idx + 1:2d}/{seq_len}: "
            f"acc={acc:.4f} (drop={baseline_acc - acc:+.4f}), "
            f"true-class prob={conf:.4f} (drop={baseline_conf - conf:+.4f})"
        )

    return {
        "baseline_accuracy": baseline_acc,
        "baseline_true_class_prob": baseline_conf,
        "layer_accuracy": layer_acc,
        "layer_true_class_prob": layer_conf,
        "layer_accuracy_drop": [baseline_acc - a for a in layer_acc],
        "layer_true_class_prob_drop": [baseline_conf - c for c in layer_conf],
    }


def plot_layer_importance(results: dict, save_path: str):
    seq_len = len(results["layer_accuracy_drop"])
    colors = LAYER_GROUP_COLORS[:seq_len]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.bar(range(seq_len), results["layer_accuracy_drop"], color=colors)
    ax1.axhline(y=0, color="black", linewidth=0.8)
    ax1.set_ylabel("Accuracy Drop\n(baseline - ablated)")
    ax1.set_title("Layer Importance via Ablation (higher drop = more important)")
    ax1.grid(axis="y", alpha=0.3)

    patches = [
        mpatches.Patch(color="#e74c3c", label="Coarse (layers 1-4: structure)"),
        mpatches.Patch(color="#2ecc71", label="Medium (layers 5-12: expression)"),
        mpatches.Patch(color="#3498db", label="Fine (layers 13-18: texture)"),
    ]
    ax1.legend(handles=patches, loc="upper right")

    ax2.bar(range(seq_len), results["layer_true_class_prob_drop"], color=colors)
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.set_xlabel("StyleGAN Layer Index (w+)")
    ax2.set_ylabel("True-Class Prob. Drop")
    ax2.set_xticks(range(seq_len))
    ax2.set_xticklabels([str(i + 1) for i in range(seq_len)])
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="w+ の各層(18層)をアブレーションし、精度低下量から層重要度を分析する"
    )
    parser.add_argument("--checkpoint_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--latent_test_dir", required=True, help="Path to test latent files")
    parser.add_argument("--output_dir", default="eval_results/layer_importance",
                         help="Output directory for results")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=64,
                         help="推論時のバッチサイズ")
    parser.add_argument("--ablation", choices=["mean", "zero"], default="mean",
                         help="mean: 層ごとの全データ平均値に置換 / zero: ゼロに置換")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "=" * 60)
    print("Loading model...")
    print("=" * 60)
    model, model_config = load_model(args.checkpoint_path, device)
    print(f"Model configuration: {json.dumps(model_config, indent=2)}")

    print("\n" + "=" * 60)
    print("Loading test dataset...")
    print("=" * 60)
    dataset = LatentFERDataset(args.latent_test_dir)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=4)
    latents, labels = next(iter(loader))
    print(f"Test dataset size: {len(dataset)}")

    print("\n" + "=" * 60)
    print(f"Running ablation analysis (mode={args.ablation})...")
    print("=" * 60)
    results = analyze_layer_importance(model, latents, labels, device, args.batch_size, args.ablation)

    print(f"\nBaseline accuracy: {results['baseline_accuracy']:.4f}")
    most_important = max(range(len(results['layer_accuracy_drop'])),
                          key=lambda i: results['layer_accuracy_drop'][i])
    print(f"Most important layer (by accuracy drop): layer {most_important + 1} "
          f"(drop={results['layer_accuracy_drop'][most_important]:+.4f})")

    plot_layer_importance(results, os.path.join(args.output_dir, "layer_importance.png"))

    results_out = dict(results)
    results_out["ablation_mode"] = args.ablation
    results_out["checkpoint_path"] = args.checkpoint_path
    results_out["model_config"] = model_config
    results_path = os.path.join(args.output_dir, "layer_importance.json")
    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"Saved: {results_path}")


if __name__ == "__main__":
    main()
