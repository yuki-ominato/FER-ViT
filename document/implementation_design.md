# LatentViT 発展研究 実装設計書

## 概要

本設計書は、既存の `yuki-ominato-fer-vit` リポジトリに対して、以下の2つの機能を追加実装するための仕様を定める。

1. **w+前処理モジュール群**（LEAM・SemanticPE・LayerWiseNorm）
2. **SeFaを用いた意味的データ拡張**（Semantic Augmentation）

---

## ディレクトリ構成（変更後）

```
yuki-ominato-fer-vit/
├── README.md
├── environment.yml
├── preprocessing.py
├── requirements.txt
├── vit-fer.py
│
├── data/
│   ├── generate_latents.py         # 既存：変更なし
│   ├── image_dataset.py            # 既存：変更なし
│   ├── latent_dataset.py           # 既存：変更なし
│   └── augment_latents.py          # 【新規】SeFaによる潜在コード拡張
│
├── models_fer_vit/
│   ├── encoder_wrapper.py          # 既存：変更なし
│   ├── image_vit.py                # 既存：変更なし
│   ├── latent_cnn.py               # 既存：変更なし
│   ├── latent_vit.py               # 既存：変更なし
│   └── latent_vit_v2.py            # 【新規】前処理モジュール統合版LatentViT
│
├── modules/                        # 【新規】前処理モジュール群
│   ├── __init__.py
│   ├── leam.py                     # 【新規】層別アテンションマスク
│   ├── semantic_pe.py              # 【新規】意味的位置エンコーディング
│   └── layer_wise_norm.py          # 【新規】層別正規化
│
├── sefa/                           # 【新規】SeFa関連
│   ├── __init__.py
│   ├── factorize.py                # 【新規】StyleGAN重みの因子分解
│   └── verify_directions.py        # 【新規】発見した方向の妥当性検証
│
├── train/
│   ├── train_image_vit.py          # 既存：変更なし
│   ├── train_latent_cnn.py         # 既存：変更なし
│   ├── train_latent_vit.py         # 既存：変更なし
│   └── train_latent_vit_v2.py      # 【新規】v2モデル用学習スクリプト
│
├── eval/
│   ├── evaluate_image_vit.py       # 既存：変更なし
│   ├── evaluate_model.py           # 既存：変更なし
│   ├── plot_data_efficiency.py     # 既存：変更なし
│   └── visualize_leam_weights.py   # 【新規】LEAM重み可視化
│
└── utils/
    ├── __init__.py                 # 既存：変更なし
    └── experiment_logger.py        # 既存：変更なし
```

---

## 実装フェーズ

```
Phase 1：前処理モジュール実装      （modules/）
Phase 2：LatentViT v2 実装         （models_fer_vit/latent_vit_v2.py）
Phase 3：学習・評価スクリプト整備  （train/ eval/）
Phase 4：SeFa実装                  （sefa/）
Phase 5：意味的データ拡張の統合    （data/augment_latents.py）
```

Phaseの順に実装することを推奨する。Phase 1〜3で前処理モジュール単体の効果を検証してから、Phase 4〜5でデータ拡張を追加する。

---

## Phase 1：前処理モジュール実装

### 1-1. `modules/leam.py`

**役割**：w+の各層（18層）に学習可能なスカラー重みを乗じ、表情に寄与する中間層を強調する。

```python
import torch
import torch.nn as nn


class LEAM(nn.Module):
    """
    Layer-wise Expression Attention Mask
    w+の各層に学習可能な重みを掛けることで、
    表情認識に寄与する層を強調し、不要な層を抑制する。

    Args:
        num_layers (int): w+のシーケンス長。pSpの場合は18。
        init_coarse (float): 浅層（層1〜4）の初期重み。デフォルト0.5。
        init_fine (float): 深層（層13〜18）の初期重み。デフォルト0.5。
    """

    def __init__(
        self,
        num_layers: int = 18,
        init_coarse: float = 0.5,
        init_fine: float = 0.5,
    ):
        super().__init__()

        # 初期値：中間層=1.0、浅層・深層=0.5
        init = torch.ones(num_layers)
        init[:4] = init_coarse   # Coarse層（層1〜4）
        init[12:] = init_fine    # Fine層（層13〜18）
        self.layer_weights = nn.Parameter(init)

    def forward(self, w_plus: torch.Tensor) -> torch.Tensor:
        """
        Args:
            w_plus: (B, num_layers, latent_dim)
        Returns:
            (B, num_layers, latent_dim)
        """
        # sigmoid で0〜1に正規化してスケール
        weights = torch.sigmoid(self.layer_weights)          # (num_layers,)
        return w_plus * weights.unsqueeze(0).unsqueeze(-1)   # (B, L, D)

    def get_weights(self) -> torch.Tensor:
        """可視化用：学習済み重みを返す"""
        return torch.sigmoid(self.layer_weights).detach().cpu()
```

---

### 1-2. `modules/semantic_pe.py`

**役割**：StyleGANのw+が持つ階層構造（Coarse/Medium/Fine）を位置情報としてViTに伝える。

```python
import torch
import torch.nn as nn


# 各層が属するグループのID（Coarse=0, Medium=1, Fine=2）
_LAYER_GROUPS = [0, 0, 0, 0,          # 層1〜4:  Coarse（顔の骨格・全体構造）
                 1, 1, 1, 1, 1, 1, 1, 1,  # 層5〜12: Medium（表情・目・口の形状）
                 2, 2, 2, 2, 2, 2]        # 層13〜18: Fine（テクスチャ・肌色）


class SemanticPE(nn.Module):
    """
    Semantic Positional Encoding
    StyleGANのw+層が持つCoarse/Medium/Fineの階層構造を
    学習可能な位置ベクトルとして付与する。

    グループベクトル（3種）と層個別ベクトル（18種）の和を
    各トークンに加算する。

    Args:
        d_model (int): 潜在ベクトルの次元数。pSpの場合は512。
        num_layers (int): w+のシーケンス長。pSpの場合は18。
    """

    def __init__(self, d_model: int = 512, num_layers: int = 18):
        super().__init__()
        self.group_embed = nn.Embedding(3, d_model)   # Coarse/Medium/Fine
        self.layer_embed = nn.Embedding(num_layers, d_model)

        # グループIDをバッファとして登録（学習不要・デバイス追従）
        self.register_buffer(
            "groups",
            torch.tensor(_LAYER_GROUPS, dtype=torch.long)
        )

    def forward(self, w_plus: torch.Tensor) -> torch.Tensor:
        """
        Args:
            w_plus: (B, num_layers, d_model)
        Returns:
            (B, num_layers, d_model)
        """
        device = w_plus.device
        layers = torch.arange(w_plus.size(1), device=device)  # (L,)

        # グループ埋め込み + 層個別埋め込み
        pe = self.group_embed(self.groups) + self.layer_embed(layers)  # (L, D)
        return w_plus + pe.unsqueeze(0)   # (B, L, D)
```

---

### 1-3. `modules/layer_wise_norm.py`

**役割**：w+の各層が持つ値分布の不均一さを、層ごとに独立したLayerNormで正規化する。

```python
import torch
import torch.nn as nn


class LayerWiseNorm(nn.Module):
    """
    Layer-wise Normalization
    w+の各層に独立したLayerNormを適用する。
    通常のViTが全トークン共通のLayerNormを使うのに対し、
    w+の層別意味構造の違いを考慮した正規化を行う。

    Args:
        num_layers (int): w+のシーケンス長。pSpの場合は18。
        d_model (int): 潜在ベクトルの次元数。pSpの場合は512。
    """

    def __init__(self, num_layers: int = 18, d_model: int = 512):
        super().__init__()
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

    def forward(self, w_plus: torch.Tensor) -> torch.Tensor:
        """
        Args:
            w_plus: (B, num_layers, d_model)
        Returns:
            (B, num_layers, d_model)
        """
        out = torch.stack(
            [self.norms[i](w_plus[:, i, :]) for i in range(len(self.norms))],
            dim=1
        )
        return out  # (B, L, D)
```

---

### 1-4. `modules/__init__.py`

```python
from .leam import LEAM
from .semantic_pe import SemanticPE
from .layer_wise_norm import LayerWiseNorm

__all__ = ["LEAM", "SemanticPE", "LayerWiseNorm"]
```

---

## Phase 2：LatentViT v2 実装

### `models_fer_vit/latent_vit_v2.py`

既存の `latent_vit.py` の `LatentViT` クラスを継承し、前処理モジュールをオプションとして追加する。フラグで有効/無効を切り替えられるため、アブレーション実験がしやすい設計にする。

```python
import torch
import torch.nn as nn
from .latent_vit import LatentViT
from modules import LEAM, SemanticPE, LayerWiseNorm


class LatentViTv2(nn.Module):
    """
    LatentViT v2：前処理モジュール統合版

    既存のLatentViTに以下のモジュールをオプションで追加する。
        - use_lwn  : LayerWiseNorm（層別正規化）
        - use_spe  : SemanticPE（意味的位置エンコーディング）
        - use_leam : LEAM（層別アテンションマスク）

    前処理の適用順：
        w+ → [LayerWiseNorm] → [SemanticPE] → [LEAM] → LatentViT

    Args:
        latent_dim (int): w+の次元数。デフォルト512。
        seq_len (int): w+のシーケンス長。デフォルト18。
        embed_dim (int): Transformer内部次元数。デフォルト512。
        depth (int): Transformerのブロック数。デフォルト6。
        heads (int): Attentionヘッド数。デフォルト8。
        mlp_dim (int): FFNの中間次元数。デフォルト2048。
        num_classes (int): 分類クラス数。デフォルト7。
        dropout (float): ドロップアウト率。デフォルト0.1。
        use_lwn (bool): LayerWiseNormを使用するか。デフォルトFalse。
        use_spe (bool): SemanticPEを使用するか。デフォルトFalse。
        use_leam (bool): LEAMを使用するか。デフォルトFalse。
    """

    def __init__(
        self,
        latent_dim: int = 512,
        seq_len: int = 18,
        embed_dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 2048,
        num_classes: int = 7,
        dropout: float = 0.1,
        use_lwn: bool = False,
        use_spe: bool = False,
        use_leam: bool = False,
    ):
        super().__init__()

        # 前処理モジュール（オプション）
        self.lwn  = LayerWiseNorm(seq_len, latent_dim) if use_lwn  else nn.Identity()
        self.spe  = SemanticPE(latent_dim, seq_len)    if use_spe  else nn.Identity()
        self.leam = LEAM(seq_len)                      if use_leam else nn.Identity()

        # 既存のLatentViTをバックボーンとして使用
        self.backbone = LatentViT(
            latent_dim=latent_dim,
            seq_len=seq_len,
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        # フラグを保持（ログ・デバッグ用）
        self.use_lwn  = use_lwn
        self.use_spe  = use_spe
        self.use_leam = use_leam

    def forward(self, w_plus: torch.Tensor) -> torch.Tensor:
        """
        Args:
            w_plus: (B, seq_len, latent_dim)
        Returns:
            logits: (B, num_classes)
        """
        x = self.lwn(w_plus)   # LayerWiseNorm
        x = self.spe(x)        # SemanticPE
        x = self.leam(x)       # LEAM
        return self.backbone(x)

    def get_leam_weights(self) -> torch.Tensor:
        """LEAM重みを取得（可視化用）。use_leam=Falseの場合はNoneを返す。"""
        if self.use_leam:
            return self.leam.get_weights()
        return None

    def get_config(self) -> dict:
        """実験ログ用にモデル設定を返す"""
        return {
            "model": "LatentViTv2",
            "use_lwn": self.use_lwn,
            "use_spe": self.use_spe,
            "use_leam": self.use_leam,
        }
```

---

## Phase 3：学習・評価スクリプト整備

### `train/train_latent_vit_v2.py` の主要引数

既存の `train/train_latent_vit.py` をベースに、以下の引数を追加する。

```python
# モデル設定（既存引数に追加）
parser.add_argument("--use_lwn",  action="store_true", help="LayerWiseNormを使用")
parser.add_argument("--use_spe",  action="store_true", help="SemanticPEを使用")
parser.add_argument("--use_leam", action="store_true", help="LEAMを使用")
```

**実行例（アブレーション実験）：**

```bash
# E0: ベースライン（前処理なし）
python train/train_latent_vit_v2.py \
    --latent_train_dir data/latents/train \
    --latent_val_dir   data/latents/val \
    --experiment_name  E0_baseline

# E3: LEAM のみ
python train/train_latent_vit_v2.py \
    --use_leam \
    --experiment_name E3_leam

# E7: 全モジュール（提案手法）
python train/train_latent_vit_v2.py \
    --use_lwn --use_spe --use_leam \
    --experiment_name E7_proposed
```

### `eval/visualize_leam_weights.py`

学習済みチェックポイントからLEAM重みを取り出し、棒グラフで可視化する。

```python
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_leam_weights(checkpoint_path: str, save_path: str = None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # state_dictからLEAM重みを抽出
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
        mpatches.Patch(color="#e74c3c", label="Coarse（層1〜4: 骨格・構造）"),
        mpatches.Patch(color="#2ecc71", label="Medium（層5〜12: 表情・形状）"),
        mpatches.Patch(color="#3498db", label="Fine（層13〜18: テクスチャ）"),
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
    plt.show()
```

---

## Phase 4：SeFa実装

### `sefa/factorize.py`

StyleGAN2の事前学習済み重みから非表情方向ベクトルを抽出する。

```python
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
```

---

### `sefa/verify_directions.py`

発見した方向ベクトルが表情を変えないことを検証する。

```python
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
                w_perturbed += step * dir_tensor.unsqueeze(0)  # (18, 512)にブロードキャスト

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
```

---

## Phase 5：意味的データ拡張の統合

### `data/augment_latents.py`

SeFaで発見・検証済みの非表情方向ベクトルを用いて、訓練用潜在コードを拡張する。

```python
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
```

---

## 実験条件の整理

### アブレーション実験（E0〜E7）

| ID | use_lwn | use_spe | use_leam | 説明 |
|----|---------|---------|----------|------|
| E0 | False | False | False | ベースライン（既存LatentViT相当） |
| E1 | True | False | False | LayerWiseNormのみ |
| E2 | False | True | False | SemanticPEのみ |
| E3 | False | False | True | LEAMのみ |
| E4 | True | True | False | LWN + SPE |
| E5 | False | True | True | SPE + LEAM |
| E6 | True | False | True | LWN + LEAM |
| **E7** | **True** | **True** | **True** | **全モジュール（提案手法）** |

### データ拡張実験（E7 + Aug）

| ID | 説明 |
|----|------|
| E7 | 提案手法（拡張なし） |
| E7+Aug | 提案手法 + SeFaによる意味的データ拡張 |

### データ量変化実験

各条件を以下のデータ使用率で評価する。

```
データ使用率: 10% / 25% / 50% / 100%
評価指標: Accuracy（7クラス）、diff-acc（= acc@100% - acc@10%）
```

---

## 注意事項

**Phase 4（SeFa）の実施前に確認すること：**

`verify_directions.py` を実行し、採用する方向ベクトルの `label_change_rate` が **0.1以下**であることを確認してから `augment_latents.py` を実行すること。`label_change_rate` が高い方向を使った場合、ラベルが不正な拡張サンプルが生成され学習が劣化する。

目安として、`label_change_rate <= 0.1` の方向のみを `direction_indices` に指定することを推奨する。
