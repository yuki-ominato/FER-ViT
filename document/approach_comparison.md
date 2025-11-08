# 感情認識のための3つのアプローチ比較

## アーキテクチャの違い

### 1. スクラッチViT（従来手法）
```
FER2013画像
  ↓
pSp Encoder
  ↓
StyleGAN潜在コード (18, 512)
  ↓
Linear Projection
  ↓
Transformer (ランダム初期化)
  ↓
分類ヘッド
```

**特徴:**
- 全パラメータをランダム初期化
- StyleGAN潜在空間の表現力を活用
- 学習に時間がかかる
- データ量に依存

---

### 2. ハイブリッドViT（提案手法）✨
```
FER2013画像
  ↓
pSp Encoder
  ↓
StyleGAN潜在コード (18, 512)
  ↓
Linear Projection
  ↓
Transformer (ImageNet事前学習済み) ← 重要!
  ↓
分類ヘッド
```

**特徴:**
- **事前学習済みTransformerを活用**
- StyleGAN潜在空間 + 事前学習の両方のメリット
- 少ないエポックで高精度
- パラメータ効率的な学習も可能（Adapter）

**実装の工夫:**
1. timmからViTをロード
2. パッチ埋め込み層をスキップ
3. Transformer Encoderのみを抽出
4. 位置埋め込みを補間（196→18）
5. 入力射影層で次元を合わせる

---

### 3. 標準ViT（ベースライン）
```
FER2013画像 (48x48)
  ↓
Patch Embedding
  ↓
Transformer (ImageNet事前学習済み)
  ↓
分類ヘッド
```

**特徴:**
- 標準的な画像ViT
- 事前学習を活用
- データ拡張が重要
- StyleGAN表現は使わない

---

## ファインチューニング戦略

### Strategy 1: 完全ファインチューニング
```bash
python train/train_hybrid_latent_vit.py \
    --use_pretrained \
    --lr 1e-4
```
- 全パラメータを学習
- 最高精度が期待できる
- 学習時間: 中程度

### Strategy 2: Adapter-based（推奨）
```bash
python train/train_hybrid_latent_vit.py \
    --freeze_transformer \
    --use_adapter \
    --adapter_dim 64 \
    --lr 1e-3
```
- Transformerを凍結
- 小さなAdapterのみ学習
- **パラメータ数: 最小（学習は~5%）**
- 学習時間: 最速
- メモリ効率的

### Strategy 3: Linear Probing
```bash
python train/train_hybrid_latent_vit.py \
    --freeze_transformer \
    --lr 1e-3
```
- ヘッドのみ学習
- ベースライン評価用
- 学習時間: 最速

### Strategy 4: 部分凍結
```bash
python train/train_hybrid_latent_vit.py \
    --freeze_stages 6 \
    --lr 3e-4
```
- 下位層凍結、上位層学習
- バランスの取れたアプローチ

---

## 期待される性能比較

| アプローチ | 精度 | 学習時間 | パラメータ数 | データ要件 |
|-----------|------|---------|------------|----------|
| スクラッチViT | ★★★☆☆ | 長い | 5-20M | 大量 |
| **ハイブリッド（完全）** | ★★★★★ | 中程度 | 22M | 中程度 |
| **ハイブリッド（Adapter）** | ★★★★☆ | 短い | 22M (1M学習) | 少量 |
| 標準ViT | ★★★★☆ | 中程度 | 22M | 中程度 |

---

## 実験設定の推奨

### 小規模データセット（FER2013等）
```bash
# 最優先: Adapter-based
python train/train_hybrid_latent_vit.py \
    --model_size small \
    --freeze_transformer \
    --use_adapter \
    --adapter_dim 64 \
    --batch_size 128 \
    --lr 1e-3 \
    --epochs 60
```

### 中規模データセット
```bash
# 完全ファインチューニング
python train/train_hybrid_latent_vit.py \
    --model_size small \
    --use_pretrained \
    --batch_size 64 \
    --lr 1e-4 \
    --use_layerwise_lr \
    --epochs 60
```

### 大規模データセット
```bash
# より大きなモデル
python train/train_hybrid_latent_vit.py \
    --model_size base \
    --use_pretrained \
    --batch_size 32 \
    --lr 5e-5 \
    --epochs 50
```

---

## 技術的な詳細

### 位置埋め込みの補間
```python
# ImageNet ViT: 14x14=196パッチ
# StyleGAN: 18層

# 1D補間で196→18に変換
patch_pos = nn.functional.interpolate(
    patch_pos,
    size=18,
    mode='linear',
    align_corners=False
)
```

### レイヤーごとの学習率
```python
# Input projection: lr × 10
# Transformer: lr × 1
# Head: lr × 10
# Position/CLS: lr × 5
```

### Adapterの構造
```python
# ボトルネック構造
Linear(embed_dim → adapter_dim)  # 圧縮
GELU()
Linear(adapter_dim → embed_dim)  # 復元
```

---

## 評価項目

### 1. 精度
- Test Accuracy
- F1 Score (macro/weighted)
- クラスごとの性能

### 2. 効率性
- 学習時間
- 推論速度
- GPU メモリ使用量

### 3. パラメータ効率
- 総パラメータ数
- 学習可能パラメータ数
- パラメータあたりの性能

### 4. データ効率
- Few-shot性能
- データ量による精度変化

---

## コードの使用例

```bash
# 1. 潜在コード生成（既存）
python scripts/generate_latents.py \
    --data_root data/fer2013/train \
    --latent_out data/latents/train \
    --encoder_model pretrained_models/psp_ffhq_encode.pt

# 2. ハイブリッドViT学習（新規）
python train/train_hybrid_latent_vit.py \
    --latent_train_dir data/latents/train \
    --latent_val_dir data/latents/val \
    --model_size small \
    --use_pretrained \
    --use_adapter \
    --epochs 60

# 3. 評価
python eval/evaluate_model.py \
    --checkpoint_path experiments/.../best_model.pt \
    --latent_test_dir data/latents/test
```

---

## まとめ

### 提案手法の利点
1. ✅ **事前学習の活用**: ImageNetで学習された強力な表現
2. ✅ **StyleGAN潜在空間**: 顔の意味的な構造を保持
3. ✅ **パラメータ効率的**: Adapterで少ないパラメータで学習可能
4. ✅ **実装が簡単**: timmを使って容易に実装

### 従来手法との違い
- スクラッチViTより**高速・高精度**
- 標準ViTより**StyleGAN表現を活用**
- 両方のメリットを組み合わせた**ハイブリッドアプローチ**