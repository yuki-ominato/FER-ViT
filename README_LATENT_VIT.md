# LatentViT for Facial Expression Recognition

StyleGANの潜在空間とVision Transformerを組み合わせた顔画像感情認識システム

## 概要

本研究では、StyleGANの潜在空間（w+）とVision Transformer（ViT）を組み合わせることで、顔画像から感情を分類する新しいアーキテクチャを提案します。

### アーキテクチャ
1. **画像 → 潜在空間**: pSp/e4eエンコーダで画像をw+潜在コード（18×512次元）に変換
2. **潜在 → ViT**: 潜在コードをトークン系列としてViTで処理
3. **分類**: CLSトークンから感情カテゴリ（7クラス）を予測

## セットアップ

### 1. 環境構築

```bash
# conda環境作成
conda env create -f environment.yml
conda activate fer-vit

# 追加依存関係
pip install -r requirements.txt
```

### 2. 外部リポジトリの準備

```bash
# pSpリポジトリをクローン
git clone https://github.com/eladrich/pixel2style2pixel third_party/pixel2style2pixel

# または e4eリポジトリ
git clone https://github.com/omertov/encoder4editing third_party/encoder4editing
```

### 3. 事前学習済みモデルの配置

```bash
# 事前学習済み重みを配置
mkdir -p pretrained_models
# pSp FFHQエンコーダの重みをダウンロードして配置
# 例: pretrained_models/psp_ffhq_encode.pt
```

## 使用方法

### 1. データ準備

FER2013データセットを以下の構造で配置：

```
dataset/fer2013/
  train/
    angry/ *.png
    disgust/ *.png
    fear/ *.png
    happy/ *.png
    neutral/ *.png
    sad/ *.png
    surprise/ *.png
  val/
    (同様の構造)
  test/
    (同様の構造)
```

### 2. 潜在コード生成

```bash
# 学習用データの潜在コード生成
PYTHONPATH=. python scripts/generate_latents.py \
  --data_root dataset/fer2013/train \
  --latent_out latents/train \
  --encoder_model pretrained_models/psp_ffhq_encode.pt \
  --encoder_type psp \
  --batch_size 4

# 検証用データの潜在コード生成
PYTHONPATH=. python scripts/generate_latents.py \
  --data_root dataset/fer2013/val \
  --latent_out latents/val \
  --encoder_model pretrained_models/psp_ffhq_encode.pt \
  --encoder_type psp \
  --batch_size 4
```

### 3. 学習

```bash
# 基本学習
PYTHONPATH=. python train/train_latent_vit.py \
  --latent_train_dir latents/train \
  --latent_val_dir latents/val \
  --epochs 60 \
  --batch_size 64 \
  --lr 1e-4 \
  --use_class_weights

# 高度な設定での学習
PYTHONPATH=. python train/train_latent_vit.py \
  --latent_train_dir latents/train \
  --latent_val_dir latents/val \
  --epochs 100 \
  --batch_size 32 \
  --lr 5e-5 \
  --weight_decay 1e-2 \
  --scheduler plateau \
  --use_class_weights \
  --depth 12 \
  --heads 16 \
  --embed_dim 768
```

### 4. 評価・可視化

```bash
# 学習済みモデルの評価
PYTHONPATH=. python eval/evaluate_model.py \
  --checkpoint_path experiments/latent_vit_d6_h8_lr0.0001_bs64_ep60/checkpoints/best_model.pt \
  --latent_test_dir latents/test \
  --output_dir eval_results
```

### 5. TensorBoardでログ確認

```bash
# 学習ログを確認
tensorboard --logdir experiments/
```

## 実験設定

### 推奨ハイパーパラメータ

| パラメータ | 推奨値 | 説明 |
|-----------|--------|------|
| `latent_dim` | 512 | 潜在次元（pSpの出力次元） |
| `seq_len` | 18 | 潜在系列長（w+の層数） |
| `embed_dim` | 512 | ViTの埋め込み次元 |
| `depth` | 6 | Transformer層数 |
| `heads` | 8 | アテンションヘッド数 |
| `batch_size` | 64 | バッチサイズ |
| `lr` | 1e-4 | 学習率 |
| `epochs` | 60 | エポック数 |

### クラス重み

不均衡データに対応するため、`--use_class_weights`フラグを推奨：

```bash
python train/train_latent_vit.py --use_class_weights ...
```

## 実験管理

### 実験ディレクトリ構造

```
experiments/
  latent_vit_d6_h8_lr0.0001_bs64_ep60_20240101_120000/
    config.json              # 実験設定
    experiment_summary.json  # 実験サマリー
    checkpoints/             # チェックポイント
      best_model.pt
      epoch_1.pt
      ...
    logs/                    # TensorBoardログ
      events.out.tfevents.*
```

### 実験比較

```python
from utils.experiment_logger import compare_experiments

# 複数実験の結果を比較
experiment_dirs = [
    "experiments/latent_vit_d6_h8_lr0.0001_bs64_ep60_20240101_120000",
    "experiments/latent_vit_d12_h16_lr5e-05_bs32_ep100_20240101_130000",
]
results = compare_experiments(experiment_dirs, metric="f1_macro")
print(results)
```

## トラブルシューティング

### よくある問題

1. **メモリ不足**
   - バッチサイズを小さくする（`--batch_size 16`）
   - 潜在生成時のバッチサイズを調整

2. **エンコーダの読み込みエラー**
   - `third_party/`ディレクトリに正しくリポジトリが配置されているか確認
   - 事前学習済み重みのパスが正しいか確認

3. **潜在コードのshape不一致**
   - エンコーダの出力次元を確認
   - `LatentViT`の`seq_len`パラメータを調整

### デバッグ

```bash
# 詳細ログで実行
PYTHONPATH=. python -u train/train_latent_vit.py ... 2>&1 | tee training.log

# 潜在生成の進捗確認
PYTHONPATH=. python scripts/generate_latents.py ... --batch_size 1
```

## ファイル構成

```
FER-ViT/
├── models/
│   ├── latent_vit.py          # 潜在入力用ViTモデル
│   └── encoder_wrapper.py     # pSp/e4eラッパー
├── data/
│   └── latent_dataset.py      # 潜在コード用Dataset
├── train/
│   └── train_latent_vit.py    # 学習スクリプト
├── eval/
│   └── evaluate_model.py      # 評価・可視化スクリプト
├── scripts/
│   └── generate_latents.py    # 潜在コード生成スクリプト
├── utils/
│   └── experiment_logger.py   # 実験管理・ロギング
├── preprocessing.py           # 既存の前処理（拡張済み）
└── requirements.txt           # 依存関係
```

## ライセンス

このプロジェクトは研究目的で作成されています。使用する外部ライブラリ（pSp、e4e等）のライセンスに従ってください。

## 引用

本研究を引用する場合は、以下の形式でお願いします：

```bibtex
@article{latent_vit_fer_2024,
  title={LatentViT: Vision Transformer for Facial Expression Recognition in StyleGAN Latent Space},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```
