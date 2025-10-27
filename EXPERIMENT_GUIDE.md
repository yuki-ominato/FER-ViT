# ViT+StyleGAN 顔感情認識実験ガイド

## 📋 概要

このガイドでは、StyleGANの潜在空間（w+）とVision Transformer（ViT）を組み合わせた顔画像感情認識システムの実験手順を説明します。

## 🏗️ 実験フロー

```
画像データ → 潜在コード生成 → 学習 → 評価 → TensorBoard可視化
```

## 📁 ディレクトリ構造

```
fer-vit/
├── dataset/fer2013/           # 元画像データ
│   ├── train/                # 学習用画像
│   ├── val/                  # 検証用画像
│   └── test/                 # テスト用画像
├── latents/                   # 生成された潜在コード
│   ├── train/                # 学習用潜在コード
│   ├── val/                  # 検証用潜在コード
│   └── test/                 # テスト用潜在コード
├── experiments/              # 実験ログ
│   └── {experiment_name}/    # 実験ディレクトリ
│       ├── config.json       # 実験設定
│       ├── checkpoints/      # モデルチェックポイント
│       └── logs/             # TensorBoardログ
└── pretrained_models/        # 事前学習済みモデル
    └── psp_ffhq_frontalization.pt
```

## 🚀 実行手順

### 1. 環境準備

```bash
# プロジェクトディレクトリに移動
cd /home/yuki/research2/fer-vit

# conda環境をアクティベート
conda activate fer-vit

# PYTHONPATHを設定
export PYTHONPATH=/home/yuki/research2/fer-vit:/home/yuki/research2/fer-vit/third_party/pixel2style2pixel
```

### 2. 潜在コード生成

#### 2.1 学習用データの潜在コード生成

```bash
conda run -n fer-vit python -u scripts/generate_latents.py \
  --data_root /home/yuki/research2/dataset/fer2013/train \
  --latent_out /home/yuki/research2/fer-vit/latents/train \
  --encoder_model /home/yuki/research2/fer-vit/pretrained_models/psp_ffhq_encode.pt \
  --encoder_type psp \
  --batch_size 4
```

#### 2.2 検証用データの潜在コード生成

```bash
conda run -n fer-vit python -u scripts/generate_latents.py \
  --data_root /home/yuki/research2/dataset/fer2013/val \
  --latent_out /home/yuki/research2/fer-vit/latents/val \
  --encoder_model /home/yuki/research2/fer-vit/pretrained_models/psp_ffhq_encode.pt \
  --encoder_type psp \
  --batch_size 4
```

#### 2.3 テスト用データの潜在コード生成

```bash
conda run -n fer-vit python -u scripts/generate_latents.py \
  --data_root /home/yuki/research2/dataset/fer2013/test \
  --latent_out /home/yuki/research2/fer-vit/latents/test \
  --encoder_model /home/yuki/research2/fer-vit/pretrained_models/psp_ffhq_encode.pt \
  --encoder_type psp \
  --batch_size 4
```

### 3. 学習実行

#### 3.1 基本学習（推奨設定）

```bash
conda run -n fer-vit python -u train/train_latent_vit.py \
  --latent_train_dir latents/train \
  --latent_val_dir latents/val \
  --epochs 60 \
  --batch_size 64 \
  --lr 1e-4 \
  --weight_decay 1e-2 \
  --scheduler plateau \
  --use_class_weights
```

#### 3.2 高度な設定での学習

```bash
conda run -n fer-vit python -u train/train_latent_vit.py \
  --latent_train_dir latents/train \
  --latent_val_dir latents/val \
  --epochs 100 \
  --batch_size 32 \
  --lr 5e-5 \
  --weight_decay 1e-2 \
  --scheduler cosine \
  --use_class_weights \
  --embed_dim 768 \
  --depth 12 \
  --heads 12
```

#### 3.3 モデルパラメータ

| パラメータ | デフォルト値 | 説明 |
|------------|--------------|------|
| `--latent_dim` | 512 | StyleGAN潜在次元 |
| `--seq_len` | 18 | w+レイヤー数 |
| `--embed_dim` | 512 | ViT埋め込み次元 |
| `--depth` | 6 | Transformer深度 |
| `--heads` | 8 | アテンションヘッド数 |
| `--mlp_dim` | 2048 | MLP次元 |
| `--num_classes` | 7 | 感情クラス数 |
| `--dropout` | 0.1 | ドロップアウト率 |

### 4. 評価実行

```bash
conda run -n fer-vit python -u eval/evaluate_latent_vit.py \
  --checkpoint_path experiments/{experiment_name}/{run_id}/checkpoints/best_model.pt \
  --latent_test_dir latents/test \
  --batch_size 32 \
  --output_file test_results.json
```

### 5. TensorBoard可視化

#### 5.1 TensorBoard起動

```bash
# 既存のTensorBoardプロセスを停止
pkill -f tensorboard

# TensorBoardを起動
conda run -n fer-vit tensorboard \
  --logdir experiments/{experiment_name}/{run_id}/logs \
  --port 6006
```

#### 5.2 ブラウザでアクセス

- URL: `http://localhost:6006`
- または: `http://127.0.0.1:6006`

#### 5.3 可視化されるメトリクス

- **SCALARS**:
  - `Loss/Train`: 学習損失
  - `Validation/accuracy`: 検証精度
  - `Validation/f1_macro`: 検証F1マクロスコア
  - `Validation/f1_weighted`: 検証F1重み付きスコア
  - `Learning_Rate/Group_0`: 学習率

- **HISTOGRAMS** (10エポックごと):
  - `Parameters`: モデルパラメータ分布
  - `Gradients`: 勾配分布

- **IMAGES** (最終エポック):
  - `Confusion_Matrix`: 混同行列

## 🔧 トラブルシューティング

### 潜在コード生成エラー

```bash
# エラー: ModuleNotFoundError: No module named 'models.psp'
# 解決: PYTHONPATHを設定
export PYTHONPATH=/home/yuki/research2/fer-vit:/home/yuki/research2/fer-vit/third_party/pixel2style2pixel

# エラー: CUDA out of memory
# 解決: バッチサイズを削減
--batch_size 2
```

### 学習エラー

```bash
# エラー: torch.use_deterministic_algorithms
# 解決: 環境変数を設定
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# エラー: ModuleNotFoundError: No module named 'utils.experiment_logger'
# 解決: utils/__init__.pyが存在することを確認
```

### TensorBoardエラー

```bash
# エラー: TensorBoard could not bind to port 6006
# 解決: 既存プロセスを停止
pkill -f tensorboard

# エラー: No dashboards are active
# 解決: 正確なログディレクトリを指定
--logdir experiments/{experiment_name}/{run_id}/logs
```

## 📊 実験結果の確認

### 1. 学習ログの確認

```bash
# 実験ディレクトリの確認
ls -la experiments/

# 最新の実験結果
latest_exp=$(ls -t experiments/ | head -1)
echo "Latest experiment: $latest_exp"

# 設定ファイルの確認
cat experiments/$latest_exp/*/config.json
```

### 2. チェックポイントの確認

```bash
# ベストモデルの確認
ls -la experiments/$latest_exp/*/checkpoints/best_model.pt

# チェックポイントの詳細
conda run -n fer-vit python -c "
import torch
ckpt = torch.load('experiments/$latest_exp/*/checkpoints/best_model.pt', map_location='cpu')
print('Epoch:', ckpt['epoch'])
print('Val F1:', ckpt['val_f1'])
print('Val Acc:', ckpt['val_acc'])
"
```

### 3. TensorBoardログの確認

```bash
# イベントファイルの確認
ls -la experiments/$latest_exp/*/logs/

# 利用可能なタグの確認
conda run -n fer-vit python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('experiments/$latest_exp/*/logs/events.out.tfevents.*')
ea.Reload()
print('Available tags:', ea.Tags())
"
```

## 🎯 実験のベストプラクティス

### 1. 実験管理

- **実験名**: 設定に基づいて自動生成される
- **チェックポイント**: ベストモデルが自動保存される
- **ログ**: TensorBoardで可視化される

### 2. パフォーマンス最適化

- **バッチサイズ**: GPU メモリに応じて調整
- **学習率**: 1e-4から開始、スケジューラーで調整
- **エポック数**: 60-100エポックで十分

### 3. デバッグ

- **スモークテスト**: 小規模データで動作確認
- **ログ監視**: TensorBoardでリアルタイム監視
- **チェックポイント**: 定期的なモデル保存

## 📝 実験記録テンプレート

### 実験設定

| 項目 | 値 |
|------|-----|
| 実験名 | {experiment_name} |
| 実行日時 | {timestamp} |
| エポック数 | {epochs} |
| バッチサイズ | {batch_size} |
| 学習率 | {lr} |
| スケジューラー | {scheduler} |

### 結果

| メトリクス | 値 |
|------------|-----|
| 最終精度 | {final_accuracy} |
| ベストF1マクロ | {best_f1_macro} |
| ベストF1重み付き | {best_f1_weighted} |

### 備考

- 学習時間: {training_time}
- GPU使用率: {gpu_usage}
- メモリ使用量: {memory_usage}

---

## 🔗 関連ファイル

- `train/train_latent_vit.py`: メイン学習スクリプト
- `scripts/generate_latents.py`: 潜在コード生成スクリプト
- `eval/evaluate_latent_vit.py`: 評価スクリプト
- `utils/experiment_logger.py`: 実験管理システム
- `data/latent_dataset.py`: 潜在コードデータセット
- `models_fer_vit/latent_vit.py`: LatentViTモデル
