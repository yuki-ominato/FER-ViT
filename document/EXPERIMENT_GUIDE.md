# FER-ViT 実験ガイド

LatentViT（StyleGAN潜在空間 + Vision Transformer）による顔画像感情認識の実験手順を説明します。

## 目次

1. [実験フロー概要](#実験フロー概要)
2. [環境構築](#環境構築)
3. [データ準備](#データ準備)
4. [潜在コード生成](#潜在コード生成)
5. [モデル学習](#モデル学習)
6. [評価・可視化](#評価可視化)
7. [実験管理](#実験管理)
8. [トラブルシューティング](#トラブルシューティング)

---

## 実験フロー概要

```
[画像データ] → [エンコード] → [潜在コード] → [ViT学習] → [評価]
   FER2013      pSp/e4e       w+ (18×512)    LatentViT    混同行列等
```

### アーキテクチャ

1. **画像 → 潜在空間**: pSp/e4eエンコーダで画像をStyleGAN w+潜在コード（18×512次元）に変換
2. **潜在 → ViT**: 潜在コードをトークン系列としてViTで処理
3. **分類**: CLSトークンから7つの感情クラスを予測

---

## 環境構築

### 1. Conda環境作成

```bash
# 環境作成
conda env create -f environment.yml
conda activate fer-vit

# 追加依存関係
pip install -r requirements.txt
```

### 2. 外部リポジトリ配置

```bash
# pixel2style2pixelをクローン
git clone https://github.com/eladrich/pixel2style2pixel.git third_party/pixel2style2pixel

# または encoder4editing
git clone https://github.com/omertov/encoder4editing.git third_party/encoder4editing
```

### 3. 事前学習済みモデル配置

```bash
mkdir -p pretrained_models

# pSp FFHQエンコーダをダウンロード
# 公式リポジトリから psp_ffhq_encode.pt を取得し配置
# 配置先: pretrained_models/psp_ffhq_encode.pt
```

### 4. paths_config.py設定

`third_party/pixel2style2pixel/configs/paths_config.py` を編集：

```python
# StyleGAN2重み
stylegan_weights = '/path/to/stylegan2-ffhq-config-f.pt'

# IR-SE50重み
ir_se50 = '/path/to/model_ir_se50.pth'
```

---

## データ準備

### FER2013データセット構造

```
dataset/fer2013/
├── train/
│   ├── angry/*.png
│   ├── disgust/*.png
│   ├── fear/*.png
│   ├── happy/*.png
│   ├── neutral/*.png
│   ├── sad/*.png
│   └── surprise/*.png
├── val/
│   └── (同様の構造)
└── test/
    └── (同様の構造)
```

### データ前処理（推奨）

顔アライメントを事前に実施すると精度が向上します：

```bash
# MTCNN/dlibなどで256×256にアライメント
python scripts/preprocess_faces.py \
  --input_dir raw_data/fer2013 \
  --output_dir dataset/fer2013 \
  --size 256
```

---

## 潜在コード生成

### 基本コマンド

```bash
# 訓練データ
PYTHONPATH=. python scripts/generate_latents.py \
  --data_root ../dataset/fer2013/train \
  --latent_out latents/train \
  --encoder_model pretrained_models/psp_ffhq_encode.pt \
  --encoder_type psp \
  --batch_size 4

# 検証データ
PYTHONPATH=. python scripts/generate_latents.py \
  --data_root ../dataset/fer2013/val \
  --latent_out latents/val \
  --encoder_model pretrained_models/psp_ffhq_encode.pt \
  --encoder_type psp \
  --batch_size 4

# テストデータ
PYTHONPATH=. python scripts/generate_latents.py \
  --data_root ../dataset/fer2013/test \
  --latent_out latents/test \
  --encoder_model pretrained_models/psp_ffhq_encode.pt \
  --encoder_type psp \
  --batch_size 4
```

### スモークテスト用（少量データ）

```bash
# 動作確認用の少量データで潜在コード生成
PYTHONPATH=. python scripts/generate_latents.py \
  --data_root dataset/fer2013/train_smoke \
  --latent_out latents/train_smoke \
  --encoder_model pretrained_models/psp_ffhq_encode.pt \
  --encoder_type psp \
  --batch_size 2
```

### 注意点

- **バッチサイズ**: GPU VRAM 12GB推奨、8GBの場合は `--batch_size 2`
- **処理時間**: 1000枚あたり約10-20分（GPU性能による）
- **既存ファイルスキップ**: 再実行時は未処理分のみ生成

### 出力確認

```bash
# 生成された潜在コードファイル数を確認
ls latents/train/*.pt | wc -l

# サンプル潜在コードの形状確認
python -c "
import torch
data = torch.load('latents/train/angry_000.pt')
print(f'Latent shape: {data[\"latent\"].shape}')
print(f'Label: {data[\"label\"]}')
"
# 期待出力: Latent shape: torch.Size([18, 512])
```

---

## モデル学習

### 基本学習（推奨設定）

```bash
PYTHONPATH=. python train/train_latent_vit.py \
  --latent_train_dir latents/train \
  --latent_val_dir latents/val \
  --epochs 60 \
  --batch_size 64 \
  --lr 1e-4 \
  --use_class_weights \
  --scheduler plateau
```

### スモークテスト（動作確認用）

```bash
# 少量データで2エポックのみ実行
PYTHONPATH=. python train/train_latent_vit.py \
  --latent_train_dir latents/train_smoke \
  --latent_val_dir latents/val_smoke \
  --epochs 2 \
  --batch_size 8 \
  --lr 1e-4 \
  --use_class_weights
```

### 高度な設定

```bash
PYTHONPATH=. python train/train_latent_vit.py \
  --latent_train_dir latents/train \
  --latent_val_dir latents/val \
  --epochs 100 \
  --batch_size 32 \
  --lr 5e-5 \
  --weight_decay 1e-2 \
  --scheduler cosine \
  --use_class_weights \
  --depth 12 \
  --heads 16 \
  --embed_dim 768 \
  --mlp_dim 3072
```

### ハイパーパラメータ説明

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `--latent_dim` | 512 | 潜在コード次元（pSp出力） |
| `--seq_len` | 18 | 潜在系列長（w+の層数） |
| `--embed_dim` | 512 | ViT埋め込み次元 |
| `--depth` | 6 | Transformer層数 |
| `--heads` | 8 | Attentionヘッド数 |
| `--mlp_dim` | 2048 | MLP隠れ層次元 |
| `--dropout` | 0.1 | ドロップアウト率 |
| `--batch_size` | 64 | バッチサイズ |
| `--lr` | 1e-4 | 学習率 |
| `--weight_decay` | 1e-2 | 重み減衰 |
| `--scheduler` | plateau | 学習率スケジューラ（plateau/cosine/none） |
| `--use_class_weights` | False | クラス不均衡対応 |
| `--epochs` | 60 | エポック数 |
| `--seed` | 42 | 乱数シード |

### 学習のモニタリング

```bash
# TensorBoardで学習進捗を確認
tensorboard --logdir experiments/ --port 6006

# ブラウザで http://localhost:6006 にアクセス
```

確認項目：
- **Loss/Train**: 訓練損失が減少しているか
- **Validation/accuracy**: 検証精度が上昇しているか
- **Validation/f1_macro**: F1スコアが改善しているか
- **Learning_Rate**: 学習率の推移

---

## 評価・可視化

### 基本評価

```bash
PYTHONPATH=. python eval/evaluate_model.py \
  --checkpoint_path experiments/<実験ディレクトリ>/checkpoints/best_model.pt \
  --latent_test_dir latents/test \
  --output_dir eval_results \
  --batch_size 32 \
  --visualize_samples 5
```

```bash
PYTHONPATH=. python eval/evaluate_model.py \
  --checkpoint_path experiments/latent_vit_d6_h8_lr0.0001_bs64_ep60/latent_vit_d6_h8_lr0.0001_bs64_ep60_20251104_113924/checkpoints/best_model.pt \
  --latent_test_dir latents/test \
  --output_dir eval_results \
  --batch_size 32 \
  --visualize_samples 5
```

### スモークテスト評価

```bash
PYTHONPATH=. python eval/evaluate_model.py \
  --checkpoint_path experiments/latent_vit_d6_h8_lr0.0001_bs8_ep2/*/checkpoints/best_model.pt \
  --latent_test_dir latents/test_smoke \
  --output_dir eval_results_smoke \
  --batch_size 4 \
  --visualize_samples 3
```

### 生成される出力ファイル

```
eval_results/
├── confusion_matrix.png           # 混同行列（正規化版・生データ版）
├── class_metrics.png               # クラス別メトリクス
├── prediction_confidence.png       # 予測信頼度分布
├── attention_sample_0.png          # Attention可視化（サンプル0）
├── attention_sample_1.png          # Attention可視化（サンプル1）
├── ...
└── evaluation_results.json         # 数値結果サマリ
```

### 結果の確認

```bash
# 全体精度
cat eval_results/evaluation_results.json | jq '.accuracy'

# クラス別F1スコア
cat eval_results/evaluation_results.json | jq '.classification_report'

# 見やすく整形
cat eval_results/evaluation_results.json | jq '.'
```

### Attention可視化の解釈

正常に学習されたモデルの場合：

**上段グラフ（CLS Token類似度）**:
- バーの高さにバラつきがある
- 特定の層（例: Layer 4-8）が高い重要度を示す
- StyleGANの中間層が感情認識に重要

**下段ヒートマップ（Token間類似度）**:
- 対角線付近が明るい（自己類似度=1.0）
- 遠い層同士は暗い（異なる特徴を捉えている）
- グラデーションのあるパターン

⚠️ **注意**: スモークテストでは学習不足により、すべての値が高くなり意味のあるパターンが見られません。

---

## 実験管理

### 実験ディレクトリ構造

```
experiments/
└── latent_vit_d6_h8_lr0.0001_bs64_ep60_20251027_120000/
    ├── config.json                 # 実験設定
    ├── experiment_summary.json     # 最終結果サマリ
    ├── checkpoints/
    │   ├── best_model.pt          # ベストモデル（F1最高）
    │   ├── epoch_1.pt
    │   ├── epoch_2.pt
    │   └── ...
    └── logs/                       # TensorBoardログ
        └── events.out.tfevents.*
```

### 実験比較

```bash
# 複数実験の結果を比較
python -c "
import json
import glob

experiments = glob.glob('experiments/*/experiment_summary.json')
for exp in sorted(experiments):
    with open(exp) as f:
        data = json.load(f)
        name = data['experiment_name']
        acc = data['final_metrics'].get('accuracy', 0)
        f1 = data['final_metrics'].get('f1_macro', 0)
        print(f'{name}: Acc={acc:.4f}, F1={f1:.4f}')
"
```

### チェックポイント管理

```bash
# ベストモデルの確認
find experiments -name "best_model.pt" -exec ls -lh {} \;

# 特定実験のチェックポイント一覧
ls -lht experiments/latent_vit_*/checkpoints/
```

---

## トラブルシューティング

### 1. メモリ不足（OOM）

**症状**: `RuntimeError: CUDA out of memory`

**解決策**:
```bash
# バッチサイズを小さく
--batch_size 16  # または 8, 4

# 潜在コード生成時も同様
--batch_size 2
```

### 2. エンコーダの読み込みエラー

**症状**: `ModuleNotFoundError: No module named 'pixel2style2pixel'`

**解決策**:
```bash
# PYTHONPATHを設定
export PYTHONPATH=/path/to/fer-vit:/path/to/fer-vit/third_party/pixel2style2pixel

# または各コマンドで指定
PYTHONPATH=. python scripts/generate_latents.py ...
```

### 3. paths_config.py未設定

**症状**: `FileNotFoundError: StyleGAN weights not found`

**解決策**:
```bash
# paths_config.pyを編集
vim third_party/pixel2style2pixel/configs/paths_config.py

# 必要な重みをダウンロード
# - stylegan2-ffhq-config-f.pt
# - model_ir_se50.pth
```

### 4. PyTorch 2.6互換性エラー

**症状**: `_pickle.UnpicklingError: Weights only load failed`

**解決策**:
```python
# eval/evaluate_model.py の修正（34行目付近）
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
```

### 5. seaborn/matplotlib不足

**症状**: `Warning: seaborn not available`

**解決策**:
```bash
conda install -c conda-forge seaborn matplotlib
```

### 6. 潜在コードの形状不一致

**症状**: `RuntimeError: size mismatch`

**解決策**:
```bash
# 潜在コードの形状を確認
python -c "
import torch
data = torch.load('latents/train/angry_000.pt')
print(data['latent'].shape)
"
# 期待: torch.Size([18, 512])

# seq_lenを自動推定（デフォルト動作）
--seq_len 0  # または指定なし
```

### 7. クラス不均衡

**症状**: 特定クラスの精度が著しく低い

**解決策**:
```bash
# クラス重み付けを有効化
--use_class_weights

# または訓練データを均衡化
```

---

## ベストプラクティス

### 1. 段階的な実験

```bash
# ステップ1: スモークテスト（数分）
# 少量データで動作確認

# ステップ2: 小規模実験（30分-1時間）
# 各クラス100-200枚、10-20エポック

# ステップ3: 本番実験（数時間）
# 全データ、60-100エポック
```

### 2. ハイパーパラメータチューニング

優先順位：
1. **学習率**: 1e-3, 1e-4, 5e-5を試す
2. **バッチサイズ**: GPUメモリの限界まで大きく
3. **エポック数**: Early stoppingで自動調整
4. **モデルサイズ**: depth/headsを調整

### 3. 実験記録

```bash
# 実験ノート作成
echo "Experiment: $(date)" >> experiments.log
echo "Config: depth=6, lr=1e-4" >> experiments.log
echo "Result: Acc=0.65, F1=0.62" >> experiments.log
```

### 4. 再現性確保

```bash
# シードを固定
--seed 42

# 環境情報を記録
conda list > environment_used.txt
pip freeze > requirements_used.txt
```

---

## 推奨実験設定

### 初期実験（ベースライン）

```bash
# 標準設定でベースライン確立
PYTHONPATH=. python train/train_latent_vit.py \
  --latent_train_dir latents/train \
  --latent_val_dir latents/val \
  --epochs 60 \
  --batch_size 64 \
  --lr 1e-4 \
  --use_class_weights \
  --scheduler plateau \
  --depth 6 \
  --heads 8
```

### 深いモデル実験

```bash
# Transformerを深く
PYTHONPATH=. python train/train_latent_vit.py \
  --latent_train_dir latents/train \
  --latent_val_dir latents/val \
  --epochs 80 \
  --batch_size 32 \
  --lr 5e-5 \
  --use_class_weights \
  --scheduler cosine \
  --depth 12 \
  --heads 16 \
  --embed_dim 768
```

### 高学習率実験

```bash
# 学習率を高く設定
PYTHONPATH=. python train/train_latent_vit.py \
  --latent_train_dir latents/train \
  --latent_val_dir latents/val \
  --epochs 40 \
  --batch_size 64 \
  --lr 5e-4 \
  --use_class_weights \
  --scheduler cosine
```

---

## 参考情報

### 関連ファイル

- `README_LATENT_VIT.md`: プロジェクト全体の概要
- `RUN_CHECKLIST.md`: 実行手順チェックリスト
- `requirements.txt`: Python依存関係
- `environment.yml`: Conda環境定義

### 論文・リポジトリ

- **StyleGAN2**: https://github.com/NVlabs/stylegan2
- **pSp**: https://github.com/eladrich/pixel2style2pixel
- **e4e**: https://github.com/omertov/encoder4editing
- **Vision Transformer**: https://arxiv.org/abs/2010.11929
- **FER2013**: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

---

## よくある質問（FAQ）

### Q1: 潜在コード生成に時間がかかりすぎる

A: バッチサイズを増やすか、GPUを使用してください。CPUでは非常に遅いです。

### Q2: 精度が上がらない

A: 以下を確認：
- クラス重み付け（`--use_class_weights`）
- 学習率の調整
- エポック数の増加
- データ前処理（顔アライメント）

### Q3: Attentionが均一になる

A: 学習不足の可能性。エポック数を増やすか、データ量を増やしてください。

### Q4: 異なるエンコーダ（e4e）を使いたい

A: `--encoder_type e4e` を指定し、対応するチェックポイントを使用してください。

### Q5: 学習を途中から再開したい

A: 現在未実装。将来のバージョンで対応予定。

---

## 更新履歴

- **2025-01-27**: 初版作成
  - 基本的な実験フローを記載
  - スモークテスト手順追加
  - PyTorch 2.6互換性対応

---

## ライセンス

本プロジェクトは研究目的で作成されています。使用する外部ライブラリ（pSp, e4e等）のライセンスに従ってください。