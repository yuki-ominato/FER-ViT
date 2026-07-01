CUBLAS_WORKSPACE_CONFIG=:16:8

# 実行コマンド集

すべてのコマンドは `fer-vit/` をカレントディレクトリとして実行する。

---

## 1. image → latent 変換

`data/generate_latents.py` を使い、画像を pSp / e4e でエンコードして `.pt` ファイルとして保存する。

### FER2013（train）

```bash
python data/generate_latents.py \
  --dataset_type fer2013 \
  --data_root ../dataset/fer2013/train \
  --latent_out latents/train \
  --encoder_model pretrained_models/psp_ffhq_encode.pt \
  --encoder_type psp \
  --batch_size 4
```

### FER2013（val / test）

```bash
python data/generate_latents.py \
  --dataset_type fer2013 \
  --data_root ../dataset/fer2013/val \
  --latent_out latents/val \
  --encoder_model pretrained_models/psp_ffhq_encode.pt \
  --encoder_type psp \
  --batch_size 4

python data/generate_latents.py \
  --dataset_type fer2013 \
  --data_root ../dataset/fer2013/test \
  --latent_out latents/test \
  --encoder_model pretrained_models/psp_ffhq_encode.pt \
  --encoder_type psp \
  --batch_size 4
```

### RAF-DB（train / test）

RAF-DB は `--data_root` にデータセットルートを指定し、`--split` で分割を切り替える。

```bash
python data/generate_latents.py \
  --dataset_type raf-db \
  --data_root ../dataset/RAF-DB \
  --split train \
  --latent_out latents/RAF-DB/train \
  --encoder_model pretrained_models/psp_ffhq_encode.pt \
  --encoder_type psp \
  --batch_size 4

python data/generate_latents.py \
  --dataset_type raf-db \
  --data_root ../dataset/RAF-DB \
  --split test \
  --latent_out latents/RAF-DB/test \
  --encoder_model pretrained_models/psp_ffhq_encode.pt \
  --encoder_type psp \
  --batch_size 4
```

### e4e エンコーダを使う場合

`--encoder_model` と `--encoder_type` を変更するだけでよい。

```bash
python data/generate_latents.py \
  --dataset_type fer2013 \
  --data_root ../dataset/fer2013/train \
  --latent_out latents/train_e4e \
  --encoder_model pretrained_models/e4e_ffhq_encode.pt \
  --encoder_type e4e \
  --batch_size 4
```

---

## 2. StyleExtractor の訓練（AFS 原論文損失）

`train/train_style_extractor.py` を使う。  
事前に latent 変換済みの `.pt` ファイルが必要。

### 基本（案B: DiskImageProvider）

```bash
python train/train_style_extractor.py \
  --latent_dir    latents/train \
  --val_latent_dir latents/val \
  --psp_path      pretrained_models/psp_ffhq_encode.pt \
  --arcface_path  pretrained_models/model_ir_se50.pth \
  --out_dir       outputs/afs \
  --provider      b \
  --img_root      ../dataset/fer2013/train \
  --val_img_root  ../dataset/fer2013/val \
  --epochs        10 \
  --batch_size    4 \
  --lr            1e-4
```

### 案A: GeneratedImageProvider（Generator でリアルタイム生成）

```bash
python train/train_style_extractor.py \
  --latent_dir    latents/rafdb_e4e/train \
  --val_latent_dir latents/rafdb_e4e/test \
  --psp_path      pretrained_models/e4e_ffhq_encode.pt\
  --arcface_path  pretrained_models/model_ir_se50.pth \
  --out_dir       outputs/afs/rafdb-a \
  --provider      a \
  --epochs        10 \
  --batch_size    4
```

チェックポイントは `outputs/afs/<YYYYMMDD_HHMMSS>/checkpoints/` 以下に保存される。  
- `best_model.pt` — val loss（指定なければ train loss）が改善したエポック

---

## 3. FER特化 StyleExtractor の訓練（AFSFERLoss）

`train/train_fer_extractor.py` を使う。  
AFS 原論文の損失を FER タスク向けに再設計したバリアント。

| 損失 | 内容 | ジェネレータ必要 |
|------|------|:---:|
| `L_expr` | h(w) が感情ラベルに識別可能か | ✗ |
| `L_neutral` | w−h(w) が無表情(4)に識別可能か | ✗ |
| `L_id` | ArcFace でアイデンティティ保存 | ✓ |
| `L_sparse` | 非表情 W+ 層(0-3, 12-17)をゼロに近づける | ✗ |
| `L_cons` | h(w_new) ≈ h(w_tgt) の一貫性 | ✗ |

### 基本（ジェネレータあり、L_id 有効）

```bash
python train/train_fer_extractor.py \
  --latent_dir     latents/fer2013/train \
  --val_latent_dir latents/fer2013/val \
  --psp_path       pretrained_models/e4e_ffhq_encode.pt \
  --arcface_path   pretrained_models/model_ir_se50.pth \
  --out_dir        outputs/afs_fer \
  --epochs         10 \
  --batch_size     4
```

### 高速版（ジェネレータなし、L_id = 0）

```bash
python train/train_fer_extractor.py \
  --latent_dir     latents/fer2013/train \
  --val_latent_dir latents/fer2013/val \
  --psp_path       pretrained_models/e4e_ffhq_encode.pt \
  --arcface_path   pretrained_models/model_ir_se50.pth \
  --out_dir        outputs/afs_fer \
  --no_generator \
  --epochs         10 \
  --batch_size     16
```

### 損失係数の調整例

```bash
python train/train_fer_extractor.py \
  --latent_dir     latents/fer2013/train \
  --val_latent_dir latents/fer2013/val \
  --psp_path       pretrained_models/e4e_ffhq_encode.pt \
  --arcface_path   pretrained_models/model_ir_se50.pth \
  --out_dir        outputs/afs_fer \
  --no_generator \
  --lambda_expr    1.0 \
  --lambda_neutral 0.5 \
  --lambda_sparse  0.02 \
  --lambda_cons    0.1 \
  --epochs         10 \
  --batch_size     16
```

チェックポイントは `outputs/afs_fer/<YYYYMMDD_HHMMSS>/checkpoints/` 以下に保存される。  
- `best_model.pt` の `'model_state'` に StyleExtractor h の重み
- `best_model.pt` の `'classifier_state'` に ExprClassifier の重み（単体利用も可能）  
- `last_model.pt` — 毎エポック上書き（学習再開用）

---

## 3. ImageViT の訓練

`train/train_image_vit.py` を使う。画像を直接入力する ViT モデル。

### FER2013 / スクラッチ学習

```bash
CUBLAS_WORKSPACE_CONFIG=:16:8 python train/train_image_vit.py \
  --dataset fer2013 \
  --train_dir ../dataset/fer2013/train \
  --val_dir   ../dataset/fer2013/val \
  --model_size small \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-3 \
  --scheduler warmup_cosine \
  --use_augmentation \
  --use_class_weights \
  --label_smoothing 0.1
```

### FER2013 / ImageNet 事前学習あり

```bash
CUBLAS_WORKSPACE_CONFIG=:16:8 python train/train_image_vit.py \
  --dataset fer2013 \
  --train_dir ../dataset/fer2013/train \
  --val_dir   ../dataset/fer2013/val \
  --use_pretrained \
  --epochs 50 \
  --batch_size 64 \
  --lr 1e-4 \
  --scheduler warmup_cosine \
  --use_augmentation \
  --use_class_weights \
  --label_smoothing 0.1
```

### FER2013 / カスタムアーキテクチャ

```bash
CUBLAS_WORKSPACE_CONFIG=:16:8 python train/train_image_vit.py \
  --dataset fer2013 \
  --train_dir ../dataset/fer2013/train \
  --val_dir   ../dataset/fer2013/val \
  --model_size custom \
  --patch_size 16 \
  --embed_dim 512 \
  --depth 6 \
  --heads 8 \
  --mlp_dim 2048 \
  --dropout 0.1 \
  --epochs 50 \
  --batch_size 64 \
  --lr 1e-4 \
  --weight_decay 0.05 \
  --optimizer adamw \
  --scheduler warmup_cosine \
  --label_smoothing 0.1 \
  --use_class_weights \
  --seed 42
```

### RAF-DB

`--train_dir` と `--val_dir` には**データセットルートを同じパスで指定**する。  
内部で train / test split が自動的に使われる。

```bash
CUBLAS_WORKSPACE_CONFIG=:16:8 python train/train_image_vit.py \
  --dataset raf-db \
  --train_dir ../dataset/RAF-DB \
  --val_dir   ../dataset/RAF-DB \
  --model_size small \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-3 \
  --scheduler warmup_cosine \
  --use_augmentation \
  --use_class_weights \
  --label_smoothing 0.1
```

### RAF-DB / ImageNet 事前学習あり・カスタムアーキテクチャ

```bash
CUBLAS_WORKSPACE_CONFIG=:16:8 python train/train_image_vit.py \
  --dataset raf-db \
  --train_dir ../dataset/RAF-DB \
  --val_dir   ../dataset/RAF-DB \
  --img_size 224 \
  --model_size custom \
  --patch_size 16 \
  --embed_dim 512 \
  --depth 6 \
  --heads 8 \
  --mlp_dim 2048 \
  --num_classes 7 \
  --dropout 0.1 \
  --use_pretrained \
  --epochs 50 \
  --batch_size 64 \
  --lr 0.0001 \
  --weight_decay 0.05 \
  --optimizer adamw \
  --scheduler warmup_cosine \
  --label_smoothing 0.1 \
  --use_class_weights \
  --seed 42
```

---

## 4. StyleExtractor の事前適用（オプション）

`train_latent_vit_afs.py` の代わりに、変換済み `.pt` を事前に作っておく方法。  
エポックをまたいで何度も StyleExtractor を呼ばずに済むため、訓練速度が向上する。

### スタイル成分（w_sty = h(w)）を保存

```bash
python data/extract_style_latents.py \
  --latent_dir           latents/fer2013_e4e/train \
  --out_dir              latents/fer2013_e4e/train_sty_fer \
  --style_extractor_path outputs/afs_fer/20260701_155645/checkpoints/best_model.pt \
  --batch_size           256

python data/extract_style_latents.py \
  --latent_dir           latents/val \
  --out_dir              latents/val_sty \
  --style_extractor_path outputs/afs/<run_id>/checkpoints/best_model.pt \
  --batch_size           256
```

### アイデンティティ成分（w_id = w − h(w)）を保存

```bash
python data/extract_style_latents.py \
  --latent_dir           latents/train \
  --out_dir              latents/train_id \
  --style_extractor_path outputs/afs/<run_id>/checkpoints/best_model.pt \
  --mode                 identity \
  --batch_size           256
```

### 両方まとめて保存（--out_dir/style/ と --out_dir/identity/ に分けて出力）

```bash
python data/extract_style_latents.py \
  --latent_dir           latents/fer2013/train \
  --out_dir              latents/fer2013/train_afs \
  --style_extractor_path experiments/extractor/fer2013/e4e/20260701_000937/checkpoints/best_model.pt \
  --mode                 both \
  --batch_size           256
```

変換後は `train_latent_vit.py` に変換済みディレクトリを渡すだけでよい:

```bash
CUBLAS_WORKSPACE_CONFIG=:16:8 python train/train_latent_vit.py \
  --latent_train_dir latents/fer2013_e4e/train_sty \
  --latent_val_dir   latents/fer2013_e4e/val_sty \
  --epochs 60 --batch_size 64
```

---

## 6. LatentViT の訓練

`train/train_latent_vit.py` を使う。latent コードを入力とする ViT モデル。  
事前に latent 変換（手順 1）が必要。

### 基本設定

```bash
CUBLAS_WORKSPACE_CONFIG=:16:8 python train/train_latent_vit.py \
  --latent_train_dir latents/train \
  --latent_val_dir   latents/val \
  --epochs 60 \
  --batch_size 64 \
  --lr 1e-4 \
  --scheduler plateau \
  --use_class_weights \
```

### カスタムアーキテクチャ

```bash
CUBLAS_WORKSPACE_CONFIG=:16:8 python train/train_latent_vit.py \
  --latent_train_dir latents/train \
  --latent_val_dir   latents/val \
  --latent_dim 512 \
  --seq_len 18 \
  --embed_dim 512 \
  --depth 6 \
  --heads 8 \
  --mlp_dim 2048 \
  --dropout 0.1 \
  --epochs 60 \
  --batch_size 64 \
  --lr 1e-4 \
  --weight_decay 1e-2 \
  --scheduler plateau \
  --use_class_weights \
  --label_smoothing 0.1 \
  --seed 42
```

### AFS StyleExtractor を組み合わせた訓練（train_latent_vit_afs.py）

```bash
CUBLAS_WORKSPACE_CONFIG=:16:8 python train/train_latent_vit_afs.py \
  --latent_train_dir    latents/train \
  --latent_val_dir      latents/val \
  --style_extractor_path outputs/afs/<run_id>/checkpoints/best_model.pt \
  --embed_dim 512 \
  --depth 6 \
  --heads 8 \
  --mlp_dim 2048 \
  --epochs 60 \
  --batch_size 64 \
  --lr 1e-4 \
  --scheduler plateau \
  --use_class_weights \
  --label_smoothing 0.1
```

# 実行コマンド集

## train_image_cnn.py（2D Image CNN）

### FER2013 × スクラッチ

```bash
python train/train_image_cnn.py \
    --dataset fer2013 \
    --train_dir ../dataset/fer2013/train \
    --val_dir   ../dataset/fer2013/val \
    --backbone resnet18
```

### FER2013 × ImageNet 事前学習あり

```bash
python train/train_image_cnn.py \
    --dataset fer2013 \
    --train_dir /path/to/fer2013/train \
    --val_dir   /path/to/fer2013/val \
    --backbone resnet18 \
    --use_pretrained
```

### RAF-DB × スクラッチ

```bash
python train/train_image_cnn.py \
    --dataset raf-db \
    --train_dir /home/yuki/research2/dataset/RAF-DB \
    --val_dir   /home/yuki/research2/dataset/RAF-DB \
    --backbone resnet18
```

### RAF-DB × ImageNet 事前学習あり

```bash
python train/train_image_cnn.py \
    --dataset raf-db \
    --train_dir /home/yuki/research2/dataset/RAF-DB \
    --val_dir   /home/yuki/research2/dataset/RAF-DB \
    --backbone resnet18 \
    --use_pretrained
```

---

## evaluate_image_cnn.py（評価）

```bash
python eval/evaluate_image_cnn.py \
    --checkpoint_path experiments/<実験名>/<タイムスタンプ>/checkpoints/best_model.pt \
    --dataset fer2013 \
    --test_dir /path/to/fer2013/test \
    --output_dir eval_results/image_cnn
```

RAF-DB の場合:

```bash
python eval/evaluate_image_cnn.py \
    --checkpoint_path experiments/<実験名>/<タイムスタンプ>/checkpoints/best_model.pt \
    --dataset raf-db \
    --test_dir /home/yuki/research2/dataset/RAF-DB \
    --output_dir eval_results/image_cnn_rafdb
```


---

## 7. InterFaceGAN SVM 感情部分空間分離

`latent_analysis/` 以下のスクリプトを順に実行する。  
すべて `fer-vit/` をカレントディレクトリとして実行する。

### ① SVM 学習（train のみ使用）

```bash
python latent_analysis/train_svm.py \
    --latent_dir latents/fer2013/train \
    --output_dir latent_analysis/svm_output_fer2013
```

### ② 感情基底 N の構築

```bash
python latent_analysis/build_svm_projection.py \
    --svm_model  latent_analysis/svm_output_fer2013/svm_model.joblib \
    --output_dir latent_analysis/svm_output_fer2013
```

### ③ ViT 訓練（3条件）

#### Baseline（射影なし・比較用）

```bash
python train/train_latent_vit_v2.py \
    --latent_train_dir latents/fer2013/train \
    --latent_val_dir   latents/fer2013/val \
    --experiment_name  svm_baseline
```

#### Emotion Only（感情成分のみ）

```bash
python train/train_latent_vit_v2.py \
    --latent_train_dir latents/fer2013/train \
    --latent_val_dir   latents/fer2013/val \
    --svm_basis        latent_analysis/svm_output_fer2013/emotion_basis_N.pt \
    --svm_projection   emotion \
    --experiment_name  svm_emotion_only
```

#### Residual Only（非感情成分のみ）

```bash
python train/train_latent_vit_v2.py \
    --latent_train_dir latents/fer2013/train \
    --latent_val_dir   latents/fer2013/val \
    --svm_basis        latent_analysis/svm_output_fer2013/emotion_basis_N.pt \
    --svm_projection   residual \
    --experiment_name  svm_residual_only
```

---

## 補足

| 項目 | パス |
|------|------|
| FER2013 train | `../dataset/fer2013/train` |
| FER2013 val | `../dataset/fer2013/val` |
| FER2013 test | `../dataset/fer2013/test` |
| RAF-DB ルート | `../dataset/RAF-DB` |
| pSp チェックポイント | `pretrained_models/psp_ffhq_encode.pt` |
| e4e チェックポイント | `pretrained_models/e4e_ffhq_encode.pt` |
| ArcFace モデル | `pretrained_models/model_ir_se50.pth` |
| latent キャッシュ (FER2013) | `latents/train`, `latents/val`, `latents/test` |
| 実験ログ | `experiments/` |
| StyleExtractor 出力 | `outputs/afs/<run_id>/` |
