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

## 2. StyleExtractor の訓練

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
  --latent_dir    latents/train \
  --val_latent_dir latents/val \
  --psp_path      pretrained_models/psp_ffhq_encode.pt \
  --arcface_path  pretrained_models/model_ir_se50.pth \
  --out_dir       outputs/afs \
  --provider      a \
  --epochs        10 \
  --batch_size    4
```

チェックポイントは `outputs/afs/<YYYYMMDD_HHMMSS>/checkpoints/` 以下に保存される。  
- `best_model.pt` — val loss（指定なければ train loss）が改善したエポック  
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

## 4. LatentViT の訓練

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
  --label_smoothing 0.1
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
