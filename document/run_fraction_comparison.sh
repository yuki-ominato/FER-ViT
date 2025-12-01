#!/bin/bash

# データセットパス設定
LATENT_TRAIN="latents/train"
LATENT_VAL="latents/val"
IMAGE_TRAIN="dataset/fer2013/train"
IMAGE_VAL="dataset/fer2013/val"

# 共通ハイパーパラメータ (公平な比較のため統一)
# LatentViTのデフォルト(Small)に合わせる
DEPTH=6
HEADS=8
EMBED_DIM=512
MLP_DIM=2048
EPOCHS=60
BATCH_SIZE=64
LR=1e-4

# データ割合リスト
FRACTIONS=(0.1 0.25 0.5 1.0)

echo "============================================"
echo "Starting Fair Data Efficiency Comparison"
echo "Common Config: Depth=$DEPTH, Embed=$EMBED_DIM, Heads=$HEADS"
echo "============================================"

for fraction in "${FRACTIONS[@]}"; do
    echo ""
    echo ">>> Testing with Data Fraction: ${fraction}"
    
    # 1. 提案手法 (Latent ViT)
    echo "--- Training Proposed Method (Latent ViT) ---"
    python train/train_latent_vit.py \
        --latent_train_dir "$LATENT_TRAIN" \
        --latent_val_dir "$LATENT_VAL" \
        --data_fraction "$fraction" \
        --depth $DEPTH \
        --heads $HEADS \
        --embed_dim $EMBED_DIM \
        --mlp_dim $MLP_DIM \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --use_class_weights \
        --seed 42

    # 2. 従来手法 (Image ViT - Scratch)
    # create_vit_small等は使わず、カスタムサイズを指定して構造を統一
    echo "--- Training Baseline Method (Image ViT) ---"
    python train/train_image_vit.py \
        --train_dir "$IMAGE_TRAIN" \
        --val_dir "$IMAGE_VAL" \
        --data_fraction "$fraction" \
        --model_size custom \
        --depth $DEPTH \
        --heads $HEADS \
        --embed_dim $EMBED_DIM \
        --mlp_dim $MLP_DIM \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --use_class_weights \
        --use_augmentation \
        --seed 42
done

# echo ""
# echo "============================================"
# echo "Experiments Completed. Generating Plots..."
# echo "============================================"

# python scripts/plot_data_efficiency.py \
#     --experiments_dir experiments \
#     --output_dir figures

# echo "Done! Check figures/ folder."