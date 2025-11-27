#!/bin/bash
# データセット削減実験スクリプト
# Few-Shot Learning性能を検証

echo "============================================"
echo "Data Fraction Experiments"
echo "データ効率性の検証実験"
echo "============================================"

# 実験設定
LATENT_TRAIN="latents/train"
LATENT_VAL="latents/val"
IMAGE_TRAIN="dataset/fer2013/train"
IMAGE_VAL="dataset/fer2013/val"

# データ削減率の設定
FRACTIONS=(0.1 0.25 0.5 0.75 1.0)

# 複数シードで実行（統計的信頼性向上）
#SEEDS=(42 123 456)
SEEDS=(42)

echo ""
echo "============================================"
echo "実験1: LatentViT（スクラッチ）のデータ効率性"
echo "============================================"

for fraction in "${FRACTIONS[@]}"; do
    echo ""
    echo "--- データ割合: ${fraction} ($(echo "$fraction * 100" | bc)%) ---"
    
    for seed in "${SEEDS[@]}"; do
        echo "Seed: ${seed}"
        python train/train_latent_vit.py \
            --latent_train_dir "$LATENT_TRAIN" \
            --latent_val_dir "$LATENT_VAL" \
            --data_fraction "$fraction" \
            --epochs 60 \
            --batch_size 64 \
            --lr 1e-4 \
            --use_class_weights \
            --seed "$seed"
    done
done

# echo ""
# echo "============================================"
# echo "実験2: HybridLatentViT（事前学習）のデータ効率性"
# echo "============================================"

# for fraction in "${FRACTIONS[@]}"; do
#     echo ""
#     echo "--- データ割合: ${fraction} ($(echo "$fraction * 100" | bc)%) ---"
    
#     for seed in "${SEEDS[@]}"; do
#         echo "Seed: ${seed}"
#         python train/train_hybrid_latent_vit_with_fraction.py \
#             --latent_train_dir "$LATENT_TRAIN" \
#             --latent_val_dir "$LATENT_VAL" \
#             --data_fraction "$fraction" \
#             --model_size small \
#             --use_pretrained \
#             --epochs 60 \
#             --batch_size 64 \
#             --lr 1e-4 \
#             --use_class_weights \
#             --use_layerwise_lr \
#             --seed "$seed"
#     done
# done

# echo ""
# echo "============================================"
# echo "実験3: 標準ImageViT（ベースライン）のデータ効率性"
# echo "============================================"

# for fraction in "${FRACTIONS[@]}"; do
#     echo ""
#     echo "--- データ割合: ${fraction} ($(echo "$fraction * 100" | bc)%) ---"
    
#     for seed in "${SEEDS[@]}"; do
#         echo "Seed: ${seed}"
#         python train/train_image_vit_with_fraction.py \
#             --train_dir "$IMAGE_TRAIN" \
#             --val_dir "$IMAGE_VAL" \
#             --data_fraction "$fraction" \
#             --model_size small \
#             --use_augmentation \
#             --epochs 100 \
#             --batch_size 32 \
#             --lr 1e-3 \
#             --use_class_weights \
#             --seed "$seed"
#     done
# done

# echo ""
# echo "============================================"
# echo "実験4: Adapter Fine-tuning のデータ効率性"
# echo "（最も効率的な手法の検証）"
# echo "============================================"

# for fraction in "${FRACTIONS[@]}"; do
#     echo ""
#     echo "--- データ割合: ${fraction} ($(echo "$fraction * 100" | bc)%) ---"
    
#     for seed in "${SEEDS[@]}"; do
#         echo "Seed: ${seed}"
#         python train/train_hybrid_latent_vit_with_fraction.py \
#             --latent_train_dir "$LATENT_TRAIN" \
#             --latent_val_dir "$LATENT_VAL" \
#             --data_fraction "$fraction" \
#             --model_size small \
#             --use_pretrained \
#             --freeze_transformer \
#             --use_adapter \
#             --adapter_dim 64 \
#             --epochs 60 \
#             --batch_size 128 \
#             --lr 1e-3 \
#             --use_class_weights \
#             --seed "$seed"
#     done
# done

echo ""
echo "============================================"
echo "すべての実験完了"
echo "============================================"

# 結果の集計スクリプトを実行
echo ""
echo "結果を集計中..."
python scripts/analyze_data_fraction_results.py --experiments_dir experiments/

echo ""
echo "実験結果の可視化..."
python scripts/plot_data_efficiency.py --experiments_dir experiments/ --output_dir figures/

echo ""
echo "完了！"
echo "結果: figures/data_efficiency_comparison.png"