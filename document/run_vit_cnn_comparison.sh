#!/bin/bash
# ViT vs CNN 比較実験
# 同じ潜在空間入力で公平な比較

echo "============================================"
echo "ViT vs CNN Comparison Experiments"
echo "潜在空間（18, 512）を入力とする比較"
echo "============================================"

LATENT_TRAIN="latents/train"
LATENT_VAL="latents/val"

# ============================================
# Phase 1: ベースライン性能比較
# ============================================
echo ""
echo "============================================"
echo "Phase 1: Baseline Performance Comparison"
echo "============================================"

# ViT（スクラッチ）
echo ""
echo "--- ViT (Scratch) ---"
python train/train_latent_vit.py \
    --latent_train_dir "$LATENT_TRAIN" \
    --latent_val_dir "$LATENT_VAL" \
    --epochs 60 \
    --batch_size 64 \
    --lr 1e-4 \
    --use_class_weights \
    --seed 42

# CNN（標準）
echo ""
echo "--- CNN (Standard) ---"
python train/train_latent_cnn.py \
    --latent_train_dir "$LATENT_TRAIN" \
    --latent_val_dir "$LATENT_VAL" \
    --model_type standard \
    --epochs 60 \
    --batch_size 64 \
    --lr 1e-4 \
    --use_class_weights \
    --seed 42

# CNN（深層）
echo ""
echo "--- CNN (Deep) ---"
python train/train_latent_cnn.py \
    --latent_train_dir "$LATENT_TRAIN" \
    --latent_val_dir "$LATENT_VAL" \
    --model_type deep \
    --epochs 60 \
    --batch_size 64 \
    --lr 1e-4 \
    --use_class_weights \
    --seed 42

# CNN（2D）
echo ""
echo "--- CNN (2D Approach) ---"
python train/train_latent_cnn.py \
    --latent_train_dir "$LATENT_TRAIN" \
    --latent_val_dir "$LATENT_VAL" \
    --model_type 2d \
    --epochs 60 \
    --batch_size 64 \
    --lr 1e-4 \
    --use_class_weights \
    --seed 42

# ============================================
# Phase 2: パラメータ効率比較
# ============================================
echo ""
echo "============================================"
echo "Phase 2: Parameter Efficiency"
echo "============================================"

# 軽量CNN
echo ""
echo "--- CNN (Light) ---"
python train/train_latent_cnn.py \
    --latent_train_dir "$LATENT_TRAIN" \
    --latent_val_dir "$LATENT_VAL" \
    --model_type light \
    --epochs 60 \
    --batch_size 128 \
    --lr 1e-4 \
    --use_class_weights \
    --seed 42

# ViT（小型）
echo ""
echo "--- ViT (Small) ---"
python train/train_latent_vit.py \
    --latent_train_dir "$LATENT_TRAIN" \
    --latent_val_dir "$LATENT_VAL" \
    --embed_dim 384 \
    --depth 6 \
    --heads 6 \
    --mlp_dim 1536 \
    --epochs 60 \
    --batch_size 64 \
    --lr 1e-4 \
    --use_class_weights \
    --seed 42

# ============================================
# Phase 3: データ効率性比較（Few-Shot）
# ============================================
echo ""
echo "============================================"
echo "Phase 3: Data Efficiency Comparison"
echo "============================================"

for fraction in 0.1 0.25 0.5 1.0; do
    echo ""
    echo "=== データ割合: ${fraction} ($(echo "$fraction * 100" | bc)%) ==="
    
    # ViT
    echo ""
    echo "--- ViT ---"
    python train/train_latent_vit.py \
        --latent_train_dir "$LATENT_TRAIN" \
        --latent_val_dir "$LATENT_VAL" \
        --data_fraction "$fraction" \
        --epochs 60 \
        --batch_size 64 \
        --lr 1e-4 \
        --use_class_weights \
        --seed 42
    
    # CNN
    echo ""
    echo "--- CNN ---"
    python train/train_latent_cnn.py \
        --latent_train_dir "$LATENT_TRAIN" \
        --latent_val_dir "$LATENT_VAL" \
        --model_type standard \
        --data_fraction "$fraction" \
        --epochs 60 \
        --batch_size 64 \
        --lr 1e-4 \
        --use_class_weights \
        --seed 42
done

# ============================================
# Phase 4: 学習速度比較
# ============================================
echo ""
echo "============================================"
echo "Phase 4: Training Speed Comparison"
echo "============================================"

# 短期学習での収束速度を比較
for epochs in 10 20 40; do
    echo ""
    echo "=== エポック数: ${epochs} ==="
    
    # ViT
    echo "--- ViT ---"
    python train/train_latent_vit.py \
        --latent_train_dir "$LATENT_TRAIN" \
        --latent_val_dir "$LATENT_VAL" \
        --epochs "$epochs" \
        --batch_size 64 \
        --lr 1e-4 \
        --use_class_weights \
        --seed 42
    
    # CNN
    echo "--- CNN ---"
    python train/train_latent_cnn.py \
        --latent_train_dir "$LATENT_TRAIN" \
        --latent_val_dir "$LATENT_VAL" \
        --model_type standard \
        --epochs "$epochs" \
        --batch_size 64 \
        --lr 1e-4 \
        --use_class_weights \
        --seed 42
done

# ============================================
# Phase 5: 複数シードでの統計評価
# ============================================
echo ""
echo "============================================"
echo "Phase 5: Multi-seed Statistical Evaluation"
echo "============================================"

for seed in 42 123 456; do
    echo ""
    echo "=== Seed: ${seed} ==="
    
    # ViT
    echo "--- ViT ---"
    python train/train_latent_vit.py \
        --latent_train_dir "$LATENT_TRAIN" \
        --latent_val_dir "$LATENT_VAL" \
        --epochs 60 \
        --batch_size 64 \
        --lr 1e-4 \
        --use_class_weights \
        --seed "$seed"
    
    # CNN Standard
    echo "--- CNN Standard ---"
    python train/train_latent_cnn.py \
        --latent_train_dir "$LATENT_TRAIN" \
        --latent_val_dir "$LATENT_VAL" \
        --model_type standard \
        --epochs 60 \
        --batch_size 64 \
        --lr 1e-4 \
        --use_class_weights \
        --seed "$seed"
    
    # CNN Deep
    echo "--- CNN Deep ---"
    python train/train_latent_cnn.py \
        --latent_train_dir "$LATENT_TRAIN" \
        --latent_val_dir "$LATENT_VAL" \
        --model_type deep \
        --epochs 60 \
        --batch_size 64 \
        --lr 1e-4 \
        --use_class_weights \
        --seed "$seed"
done

# ============================================
# Phase 6: ハイパーパラメータ感度分析
# ============================================
echo ""
echo "============================================"
echo "Phase 6: Hyperparameter Sensitivity"
echo "============================================"

# 学習率の影響
for lr in 5e-5 1e-4 2e-4; do
    echo ""
    echo "=== Learning Rate: ${lr} ==="
    
    python train/train_latent_cnn.py \
        --latent_train_dir "$LATENT_TRAIN" \
        --latent_val_dir "$LATENT_VAL" \
        --model_type standard \
        --epochs 60 \
        --batch_size 64 \
        --lr "$lr" \
        --use_class_weights \
        --seed 42
done

# ドロップアウトの影響
for dropout in 0.1 0.3 0.5; do
    echo ""
    echo "=== Dropout: ${dropout} ==="
    
    python train/train_latent_cnn.py \
        --latent_train_dir "$LATENT_TRAIN" \
        --latent_val_dir "$LATENT_VAL" \
        --model_type standard \
        --dropout "$dropout" \
        --epochs 60 \
        --batch_size 64 \
        --lr 1e-4 \
        --use_class_weights \
        --seed 42
done

echo ""
echo "============================================"
echo "すべての比較実験完了"
echo "============================================"

# 結果の分析
echo ""
echo "結果を分析中..."
python scripts/compare_vit_cnn.py --experiments_dir experiments/

echo ""
echo "完了！"
echo "結果: experiments/"
echo "比較レポート: analysis_results/vit_vs_cnn_comparison.png"