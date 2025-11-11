#!/bin/bash
# ハイブリッドViT（事前学習Transformer + StyleGAN潜在コード）の実験

# ==============================
# 必要なライブラリのインストール
# ==============================
# pip install timm torch torchvision

# ==============================
# 実験1: 完全ファインチューニング（推奨）
# 事前学習TransformerをStyleGAN潜在空間に適応
# ==============================
python train/train_hybrid_latent_vit.py \
    --latent_train_dir ../dataset/latents/train \
    --latent_val_dir ../dataset/latents/val \
    --model_size small \
    --use_pretrained \
    --epochs 60 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --scheduler plateau \
    --use_class_weights \
    --use_layerwise_lr \
    --seed 42

# ==============================
# 実験2: Adapter-based Fine-tuning
# Parameter-Efficient: Transformerを凍結、Adapterのみ学習
# メモリ効率的、学習時間短縮
# ==============================
python train/train_hybrid_latent_vit.py \
    --latent_train_dir ../dataset/latents/train \
    --latent_val_dir ../dataset/latents/val \
    --model_size small \
    --use_pretrained \
    --freeze_transformer \
    --use_adapter \
    --adapter_dim 64 \
    --epochs 60 \
    --batch_size 128 \
    --lr 1e-3 \
    --use_class_weights

# ==============================
# 実験3: Linear Probing（ベースライン）
# Transformerを完全凍結、ヘッドのみ学習
# ==============================
python train/train_hybrid_latent_vit.py \
    --latent_train_dir data/latents/train \
    --latent_val_dir data/latents/val \
    --model_size small \
    --use_pretrained \
    --freeze_transformer \
    --epochs 30 \
    --batch_size 128 \
    --lr 1e-3 \
    --use_class_weights

# ==============================
# 実験4: 部分凍結
# 下位6層を凍結、上位6層のみ学習
# ==============================
python train/train_hybrid_latent_vit.py \
    --latent_train_dir data/latents/train \
    --latent_val_dir data/latents/val \
    --model_size small \
    --use_pretrained \
    --freeze_stages 6 \
    --epochs 60 \
    --batch_size 64 \
    --lr 3e-4 \
    --use_class_weights \
    --use_layerwise_lr

# ==============================
# 実験5: スクラッチ学習（比較用）
# 事前学習なし、完全にランダム初期化
# ==============================
python train/train_latent_vit.py \
    --latent_train_dir data/latents/train \
    --latent_val_dir data/latents/val \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-3 \
    --use_class_weights

# ==============================
# モデルサイズ別の実験
# ==============================

# Tiny: 高速、軽量
python train/train_hybrid_latent_vit.py \
    --latent_train_dir data/latents/train \
    --latent_val_dir data/latents/val \
    --model_size tiny \
    --use_pretrained \
    --batch_size 128 \
    --lr 5e-4

# Base: 高精度、大規模
python train/train_hybrid_latent_vit.py \
    --latent_train_dir data/latents/train \
    --latent_val_dir data/latents/val \
    --model_size base \
    --use_pretrained \
    --batch_size 32 \
    --lr 5e-5 \
    --use_layerwise_lr

# ==============================
# 包括的な比較実験
# 3つの異なるアプローチを比較
# ==============================

echo "============================================"
echo "Comprehensive Comparison Experiments"
echo "============================================"

# 1. StyleGAN潜在空間 + スクラッチViT（従来）
echo "Experiment 1: Latent + Scratch ViT"
python train/train_latent_vit.py \
    --latent_train_dir data/latents/train \
    --latent_val_dir data/latents/val \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-3

# 2. StyleGAN潜在空間 + 事前学習Transformer（提案）
echo "Experiment 2: Latent + Pretrained Transformer (PROPOSED)"
python train/train_hybrid_latent_vit.py \
    --latent_train_dir data/latents/train \
    --latent_val_dir data/latents/val \
    --model_size small \
    --use_pretrained \
    --epochs 60 \
    --batch_size 64 \
    --lr 1e-4 \
    --use_layerwise_lr

# 3. 画像 + 事前学習ViT（ベースライン）
echo "Experiment 3: Image + Pretrained ViT"
python train/train_pretrained_vit.py \
    --train_dir data/fer2013/train \
    --val_dir data/fer2013/val \
    --model_size small \
    --use_pretrained \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --use_augmentation

# ==============================
# 評価
# ==============================

# ハイブリッドモデルの評価
python eval/evaluate_model.py \
    --checkpoint_path experiments/hybrid_vit_<n>/checkpoints/best_model.pt \
    --latent_test_dir data/latents/test \
    --output_dir eval_results/hybrid_vit

# ==============================
# パラメータ数と学習時間の比較
# ==============================
python -c "
from models_fer_vit.hybrid_latent_vit import create_hybrid_latent_vit
from models_fer_vit.latent_vit import LatentViT

print('='*60)
print('Model Comparison: Parameters')
print('='*60)

# スクラッチViT
scratch = LatentViT(embed_dim=512, depth=6, heads=8)
scratch_params = sum(p.numel() for p in scratch.parameters())
print(f'Scratch ViT: {scratch_params:,} parameters')

# ハイブリッド（Adapter）
hybrid_adapter = create_hybrid_latent_vit(
    model_size='small',
    freeze_transformer=True,
    use_adapter=True,
)
adapter_trainable = sum(p.numel() for p in hybrid_adapter.parameters() if p.requires_grad)
adapter_total = sum(p.numel() for p in hybrid_adapter.parameters())
print(f'Hybrid (Adapter): {adapter_trainable:,} / {adapter_total:,} trainable')

# ハイブリッド（完全）
hybrid_full = create_hybrid_latent_vit(
    model_size='small',
    freeze_transformer=False,
)
full_params = sum(p.numel() for p in hybrid_full.parameters())
print(f'Hybrid (Full): {full_params:,} parameters')
"