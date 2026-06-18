# CUBLAS_WORKSPACE_CONFIG=:16:8
# E0 × 4段階
python train/train_latent_vit_v2.py \
    --latent_train_dir data/latents/train \
    --latent_val_dir   data/latents/val \
    --data_fraction 0.10 --experiment_name E0_baseline_frac10

python train/train_latent_vit_v2.py \
    --latent_train_dir data/latents/train \
    --latent_val_dir   data/latents/val \
    --data_fraction 0.25 --experiment_name E0_baseline_frac25

python train/train_latent_vit_v2.py \
    --latent_train_dir data/latents/train \
    --latent_val_dir   data/latents/val \
    --data_fraction 0.50 --experiment_name E0_baseline_frac50

# E7 × 4段階
CUBLAS_WORKSPACE_CONFIG=:16:8 python train/train_latent_vit_v2.py \
    --latent_train_dir latents/train \
    --latent_val_dir   latents/val \
    --use_lwn --use_spe --use_leam \
    --data_fraction 0.10 --experiment_name lwn-spe-leam/E7/proposed_frac10

python train/train_latent_vit_v2.py \
    --latent_train_dir data/latents/train \
    --latent_val_dir   data/latents/val \
    --use_lwn --use_spe --use_leam \
    --data_fraction 0.25 --experiment_name E7_proposed_frac25

python train/train_latent_vit_v2.py \
    --latent_train_dir data/latents/train \
    --latent_val_dir   data/latents/val \
    --use_lwn --use_spe --use_leam \
    --data_fraction 0.50 --experiment_name E7_proposed_frac50

# Step 1: 方向ベクトルの抽出
python -c "
from sefa.factorize import factorize_stylegan_weights
import numpy as np
result = factorize_stylegan_weights('path/to/stylegan.pkl', num_semantics=10)
np.save('sefa/directions.npy', result['directions'])
np.save('sefa/eigenvalues.npy', result['eigenvalues'])
"

# Step 2: 非表情方向の検証（label_change_rate <= 0.1 の方向を選ぶ）
# → verify_directions.py を呼び出すスクリプトを別途作成

# Step 3: 潜在コードの拡張（direction_indices は Step2 で選んだもの）
python -c "
from data.augment_latents import augment_latents_with_directions
import numpy as np
directions = np.load('sefa/directions.npy')
augment_latents_with_directions(
    latent_dir='data/latents/train',
    output_dir='data/latents/train_augmented',
    directions=directions,
    direction_indices=[0, 2, 4],  # 検証済みの非表情方向
)
"

# Step 4: 拡張データで E7+Aug 学習
python train/train_latent_vit_v2.py \
    --latent_train_dir data/latents/train_augmented \
    --latent_val_dir   data/latents/val \
    --use_lwn --use_spe --use_leam \
    --experiment_name  E7_proposed_aug

# Another: LEAM 重み可視化
CUBLAS_WORKSPACE_CONFIG=:16:8 python eval/visualize_leam_weights.py \
    experiments/lwn-spe-leam/E7/proposed_frac10/20260618_170710/checkpoints/best_model.pt \
    --save_path experiments/lwn-spe-leam/E7/proposed_frac10/20260618_170710/leam_weights.png