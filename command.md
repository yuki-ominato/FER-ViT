# 実行コマンド集

## train_image_cnn.py（2D Image CNN）

### FER2013 × スクラッチ

```bash
python train/train_image_cnn.py \
    --dataset fer2013 \
    --train_dir /path/to/fer2013/train \
    --val_dir   /path/to/fer2013/val \
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
