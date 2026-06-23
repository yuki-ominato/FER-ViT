import csv
import sys
import os
import argparse
from typing import List

from PIL import Image
from tqdm import tqdm
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# このスクリプト(generate_latents.py)の絶対パスを取得
current_file_path = os.path.abspath(__file__)
# このスクリプトが置かれているディレクトリ(scripts)のパスを取得
scripts_dir = os.path.dirname(current_file_path)
# さらにその親ディレクトリ(プロジェクトルート fer-vit)のパスを取得
project_root = os.path.dirname(scripts_dir)

# プロジェクトルートをPythonの検索パスリストに追加
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# pSp 内部の bare import (from models.encoders import ...) が解決できるよう
# third_party/pixel2style2pixel をプロジェクトルートよりも先に追加する
psp_root = os.path.join(project_root, 'third_party', 'pixel2style2pixel')
if os.path.isdir(psp_root) and psp_root not in sys.path:
    sys.path.insert(0, psp_root)

from models_fer_vit.encoder_wrapper import EncoderWrapper


CLASS_TO_LABEL = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6,
}

# RAF-DB 公式ラベル (1-7) → 共通ラベル (0-6)
RAFDB_TO_LABEL = {
    1: 6,  # Surprise
    2: 2,  # Fear
    3: 1,  # Disgust
    4: 3,  # Happiness
    5: 5,  # Sadness
    6: 0,  # Anger
    7: 4,  # Neutral
}

def prepare_images_for_model(imgs, model):
    # imgs が list なら積んで tensor にする
    if isinstance(imgs, list):
        imgs = torch.stack(imgs, dim=0)

    # もし5次元（例 [B,1,C,H,W]）なら余分な次元を潰す
    if imgs.dim() == 5:
        # よくあるケース: [B, 1, C, H, W]
        if imgs.size(1) == 1:
            imgs = imgs.squeeze(1)  # -> [B, C, H, W]
        # まれに [B, C, 1, H, W] のような順序のミスがあるなら別条件も追加可能

    # デバイス合わせと dtype合わせ（モデルのパラメータの dtype に合わせる）
    model_param = next(model.parameters())
    target_dtype = model_param.dtype
    imgs = imgs.to(device=device, dtype=target_dtype)

    return imgs

def process_images_batch(encoder: EncoderWrapper, image_paths: List[str], 
                        labels: List[int], output_paths: List[str], 
                        batch_size: int = 4):
    """バッチで画像を処理"""
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        batch_outputs = output_paths[i:i + batch_size]
        
        # 画像を読み込み
        pil_images = []
        valid_indices = []
        
        for j, img_path in enumerate(batch_paths):
            try:
                pil = Image.open(img_path).convert("RGB")
                pil_images.append(pil)
                valid_indices.append(j)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        if not pil_images:
            continue

        print(f"Encoding {len(pil_images)} images")
        # バッチエンコード
        try:
            latents = encoder.encode_batch(pil_images, batch_size=len(pil_images))
            
            # 結果を保存
            for k, idx in enumerate(valid_indices):
                if idx < len(batch_outputs):
                    latent = latents[k].squeeze(0) if latents[k].dim() == 3 else latents[k]
                    torch.save({
                        "latent": latent, 
                        "label": batch_labels[idx], 
                        "img_path": batch_paths[idx]
                    }, batch_outputs[idx])
        except Exception as e:
            print(f"Error processing batch: {e}")
            # 個別処理にフォールバック
            for k, idx in enumerate(valid_indices):
                try:
                    w = encoder.encode_image(pil_images[k])
                    torch.save({
                        "latent": w.squeeze(0), 
                        "label": batch_labels[idx], 
                        "img_path": batch_paths[idx]
                    }, batch_outputs[idx])
                except Exception as e2:
                    print(f"Error processing individual image {batch_paths[idx]}: {e2}")


def collect_samples_fer2013(data_root: str):
    """FER2013 形式 (クラス名サブディレクトリ) からサンプルリストを作成"""
    samples = []
    for cls in sorted(os.listdir(data_root)):
        cls_dir = os.path.join(data_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        label = CLASS_TO_LABEL.get(cls.lower(), -1)
        if label == -1:
            print(f"Unknown class: {cls}, skipping...")
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            samples.append((os.path.join(cls_dir, fname), label, cls, fname))
    return samples


def collect_samples_rafdb(data_root: str, split: str, csv_path: str = None):
    """RAF-DB 形式 (CSV + 番号付きサブディレクトリ) からサンプルリストを作成"""
    if csv_path is None:
        csv_path = os.path.join(data_root, f'{split}_labels.csv')
    samples = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row['image']
            raf_label = int(row['label'])
            label = RAFDB_TO_LABEL[raf_label]
            img_path = os.path.join(data_root, split, str(raf_label), img_name)
            cls_name = f"cls{raf_label}"
            samples.append((img_path, label, cls_name, img_name))
    return samples


def main(args):
    os.makedirs(args.latent_out, exist_ok=True)
    encoder = EncoderWrapper(args.encoder_model, device="cuda", encoder_type=args.encoder_type)

    if args.dataset_type == 'fer2013':
        all_samples = collect_samples_fer2013(args.data_root)
    else:
        all_samples = collect_samples_rafdb(args.data_root, args.split, args.csv_path)

    # クラスごとにグループ化して処理
    from collections import defaultdict
    class_samples = defaultdict(list)
    for img_path, label, cls_name, fname in all_samples:
        class_samples[cls_name].append((img_path, label, fname))

    total_processed = 0
    total_skipped = 0

    for cls_name, items in sorted(class_samples.items()):
        print(f"Processing class: {cls_name} ({len(items)} images)")

        image_paths = []
        output_paths = []
        labels = []

        for img_path, label, fname in items:
            base = os.path.splitext(fname)[0]
            out_path = os.path.join(args.latent_out, f"{cls_name}_{base}.pt")

            if os.path.exists(out_path):
                total_skipped += 1
                continue

            image_paths.append(img_path)
            output_paths.append(out_path)
            labels.append(label)

        if not image_paths:
            print(f"No new images to process for class {cls_name}")
            continue

        print(f"Processing {len(image_paths)} images for class {cls_name}")

        with tqdm(total=len(image_paths), desc=f"Encoding {cls_name}") as pbar:
            process_images_batch(encoder, image_paths, labels, output_paths, args.batch_size)
            total_processed += len(image_paths)
            pbar.update(len(image_paths))

    print(f"\nProcessing completed!")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped (already exists): {total_skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate latent codes from images using pSp/e4e")
    parser.add_argument("--dataset_type", choices=['fer2013', 'raf-db'], default='fer2013',
                       help="Dataset format: 'fer2013' (class-named subdirs) or 'raf-db' (numbered subdirs + CSV)")
    parser.add_argument("--data_root", required=True,
                       help="FER2013: class-subdir root; RAF-DB: dataset root containing train/ test/ and *.csv")
    parser.add_argument("--split", default='train', choices=['train', 'test'],
                       help="RAF-DB only: which split to encode")
    parser.add_argument("--csv_path", default=None,
                       help="RAF-DB only: explicit path to label CSV (default: data_root/{split}_labels.csv)")
    parser.add_argument("--latent_out", required=True, help="Output directory for latent files")
    parser.add_argument("--encoder_model", required=True, help="Path to encoder model checkpoint")
    parser.add_argument("--encoder_type", choices=['psp', 'e4e'], default='psp',
                       help="Type of encoder to use")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for encoding (adjust based on GPU memory)")
    args = parser.parse_args()
    main(args)


