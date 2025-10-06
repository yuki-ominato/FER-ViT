import sys
import os
import argparse
from typing import List

from PIL import Image
from tqdm import tqdm
import torch

# このスクリプト(generate_latents.py)の絶対パスを取得
current_file_path = os.path.abspath(__file__)
# このスクリプトが置かれているディレクトリ(scripts)のパスを取得
scripts_dir = os.path.dirname(current_file_path)
# さらにその親ディレクトリ(プロジェクトルート fer-vit)のパスを取得
project_root = os.path.dirname(scripts_dir)

# プロジェクトルートをPythonの検索パスリストに追加
sys.path.append(project_root)

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


def main(args):
    os.makedirs(args.latent_out, exist_ok=True)
    encoder = EncoderWrapper(args.encoder_model, device="cuda", encoder_type=args.encoder_type)

    total_processed = 0
    total_skipped = 0
    
    for cls in sorted(os.listdir(args.data_root)):
        cls_dir = os.path.join(args.data_root, cls)
        if not os.path.isdir(cls_dir):
            continue
            
        label = CLASS_TO_LABEL.get(cls, -1)
        if label == -1:
            print(f"Unknown class: {cls}, skipping...")
            continue
        
        print(f"Processing class: {cls} (label: {label})")
        
        # ファイルリストを準備
        image_paths = []
        output_paths = []
        labels = []
        
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
                
            img_path = os.path.join(cls_dir, fname)
            base = os.path.splitext(fname)[0]
            out_path = os.path.join(args.latent_out, f"{cls}_{base}.pt")
            
            if os.path.exists(out_path):
                total_skipped += 1
                continue
                
            image_paths.append(img_path)
            output_paths.append(out_path)
            labels.append(label)
        
        if not image_paths:
            print(f"No new images to process for class {cls}")
            continue
        
        print(f"Processing {len(image_paths)} images for class {cls}")
        
        # バッチ処理
        with tqdm(total=len(image_paths), desc=f"Encoding {cls}") as pbar:
            process_images_batch(encoder, image_paths, labels, output_paths, args.batch_size)
            total_processed += len(image_paths)
            pbar.update(len(image_paths))
    
    print(f"\nProcessing completed!")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped (already exists): {total_skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate latent codes from images using pSp/e4e")
    parser.add_argument("--data_root", required=True, help="Directory with class-subdir structured images")
    parser.add_argument("--latent_out", required=True, help="Output directory for latent files")
    parser.add_argument("--encoder_model", required=True, help="Path to encoder model checkpoint")
    parser.add_argument("--encoder_type", choices=['psp', 'e4e'], default='psp', 
                       help="Type of encoder to use")
    parser.add_argument("--batch_size", type=int, default=4, 
                       help="Batch size for encoding (adjust based on GPU memory)")
    args = parser.parse_args()
    main(args)


