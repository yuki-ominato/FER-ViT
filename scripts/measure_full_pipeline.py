import time
import torch
import sys
import os
import argparse
import numpy as np
from tqdm import tqdm

# パス設定
sys.path.append(os.getcwd())

from models_fer_vit.latent_vit import LatentViT
from models_fer_vit.image_vit import ImageViT
from models_fer_vit.encoder_wrapper import EncoderWrapper
import timm

def measure_latency(model_func, input_data, device='cuda', n_warmup=20, n_runs=100):
    """汎用的なレイテンシ計測関数"""
    # ウォームアップ
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model_func(input_data)
    
    if device == 'cuda':
        torch.cuda.synchronize()
        
    # 計測
    timings = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model_func(input_data)
                end.record()
                torch.cuda.synchronize()
                timings.append(start.elapsed_time(end)) # ms
            else:
                start = time.time()
                _ = model_func(input_data)
                end = time.time()
                timings.append((end - start) * 1000) # ms
                
    return np.mean(timings), np.std(timings)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("="*60)

    # 1. モデルの準備
    print("Loading models...")
    
    # A. 提案手法 (pSp + Latent ViT)
    # pSp
    encoder = EncoderWrapper(args.encoder_path, device=device, encoder_type=args.encoder_type)
    # Latent ViT (Depth=2)
    latent_vit = LatentViT(depth=2, embed_dim=512, heads=8, seq_len=18).to(device)
    latent_vit.eval()
    
    # B. 比較手法1 (Image ViT - Scratch, Depth=6)
    # 実験で使用した設定に合わせる
    image_vit_scratch = ImageViT(
        img_size=224, patch_size=16, embed_dim=512, depth=2, heads=8, mlp_dim=2048, num_classes=7
    ).to(device)
    image_vit_scratch.eval()

    # C. 比較手法2 (Image ViT - Pretrained, Depth=12)
    image_vit_pretrained = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=7).to(device)
    image_vit_pretrained.eval()

    # 2. 入力データの準備
    img_input_psp = torch.randn(1, 3, 256, 256).to(device)
    img_input_vit = torch.randn(1, 3, 224, 224).to(device)
    latent_input = torch.randn(1, 18, 512).to(device)

    print("\nStarting measurement...")
    print("-" * 60)
    print(f"{'Component':<35} | {'Mean Latency (ms)':<20}")
    print("-" * 60)

    # --- 計測1: Image ViT (Scratch) ---
    t_img_scratch, _ = measure_latency(lambda x: image_vit_scratch(x), img_input_vit, device)
    print(f"{'Image ViT (Scratch, d=6)':<35} | {t_img_scratch:.2f} ms")

    # --- 計測2: Image ViT (Pretrained) ---
    t_img_pre, _ = measure_latency(lambda x: image_vit_pretrained(x), img_input_vit, device)
    print(f"{'Image ViT (Pretrained, d=12)':<35} | {t_img_pre:.2f} ms")

    print("-" * 60)

    # --- 計測3: Latent ViT (Model Only) ---
    t_lat_vit, _ = measure_latency(lambda x: latent_vit(x), latent_input, device)
    print(f"{'Latent ViT (Model Only, d=2)':<35} | {t_lat_vit:.2f} ms")

    # --- 計測4: pSp Encoder (Preprocessing) ---
    # EncoderWrapperの内部モデルを直接叩く
    def run_psp_core(x):
        return encoder.encoder(x)

    t_psp, _ = measure_latency(run_psp_core, img_input_psp, device)
    print(f"{'pSp Encoder (Preprocessing)':<35} | {t_psp:.2f} ms")

    print("-" * 60)
    
    # --- 合計時間の比較 ---
    total_proposed = t_psp + t_lat_vit
    print(f"{'Proposed Total (pSp + ViT)':<35} | {total_proposed:.2f} ms")
    
    print("\n[Speedup Ratios]")
    print(f"Proposed vs Scratch (d=6):    {total_proposed / t_img_scratch:.2f}x (lower is faster)")
    print(f"Proposed vs Pretrained (d=12): {total_proposed / t_img_pre:.2f}x")
    print(f"Latent ViT ONLY vs Scratch:   {t_lat_vit / t_img_scratch:.2f}x")
    
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_path", default="pretrained_models/psp_ffhq_encode.pt", help="Path to pSp weights")
    parser.add_argument("--encoder_type", default="psp", choices=["psp", "e4e"])
    args = parser.parse_args()
    
    if not os.path.exists(args.encoder_path):
        print(f"Error: Encoder weights not found at {args.encoder_path}")
        print("Please specify correct path with --encoder_path")
    else:
        main(args)