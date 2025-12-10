import torch
import os
import argparse
from tqdm import tqdm
# ... 前述のクラスをインポート ...
from models_fer_vit.vqgan_wrapper import VQGANWrapper
from data.raf_dataset import RafDbDataset

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデルのロード
    model = VQGANWrapper(args.config_path, args.ckpt_path, device=device)
    
    # データセット
    dataset = RafDbDataset(args.data_root, split=args.split, img_size=args.img_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Generating tokens for {args.split}...")
    
    encoded_data = []
    labels_list = []
    
    # バッチ処理
    for imgs, labels in tqdm(loader):
        imgs = imgs.to(device)
        
        # エンコード (B, Seq_Len) 例: (B, 256) (16x16の場合)
        indices = model.get_codebook_indices(imgs)
        
        # CPUに戻してリストに追加
        encoded_data.append(indices.cpu())
        labels_list.append(labels)
        
    # 全データを結合して保存 (数万枚程度なら1ファイルでOKだが、分割も検討)
    all_tokens = torch.cat(encoded_data, dim=0) # (N, Seq_Len)
    all_labels = torch.cat(labels_list, dim=0)  # (N,)
    
    save_path = os.path.join(args.out_dir, f'rafdb_{args.split}_tokens.pt')
    torch.save({'tokens': all_tokens, 'labels': all_labels}, save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    # argparse設定...
    pass