"""
StyleExtractor を事前適用して潜在コードを変換・保存するスクリプト。

変換済み .pt を用意しておけば train_latent_vit.py で直接訓練でき、
毎バッチ・毎エポックの StyleExtractor 推論コストが不要になる。

変換モード:
    style    : w_sty = h(w)           → 感情スタイル成分
    identity : w_id  = w − h(w)       → アイデンティティ残差成分
    both     : 上記を --out_dir/style/ と --out_dir/identity/ に両方保存

使い方:
    python data/extract_style_latents.py \\
        --latent_dir           latents/train \\
        --out_dir              latents/train_sty \\
        --style_extractor_path outputs/afs/<run_id>/checkpoints/best_model.pt

    # identity 成分を保存
    python data/extract_style_latents.py \\
        --latent_dir           latents/train \\
        --out_dir              latents/train_id \\
        --style_extractor_path outputs/afs/<run_id>/checkpoints/best_model.pt \\
        --mode                 identity

    # 両方まとめて保存
    python data/extract_style_latents.py \\
        --latent_dir           latents/train \\
        --out_dir              latents/train_afs \\
        --style_extractor_path outputs/afs/<run_id>/checkpoints/best_model.pt \\
        --mode                 both
"""

import os
import sys
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from afs.style_extractor import StyleExtractor


# ──────────────────────────────────────────────────────────────────────────────
# Dataset: ファイル名も一緒に返す
# ──────────────────────────────────────────────────────────────────────────────

class LatentFileDataset(Dataset):
    """(latent, label, filename) を返す最小データセット"""

    def __init__(self, latent_dir: str) -> None:
        self.files = sorted(
            os.path.join(latent_dir, f)
            for f in os.listdir(latent_dir)
            if f.endswith('.pt')
        )
        if not self.files:
            raise ValueError(f"No .pt files in {latent_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        data = torch.load(path, map_location='cpu', weights_only=True)
        latent = data['latent'].float()    # (18, 512)
        label  = torch.tensor(int(data['label']), dtype=torch.long)
        fname  = os.path.basename(path)
        return latent, label, fname


def collate_fn(batch):
    latents = torch.stack([b[0] for b in batch])
    labels  = torch.stack([b[1] for b in batch])
    fnames  = [b[2] for b in batch]
    return latents, labels, fnames


# ──────────────────────────────────────────────────────────────────────────────
# StyleExtractor のロード
# ──────────────────────────────────────────────────────────────────────────────

def load_style_extractor(ckpt_path: str, device: torch.device) -> StyleExtractor:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    h = StyleExtractor()
    h.load_state_dict(ckpt['model_state'])
    h.eval()
    for p in h.parameters():
        p.requires_grad_(False)
    return h.to(device)


# ──────────────────────────────────────────────────────────────────────────────
# メイン処理
# ──────────────────────────────────────────────────────────────────────────────

def prepare_out_dirs(out_dir: str, mode: str) -> dict:
    if mode == 'both':
        dirs = {
            'style':    os.path.join(out_dir, 'style'),
            'identity': os.path.join(out_dir, 'identity'),
        }
    else:
        dirs = {mode: out_dir}

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


@torch.no_grad()
def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # StyleExtractor
    print(f"StyleExtractor をロード: {args.style_extractor_path}")
    h = load_style_extractor(args.style_extractor_path, device)

    # データセット
    dataset = LatentFileDataset(args.latent_dir)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
    )
    print(f"入力: {args.latent_dir}  ({len(dataset)} ファイル)")

    # 出力ディレクトリ
    out_dirs = prepare_out_dirs(args.out_dir, args.mode)
    for k, d in out_dirs.items():
        print(f"出力 [{k}]: {d}")

    saved = 0
    for latents, labels, fnames in tqdm(loader, desc='extracting'):
        latents = latents.to(device)   # (B, 18, 512)

        w_sty = h(latents)             # (B, 18, 512)

        for i, fname in enumerate(fnames):
            w_orig = latents[i].cpu()
            w_s    = w_sty[i].cpu()
            w_id   = (w_orig - w_s)
            label  = labels[i].item()

            if args.mode in ('style', 'both'):
                torch.save(
                    {'latent': w_s, 'label': label},
                    os.path.join(out_dirs['style'], fname),
                )
            if args.mode in ('identity', 'both'):
                torch.save(
                    {'latent': w_id, 'label': label},
                    os.path.join(out_dirs['identity'], fname),
                )

        saved += len(fnames)

    print(f"\n完了: {saved} ファイルを変換・保存しました。")
    if args.mode == 'both':
        print(f"  style    → {out_dirs['style']}")
        print(f"  identity → {out_dirs['identity']}")
    else:
        print(f"  {args.mode} → {list(out_dirs.values())[0]}")

    print("\n次のステップ:")
    mode_dir = out_dirs.get('style', list(out_dirs.values())[0])
    print(f"  CUBLAS_WORKSPACE_CONFIG=:16:8 python train/train_latent_vit.py \\")
    print(f"      --latent_train_dir {mode_dir} \\")
    print(f"      --latent_val_dir   <val出力パス> \\")
    print(f"      --epochs 60 --batch_size 64")


def main():
    parser = argparse.ArgumentParser(
        description="StyleExtractor を事前適用して潜在コードを保存"
    )
    parser.add_argument('--latent_dir',           required=True,
                        help='入力: 変換前の .pt ディレクトリ')
    parser.add_argument('--out_dir',              required=True,
                        help='出力: 変換後の .pt を保存するディレクトリ')
    parser.add_argument('--style_extractor_path', required=True,
                        help='訓練済み StyleExtractor チェックポイント (best_model.pt)')
    parser.add_argument('--mode',   choices=['style', 'identity', 'both'],
                        default='style',
                        help='保存する成分 (default: style → w_sty = h(w))')
    parser.add_argument('--batch_size',  type=int, default=256,
                        help='バッチサイズ (default: 256)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader ワーカー数 (default: 4)')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
