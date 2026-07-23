"""
PCA 感情部分空間の「圧縮次元数 k」を段階的に振って分類精度の変化を測るスクリプト。

build_pca_projection.py → project_latents_pca.py → evaluate_pca_subspace.py の
3段パイプラインを毎 k 実行すると、PCA 再適合と全ファイル射影書き出しが k の数だけ
繰り返されて非常に遅い。本スクリプトはそれを1プロセスに畳み込む:

    1. train / test の latent を一度だけメモリにロード (N, 9216)
    2. PCA を一度だけ適合（--pool 個の主成分を計算）
    3. 各主成分と感情ラベルの ANOVA F 値で成分を順位付け（build_pca と同じ基準）
    4. k を振りながら「F 値上位 k 個」の圧縮座標で LinearSVC を学習・評価
       → 9216 次元へ逆射影せず k 次元座標のまま分類する（分離性は等価で高速）

Baseline（全9216次元）と、任意で Residual（感情成分を除いた9216次元, --with_residual）も
同一の LinearSVC 設定で評価し、Emotion Only(k) がどこまで Baseline に近づくかを比較する。

中間 .pt は書き出さない。結果は表として標準出力に出し、--out_csv で CSV 保存も可能。

使い方:
    python latent_analysis/sweep_pca_dim.py \
        --train_dir latents/fer2013/train \
        --test_dir  latents/fer2013/test \
        --k_list 7,15,30,50,100,200 \
        --out_csv latent_analysis/pca_sweep_result.csv

    # 残差成分（9216次元）も評価（重いので任意）
    python latent_analysis/sweep_pca_dim.py \
        --train_dir latents/fer2013/train \
        --test_dir  latents/fer2013/test \
        --with_residual
"""

import os
import sys
import csv
import time
import argparse

import numpy as np
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EMOTION_NAMES = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
    4: 'neutral', 5: 'sad', 6: 'surprise',
}
SEQ_LEN = 18
LATENT_DIM = 512
FLAT_DIM = SEQ_LEN * LATENT_DIM   # 9216


def load_latents(latent_dir: str):
    """ディレクトリ内の .pt を全件ロードし X (N, 9216) と y (N,) を返す"""
    files = sorted(f for f in os.listdir(latent_dir) if f.endswith('.pt'))
    if not files:
        raise ValueError(f"No .pt files found in {latent_dir}")

    all_w, all_y = [], []
    print(f"Loading {len(files)} latent files from {latent_dir} ...")
    for fname in tqdm(files):
        data = torch.load(os.path.join(latent_dir, fname),
                          map_location='cpu', weights_only=True)
        all_w.append(data['latent'].numpy().reshape(-1))   # (9216,)
        all_y.append(int(data['label']))

    X = np.stack(all_w, axis=0).astype(np.float32)   # (N, 9216)
    y = np.array(all_y, dtype=np.int64)              # (N,)
    print(f"  X={X.shape}, y={y.shape}")
    return X, y


def evaluate(clf, X_tr, y_tr, X_te, y_te):
    """LinearSVC を学習して (accuracy, macro_f1, balanced_accuracy) を返す"""
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    return (
        accuracy_score(y_te, y_pred),
        f1_score(y_te, y_pred, average='macro', zero_division=0),
        balanced_accuracy_score(y_te, y_pred),
    )


def make_clf(args):
    return LinearSVC(C=args.C, dual=False, max_iter=args.max_iter)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep PCA emotion-subspace dimension k and measure classification accuracy"
    )
    parser.add_argument('--train_dir', required=True,
                        help='学習用 latent .pt ディレクトリ')
    parser.add_argument('--test_dir', required=True,
                        help='評価用 latent .pt ディレクトリ')
    parser.add_argument('--k_list', default='7,15,30,50,100,200',
                        help='カンマ区切りの圧縮次元数リスト (default: 7,15,30,50,100,200)')
    parser.add_argument('--pool', type=int, default=None,
                        help='PCA で計算する主成分プール数。省略時は max(k_list) を使用')
    parser.add_argument('--C', type=float, default=1.0,
                        help='LinearSVC の正則化パラメータ C (default: 1.0)')
    parser.add_argument('--max_iter', type=int, default=10000,
                        help='LinearSVC の最大反復数 (default: 10000)')
    parser.add_argument('--with_baseline', action='store_true', default=True,
                        help='全9216次元の Baseline も評価（デフォルト有効）')
    parser.add_argument('--no_baseline', dest='with_baseline', action='store_false',
                        help='Baseline 評価をスキップ')
    parser.add_argument('--with_residual', action='store_true',
                        help='各 k で残差成分(9216次元)も評価（重い）')
    parser.add_argument('--out_csv', default=None,
                        help='結果を保存する CSV パス（省略時は保存しない）')
    args = parser.parse_args()

    k_list = sorted(int(x) for x in args.k_list.split(',') if x.strip())
    if not k_list:
        raise ValueError("--k_list が空です")
    pool = args.pool if args.pool is not None else max(k_list)
    if pool < max(k_list):
        raise ValueError(f"--pool ({pool}) は max(k_list) ({max(k_list)}) 以上にしてください")

    # ── データロード ──────────────────────────────────────────────
    X_tr, y_tr = load_latents(args.train_dir)
    X_te, y_te = load_latents(args.test_dir)

    # pool は PCA の制約 (<= min(n_samples-1, n_features)) を超えられない
    pool = min(pool, X_tr.shape[0] - 1, X_tr.shape[1])
    print(f"\nPCA pool = {pool},  k_list = {k_list}")

    # ── PCA を一度だけ適合し、成分を F 値で順位付け ────────────────
    print(f"Fitting PCA(n_components={pool}) on train ...")
    t0 = time.time()
    pca = PCA(n_components=pool, random_state=42)
    coords_tr = pca.fit_transform(X_tr)          # (Ntr, pool)  中心化済み座標
    coords_te = pca.transform(X_te)              # (Nte, pool)
    print(f"  done in {time.time()-t0:.1f}s")

    f_scores, _ = f_classif(coords_tr, y_tr)     # (pool,)
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    order = np.argsort(f_scores)[::-1]           # F 値降順の成分インデックス

    # ── Baseline（全9216次元） ────────────────────────────────────
    results = []
    if args.with_baseline:
        print("\n[Baseline] LinearSVC on full 9216-dim ...")
        t0 = time.time()
        acc, f1, bal = evaluate(make_clf(args), X_tr, y_tr, X_te, y_te)
        print(f"  acc={acc:.4f} f1={f1:.4f} bal={bal:.4f}  ({time.time()-t0:.1f}s)")
        results.append({'variant': 'baseline', 'k': FLAT_DIM,
                        'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bal,
                        'expl_var': 1.0})

    # ── k を振って Emotion Only（k次元圧縮座標）を評価 ────────────
    comp = pca.components_            # (pool, 9216)  各行が単位主成分
    for k in k_list:
        top = order[:k]
        Etr = coords_tr[:, top]      # (Ntr, k)  感情部分空間の圧縮座標
        Ete = coords_te[:, top]
        ev = float(pca.explained_variance_ratio_[top].sum())

        print(f"\n[Emotion Only k={k}] LinearSVC on {k}-dim compressed coords "
              f"(cum. expl.var={ev:.4f}) ...")
        t0 = time.time()
        acc, f1, bal = evaluate(make_clf(args), Etr, y_tr, Ete, y_te)
        print(f"  acc={acc:.4f} f1={f1:.4f} bal={bal:.4f}  ({time.time()-t0:.1f}s)")
        results.append({'variant': 'emotion', 'k': k,
                        'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bal,
                        'expl_var': ev})

        if args.with_residual:
            # 残差 = X - N @ (N^T_centered proj) を 9216 次元で復元して評価
            # 逆変換: emotion_recon = coords_topk @ comp_topk (+ mean)
            emo_recon_tr = Etr @ comp[top] + pca.mean_
            emo_recon_te = Ete @ comp[top] + pca.mean_
            Rtr = X_tr - emo_recon_tr
            Rte = X_te - emo_recon_te
            print(f"[Residual k={k}] LinearSVC on 9216-dim residual ...")
            t0 = time.time()
            racc, rf1, rbal = evaluate(make_clf(args), Rtr, y_tr, Rte, y_te)
            print(f"  acc={racc:.4f} f1={rf1:.4f} bal={rbal:.4f}  ({time.time()-t0:.1f}s)")
            results.append({'variant': 'residual', 'k': k,
                            'accuracy': racc, 'macro_f1': rf1, 'balanced_accuracy': rbal,
                            'expl_var': ev})

    # ── サマリ表 ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'variant':<12}{'k':>7}{'expl.var':>10}{'accuracy':>11}"
          f"{'macro_f1':>11}{'bal_acc':>11}")
    print("-" * 72)
    for r in results:
        print(f"{r['variant']:<12}{r['k']:>7}{r['expl_var']:>10.4f}"
              f"{r['accuracy']:>11.4f}{r['macro_f1']:>11.4f}{r['balanced_accuracy']:>11.4f}")
    print("=" * 72)

    # ── CSV 保存 ──────────────────────────────────────────────────
    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['variant', 'k', 'expl_var',
                               'accuracy', 'macro_f1', 'balanced_accuracy'])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    'variant': r['variant'], 'k': r['k'], 'expl_var': r['expl_var'],
                    'accuracy': r['accuracy'], 'macro_f1': r['macro_f1'],
                    'balanced_accuracy': r['balanced_accuracy'],
                })
        print(f"\nSaved -> {args.out_csv}")


if __name__ == '__main__':
    main()
