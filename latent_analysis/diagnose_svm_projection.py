"""
SVM感情射影の品質診断スクリプト。

Step 1: 基底 N の品質確認（列ノルム・グラム行列・射影の冪等性）
Step 2: 感情部分空間が捉える分散の割合
Step 3: PCA次元数スイープ（感情情報が何次元に分布しているかの診断）
Step 4: 正規直交化済み基底の保存（オプション、SVDまたはQR）

使い方:
    # 基本診断
    python latent_analysis/diagnose_svm_projection.py \\
        --svm_basis      latent_analysis/svm_output/emotion_basis_N.pt \\
        --latent_train_dir latents/fer2013/train \\
        --latent_val_dir   latents/fer2013/val

    # 正規直交化基底も保存
    python latent_analysis/diagnose_svm_projection.py \\
        --svm_basis      latent_analysis/svm_output/emotion_basis_N.pt \\
        --latent_train_dir latents/fer2013/train \\
        --latent_val_dir   latents/fer2013/val \\
        --save_ortho     latent_analysis/svm_output/emotion_basis_N_ortho.pt

    # PCAスイープの次元数を変更
    python latent_analysis/diagnose_svm_projection.py \\
        --svm_basis      latent_analysis/svm_output/emotion_basis_N.pt \\
        --latent_train_dir latents/fer2013/train \\
        --latent_val_dir   latents/fer2013/val \\
        --pca_components 7 14 20 30 50 100 200 500 1000
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EMOTION_NAMES = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
    4: 'neutral', 5: 'sad',   6: 'surprise',
}
SEQ_LEN    = 18
LATENT_DIM = 512
FLAT_DIM   = SEQ_LEN * LATENT_DIM   # 9216


# ──────────────────────────────────────────────
# データロード
# ──────────────────────────────────────────────

def load_latents(latent_dir: str):
    """ディレクトリ内の .pt を全件ロードして (N, 9216) float32 配列を返す"""
    files = sorted(f for f in os.listdir(latent_dir) if f.endswith('.pt'))
    if not files:
        raise ValueError(f"No .pt files in {latent_dir}")
    all_w, all_labels = [], []
    print(f"Loading {len(files)} files from {latent_dir} ...")
    for fname in tqdm(files):
        data = torch.load(os.path.join(latent_dir, fname),
                          map_location='cpu', weights_only=True)
        all_w.append(data['latent'].numpy().reshape(-1))
        all_labels.append(int(data['label']))
    X = np.stack(all_w, axis=0).astype(np.float32)   # (N, 9216)
    y = np.array(all_labels)
    print(f"  X={X.shape}  y={y.shape}")
    return X, y


# ──────────────────────────────────────────────
# Step 1: 基底品質確認
# ──────────────────────────────────────────────

def step1_basis_quality(N: np.ndarray) -> None:
    print("\n" + "=" * 64)
    print("Step 1: 基底 N の品質確認")
    print("=" * 64)

    # 列ノルム
    norms = np.linalg.norm(N, axis=0)
    print(f"\n■ 列ノルム（全て 1.0 が理想）:")
    for name, norm in zip(EMOTION_NAMES.values(), norms):
        ok = "✓" if abs(norm - 1.0) < 1e-5 else "✗"
        print(f"   [{name:>8}]: {norm:.6f} {ok}")

    # グラム行列
    gram = N.T @ N
    print(f"\n■ Gram 行列 N^T N（単位行列に近いほど互いに直交）:")
    header = "".join(f"  {n:>8}" for n in EMOTION_NAMES.values())
    print(f"  {'':10}{header}")
    for i in range(7):
        row = "".join(f"  {gram[i, j]:+.3f}" for j in range(7))
        print(f"  [{list(EMOTION_NAMES.values())[i]:>8}]{row}")

    off = gram.copy(); np.fill_diagonal(off, 0.0)
    print(f"\n  非対角絶対値の最大: {np.abs(off).max():.4f}")
    print(f"  非対角絶対値の RMS : {np.sqrt(np.mean(off ** 2)):.4f}")

    # 射影の冪等性 P² = P の検証（単一サンプルで確認）
    print(f"\n■ 射影の冪等性検証（P² = P ？）:")
    pinv_N = np.linalg.pinv(N)   # (7, 9216)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(FLAT_DIM).astype(np.float32)
    proj1 = N @ (pinv_N @ x)
    proj2 = N @ (pinv_N @ proj1)
    err = np.max(np.abs(proj1 - proj2))
    ok = "✓" if err < 1e-4 else "✗"
    print(f"   max|P²x - Px| = {err:.2e}  {ok}")


# ──────────────────────────────────────────────
# Step 2: 感情部分空間が捉える分散の割合
# ──────────────────────────────────────────────

def step2_variance_captured(X: np.ndarray, N: np.ndarray,
                             batch_size: int = 512) -> float:
    print("\n" + "=" * 64)
    print("Step 2: 感情部分空間が捉える分散の割合")
    print("=" * 64)

    pinv_N = np.linalg.pinv(N)          # (7, 9216)
    X_c    = X - X.mean(axis=0)         # centered

    var_proj  = 0.0
    var_total = 0.0
    for i in range(0, len(X_c), batch_size):
        batch  = X_c[i: i + batch_size]          # (B, 9216)
        coords = batch @ pinv_N.T                 # (B, 7)
        proj   = coords @ N.T                     # (B, 9216)
        var_proj  += float(np.sum(proj  ** 2))
        var_total += float(np.sum(batch ** 2))

    ratio = var_proj / (var_total + 1e-12)
    print(f"\n  7次元感情部分空間が捉える分散割合: {ratio * 100:.4f}%")
    print(f"  （残差に残る割合: {(1 - ratio) * 100:.4f}%）")
    print(f"\n  ※ ViTの入力次元数: {FLAT_DIM}")
    print(f"     感情基底の次元数: {N.shape[1]}")
    print(f"     割合: {N.shape[1]}/{FLAT_DIM} = {N.shape[1]/FLAT_DIM*100:.4f}%")
    return ratio


# ──────────────────────────────────────────────
# Step 3: PCA 次元数スイープ
# ──────────────────────────────────────────────

def step3_pca_sweep(X_train: np.ndarray, y_train: np.ndarray,
                    X_val:   np.ndarray, y_val:   np.ndarray,
                    components: list) -> list:
    print("\n" + "=" * 64)
    print("Step 3: PCA 次元数スイープ")
    print("  → 感情情報が実際に何次元に分布しているかを診断")
    print("=" * 64)
    print(f"\n  {'n_comp':>7}  {'var_ratio':>10}  {'train_F1':>9}  {'val_F1':>8}  "
          f"{'val_BalAcc':>11}")
    print("  " + "-" * 52)

    results = []
    for n in components:
        n = min(n, min(X_train.shape))   # PCAの上限を超えないよう
        pca = PCA(n_components=n, random_state=42)
        X_tr = pca.fit_transform(X_train)
        X_va = pca.transform(X_val)

        clf = LinearSVC(C=1.0, dual=False, max_iter=5000)
        clf.fit(X_tr, y_train)

        f1_tr  = f1_score(y_train, clf.predict(X_tr), average='macro', zero_division=0)
        f1_va  = f1_score(y_val,   clf.predict(X_va), average='macro', zero_division=0)
        bal_va = balanced_accuracy_score(y_val, clf.predict(X_va))
        var    = pca.explained_variance_ratio_.sum()

        print(f"  {n:>7d}  {var:>10.4f}  {f1_tr:>9.4f}  {f1_va:>8.4f}  {bal_va:>11.4f}")
        results.append({
            'n_components': n,
            'var_ratio':    float(var),
            'f1_train':     float(f1_tr),
            'f1_val':       float(f1_va),
            'bal_acc_val':  float(bal_va),
        })

    return results


# ──────────────────────────────────────────────
# Step 4: 正規直交化基底の保存（オプション）
# ──────────────────────────────────────────────

def step4_orthogonalize(N: np.ndarray, method: str,
                         save_path: str, meta: dict) -> None:
    print("\n" + "=" * 64)
    print(f"Step 4: 正規直交化基底の生成・保存 (method={method})")
    print("=" * 64)

    if method == 'svd':
        U, S, Vt = np.linalg.svd(N, full_matrices=False)
        N_ortho = U.astype(np.float32)   # (9216, 7)
        print(f"  SVD 特異値: {S.round(4)}")
    else:   # qr
        Q, R = np.linalg.qr(N)
        N_ortho = Q.astype(np.float32)   # (9216, 7)

    gram = N_ortho.T @ N_ortho
    off  = gram.copy(); np.fill_diagonal(off, 0.0)
    print(f"  正規直交化後 非対角最大値: {np.abs(off).max():.2e}  (理想: 0)")

    torch.save({
        'N':            torch.tensor(N_ortho, dtype=torch.float32),
        'emotion_names': {int(k): v for k, v in EMOTION_NAMES.items()},
        'seq_len':       int(meta.get('seq_len',   SEQ_LEN)),
        'latent_dim':    int(meta.get('latent_dim', LATENT_DIM)),
        'ortho_method':  method,
    }, save_path)
    print(f"\n  保存先: {save_path}")
    print(f"  ※ 正規直交化は部分空間を変えないため、")
    print(f"     射影結果（感情/残差成分）は元の N と同一になります。")


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SVM感情射影の品質診断")
    parser.add_argument('--svm_basis',         required=True,
                        help='emotion_basis_N.pt へのパス')
    parser.add_argument('--latent_train_dir',  required=True,
                        help='訓練用潜在コードのディレクトリ')
    parser.add_argument('--latent_val_dir',    required=True,
                        help='検証用潜在コードのディレクトリ')
    parser.add_argument('--save_ortho',        default=None,
                        help='正規直交化済み基底の保存先 (.pt)。省略時は保存しない。')
    parser.add_argument('--ortho_method',      choices=['qr', 'svd'], default='svd',
                        help='正規直交化の方法 (default: svd)')
    parser.add_argument('--pca_components',    nargs='+', type=int,
                        default=[7, 14, 20, 30, 50, 75, 100, 150, 200, 300, 500],
                        help='PCA スイープで試す次元数のリスト')
    parser.add_argument('--save_results',      default=None,
                        help='診断結果の JSON 保存先。省略時は保存しない。')
    args = parser.parse_args()

    # 基底のロード
    print(f"\n基底をロード: {args.svm_basis}")
    meta = torch.load(args.svm_basis, map_location='cpu', weights_only=True)
    N = meta['N'].numpy()   # (9216, 7)
    print(f"  N.shape: {N.shape}")

    # Step 1
    step1_basis_quality(N)

    # 潜在コードのロード
    print("\n潜在コードをロード中...")
    X_train, y_train = load_latents(args.latent_train_dir)
    X_val,   y_val   = load_latents(args.latent_val_dir)

    # Step 2
    var_ratio = step2_variance_captured(X_train, N)

    # Step 3
    pca_results = step3_pca_sweep(
        X_train, y_train, X_val, y_val, args.pca_components
    )

    # 診断サマリー
    best = max(pca_results, key=lambda r: r['f1_val'])
    print("\n" + "=" * 64)
    print("診断サマリー")
    print("=" * 64)
    print(f"\n  SVM 基底次元数              : {N.shape[1]}")
    print(f"  感情部分空間の分散捕捉率      : {var_ratio * 100:.4f}%")
    print(f"  PCA スイープ最良次元数        : {best['n_components']}")
    print(f"    → val F1   = {best['f1_val']:.4f}")
    print(f"    → val BalAcc = {best['bal_acc_val']:.4f}")
    if best['n_components'] > N.shape[1]:
        ratio = best['n_components'] / N.shape[1]
        print(f"\n  ▶ 感情情報は SVM 基底の約 {ratio:.1f} 倍の次元数に分布しています")
        print(f"    → 7 本の法線では感情情報を十分に捉えられていない可能性が高い")

    # JSON 保存
    if args.save_results:
        out = {
            'svm_basis':        args.svm_basis,
            'n_basis':          int(N.shape[1]),
            'var_ratio':        float(var_ratio),
            'pca_sweep':        pca_results,
            'best_n_components': best['n_components'],
            'best_f1_val':      best['f1_val'],
        }
        with open(args.save_results, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\n  診断結果 JSON: {args.save_results}")

    # Step 4 (オプション)
    if args.save_ortho:
        step4_orthogonalize(N, args.ortho_method, args.save_ortho, meta)


if __name__ == '__main__':
    main()
