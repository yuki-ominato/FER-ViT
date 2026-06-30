"""
InterFaceGAN に基づく感情方向抽出のための LinearSVC 学習スクリプト。

W+ 潜在コード (18, 512) を flatten して (N, 9216) の特徴行列を構成し、
7クラス LinearSVC を学習する。

test / val を含めると感情方向 N が test ラベルに最適化されリークが生じるため、
SVM 学習には train スプリットのみを使用する。

使い方:
    python latent_analysis/train_svm.py \
        --latent_dir latents/fer2013/train \
        --output_dir latent_analysis/svm_output_fer2013
"""

import os
import sys
import argparse
import numpy as np
import torch
import joblib
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EMOTION_NAMES = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
    4: 'neutral', 5: 'sad', 6: 'surprise',
}
NUM_CLASSES = 7


def load_latents(latent_dir: str):
    """ディレクトリ内の .pt ファイルを全件ロードして numpy 配列で返す"""
    files = sorted([f for f in os.listdir(latent_dir) if f.endswith('.pt')])
    if not files:
        raise ValueError(f"No .pt files found in {latent_dir}")

    all_w, all_labels = [], []
    print(f"Loading {len(files)} latent files from {latent_dir} ...")
    for fname in tqdm(files):
        data = torch.load(os.path.join(latent_dir, fname),
                          map_location='cpu', weights_only=True)
        all_w.append(data['latent'].numpy())   # (18, 512)
        all_labels.append(int(data['label']))

    X = np.stack(all_w, axis=0)            # (N, 18, 512)
    N = X.shape[0]
    X = X.reshape(N, -1)                   # (N, 9216)
    y = np.array(all_labels)               # (N,)
    print(f"  X={X.shape}, y={y.shape}")
    return X, y


def print_class_distribution(y: np.ndarray) -> None:
    N = len(y)
    print("\nClass distribution:")
    for cls_id in range(NUM_CLASSES):
        count = (y == cls_id).sum()
        print(f"  {EMOTION_NAMES[cls_id]:>8}: {count:5d} ({count / N * 100:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Train LinearSVC on W+ latents for emotion direction extraction"
    )
    parser.add_argument('--latent_dir', type=str, required=True,
                        help='Directory containing train .pt latent files')
    parser.add_argument('--output_dir', type=str,
                        default='latent_analysis/svm_output',
                        help='Directory to save the trained SVM model')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Regularization parameter for LinearSVC')
    parser.add_argument('--max_iter', type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    X, y = load_latents(args.latent_dir)
    print_class_distribution(y)

    print(f"\nTraining LinearSVC (C={args.C}, dual=False, max_iter={args.max_iter}) ...")
    clf = LinearSVC(C=args.C, dual=False, max_iter=args.max_iter)
    clf.fit(X, y)

    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)
    print(f"\nTrain accuracy        : {acc:.4f}")
    print(f"Train balanced accuracy: {bal_acc:.4f}")
    print("\nClassification report (train):")
    print(classification_report(y, y_pred, target_names=list(EMOTION_NAMES.values())))

    # coef_ shape: (7, 9216)
    print(f"clf.coef_.shape: {clf.coef_.shape}")

    out_path = os.path.join(args.output_dir, 'svm_model.joblib')
    joblib.dump(clf, out_path)
    print(f"\nSaved SVM model -> {out_path}")


if __name__ == '__main__':
    main()
