"""
感情部分空間分離の評価スクリプト。

train/val/test の射影済み潜在コードを読み込み、
3種類の入力でそれぞれ LinearSVC を学習・評価する。

    Baseline      : latent          (元の W+ コード)
    Emotion Only  : emotion_latent  (感情部分空間への射影)
    Residual Only : residual_latent (感情成分を除いた残差)

評価指標:
    Accuracy / Macro F1 / Balanced Accuracy

使い方:
    python latent_analysis/evaluate_svm_subspace.py \
        --latent_root latents_svm \
        --train_split train \
        --eval_split test
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EMOTION_NAMES = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
    4: 'neutral', 5: 'sad', 6: 'surprise',
}
KEYS = ['latent', 'emotion_latent', 'residual_latent']
LABELS = ['Baseline', 'Emotion Only', 'Residual Only']

SEQ_LEN = 18
LATENT_DIM = 512


def load_split(split_dir: str):
    """
    射影済み .pt ファイルを全件ロードして
    {key: ndarray (N, 9216)} と ラベル配列 (N,) を返す
    """
    files = sorted([f for f in os.listdir(split_dir) if f.endswith('.pt')])
    if not files:
        raise ValueError(f"No .pt files found in {split_dir}")

    arrays = {k: [] for k in KEYS}
    labels = []

    print(f"Loading {len(files)} files from {split_dir} ...")
    for fname in tqdm(files):
        data = torch.load(os.path.join(split_dir, fname),
                          map_location='cpu', weights_only=True)
        for k in KEYS:
            arrays[k].append(data[k].numpy().reshape(-1))   # (9216,)
        labels.append(int(data['label']))

    return (
        {k: np.stack(v, axis=0) for k, v in arrays.items()},   # each (N, 9216)
        np.array(labels),
    )


def evaluate_variant(
    label: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> dict:
    """LinearSVC で学習・評価して指標を返す"""
    clf = LinearSVC(C=1.0, dual=False, max_iter=10000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_eval)
    acc = accuracy_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(y_eval, y_pred)

    return {'label': label, 'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bal_acc,
            'y_pred': y_pred}


def print_results(results: list, y_eval: np.ndarray) -> None:
    print("\n" + "=" * 64)
    print(f"{'Variant':<16}  {'Accuracy':>10}  {'Macro F1':>10}  {'Bal. Acc':>10}")
    print("-" * 64)
    for r in results:
        print(f"  {r['label']:<14}  {r['accuracy']:>10.4f}  "
              f"{r['macro_f1']:>10.4f}  {r['balanced_accuracy']:>10.4f}")
    print("=" * 64)

    for r in results:
        print(f"\n--- Classification report: {r['label']} ---")
        print(classification_report(
            y_eval, r['y_pred'],
            target_names=list(EMOTION_NAMES.values()),
            zero_division=0,
        ))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate emotion / residual subspace separation via LinearSVC"
    )
    parser.add_argument('--latent_root', type=str, required=True,
                        help='Root directory containing projected split subdirectories')
    parser.add_argument('--train_split', type=str, default='train',
                        help='Split name used for training the evaluation classifier')
    parser.add_argument('--eval_split', type=str, default='test',
                        help='Split name used for evaluation')
    args = parser.parse_args()

    train_dir = os.path.join(args.latent_root, args.train_split)
    eval_dir = os.path.join(args.latent_root, args.eval_split)

    print(f"Train split : {train_dir}")
    print(f"Eval split  : {eval_dir}")

    train_arrays, y_train = load_split(train_dir)
    eval_arrays, y_eval = load_split(eval_dir)

    results = []
    for key, label in zip(KEYS, LABELS):
        print(f"\nTraining evaluation classifier for [{label}] ...")
        r = evaluate_variant(
            label,
            train_arrays[key], y_train,
            eval_arrays[key], y_eval,
        )
        results.append(r)

    print_results(results, y_eval)


if __name__ == '__main__':
    main()
