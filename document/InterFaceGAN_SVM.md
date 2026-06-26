# 感情成分のみで訓練
python train/train_latent_vit_v2.py \
    --latent_train_dir latents/train \
    --latent_val_dir   latents/val \
    --svm_basis        latent_analysis/svm_output/emotion_basis_N.pt \
    --svm_projection   emotion

# 非感情成分（アイデンティティ等）のみで訓練
python train/train_latent_vit_v2.py \
    --latent_train_dir latents/train \
    --latent_val_dir   latents/val \
    --svm_basis        latent_analysis/svm_output/emotion_basis_N.pt \
    --svm_projection   residual

# 射影なし（従来通り）
python train/train_latent_vit_v2.py \
    --latent_train_dir latents/train \
    --latent_val_dir   latents/val


# 1. 基底を構築
python latent_analysis/build_svm_projection.py \
    --svm_model latent_analysis/svm_output/svm_model.joblib \
    --output_dir latent_analysis/svm_output

# 2. 全スプリットを射影
python latent_analysis/project_latents_svm.py \
    --basis latent_analysis/svm_output/emotion_basis_N.pt \
    --latent_root latents \
    --output_root latents_svm

# 3. 分離品質を評価
python latent_analysis/evaluate_svm_subspace.py \
    --latent_root latents_svm \
    --train_split train \
    --eval_split test


# SVM感情部分空間抽出 実装設計書

## 1. 研究目的

StyleGAN2のW+潜在空間上に存在する感情情報を線形SVMで抽出し、

* 感情部分空間
* 非感情部分空間

へ分離できるか検証する。

InterFaceGANの

「意味属性は線形境界で分離可能」

という仮説をFERへ応用する。

---

# 2. 全体フロー

```text
画像
 ↓
pSp / e4e
 ↓
W+ latent
 ↓
LinearSVC学習
 ↓
感情方向ベクトル取得

n_angry
n_disgust
...
n_surprise

 ↓
感情部分空間構築

 ↓
Projection

w_emotion
w_residual

 ↓
LatentViT
```

---

# 3. ディレクトリ構成

```text
latent_analysis/

├── train_svm.py
├── build_svm_projection.py
├── project_latents_svm.py
└── evaluate_svm_subspace.py
```

---

# 4. SVM学習

## 入力

```python
latent.shape
=
(18,512)
```

flatten

```python
x
=
latent.reshape(-1)

shape=(9216,)
```

---

## 学習データ

train スプリットのみを使用する。

test / val を SVM 学習に含めると感情方向 N が test ラベルに最適化され、
射影後の ViT 評価でリークが生じるため。

```python
X.shape
=
(N,9216)

y.shape
=
(N,)
```

---

## 学習

```python
from sklearn.svm import LinearSVC

clf = LinearSVC(
    C=1.0,
    dual=False,
    max_iter=10000
)

clf.fit(X,y)
```

---

# 5. 感情方向抽出

学習後

```python
clf.coef_
```

取得

```python
shape

(7,9216)
```

---

定義

```python
N = clf.coef_.T
```

```python
shape

(9216,7)
```

---

各列

```text
n_angry
n_disgust
n_fear
n_happy
n_neutral
n_sad
n_surprise
```

を表す。

---

# 6. 感情部分空間

感情部分空間

```math
E = span(N)
```

---

射影行列

```math
P = N(N^TN)^-1N^T
```

実装

```python
P = N @ np.linalg.pinv(N)
```

---

# 7. 潜在コード射影

## 感情成分

```python
w_emotion = P @ w
```

---

## 非感情成分

```python
w_residual = w - w_emotion
```

---

# 8. 保存形式

```python
torch.save({
    "latent": latent,
    "emotion_latent": w_emotion,
    "residual_latent": w_residual,
    "label": label,
})
```

---

# 9. ViT実験

## Baseline

```python
latent
```

---

## Emotion Only

```python
emotion_latent
```

---

## Residual Only

```python
residual_latent
```

---

# 10. 評価項目

Accuracy

Macro F1

Balanced Accuracy

---

# 11. 期待される結果

理想

```text
Baseline          75%

Emotion Only      70〜75%

Residual Only     15〜25%
```

---

# 12. 研究的主張

Residualで性能が大きく低下する場合、

「StyleGAN潜在空間の感情情報は線形分離可能な部分空間として存在する」

ことを示せる。
