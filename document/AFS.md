# FER向けAFSベース表情データ拡張システム設計書

## 1. 研究目的

StyleGAN2の潜在空間W+上でIdentityとStyleを分離し、異なる人物間で表情を転写することでFER(Facial Expression Recognition)用のデータ拡張を行う。

本研究ではAFS(Arithmetic Face Swapping)の考え方を利用し、

w = w_id + w_sty

を学習する。

最終的に、

Identity(A)
+
Expression(B)

から

AさんがBさんの表情をしている画像

を生成する。

---

# 2. システム全体構成

```text
画像
 ↓
pSp Encoder
 ↓
W+ latent
 ↓
Style Extractor h
 ↓
 ┌──────────────┐
 │ w_id         │
 │ w_sty        │
 └──────────────┘

w_new = w_id(A) + w_sty(B)

 ↓
StyleGAN2 Generator
 ↓
生成画像
 ↓
FER Dataset
 ↓
ViT Training
```

---

# 3. ディレクトリ構成

```text
project/

├── configs/
│   ├── train_style_extractor.yaml
│   └── generate_dataset.yaml
│
├── datasets/
│   ├── RAFDB/
│   ├── AffectNet/
│   └── cache_latent/
│
├── models/
│   ├── psp/
│   ├── stylegan2/
│   ├── style_extractor/
│   └── vit/
│
├── src/
│   ├── encoder/
│   ├── extractor/
│   ├── generator/
│   ├── augmentation/
│   ├── fer/
│   └── utils/
│
├── outputs/
│   ├── latent/
│   ├── generated/
│   └── checkpoints/
│
└── train.py
```

---

# 4. 使用モデル

## pSp

役割

画像 → W+

```python
w = psp(image)
```

出力

```python
shape = [18,512]
```

---

## StyleGAN2

役割

W+ → 画像

```python
image = G(w)
```

---

## Style Extractor

役割

Style成分抽出

```python
w_sty = h(w)
```

Identity成分

```python
w_id = w - w_sty
```

---

# 5. Style Extractor設計

入力

```python
[18,512]
```

出力

```python
[18,512]
```

---

## Block

各layer独立処理

```python
Linear(512,256)

HighwayLayer

HighwayLayer

Linear(256,512)
```

AFS論文と同じ構造を採用。

---

# 6. 潜在空間キャッシュ

学習高速化のため事前計算する。

```python
for image in dataset:

    w = psp(image)

    save(
        image_id,
        w
    )
```

保存形式

```python
npy
```

推奨。

---

# 7. 損失関数

## Identity Loss

ArcFace使用

目的

生成画像の人物がSourceと同一人物であること。

```python
L_id
=
1 - cosine(
    ArcFace(gen),
    ArcFace(src)
)
```

---

## LPIPS Loss

目的

Style保持

```python
L_lpips
=
LPIPS(
    gen,
    target
)
```

---

## Consistency Loss

```python
w_swap
=
w_src
-
h(w_src)
+
h(w_tgt)
```

制約

```python
h(w_swap)
≈
h(w_tgt)
```

損失

```python
L_cons
=
| h(w_swap)
-
h(w_tgt) |
```

---

## Total Loss

```python
L
=
L_id
+
L_lpips
+
0.1 * L_cons
```

初期実装ではFeature Lossは省略可能。

まず動かすことを優先する。

---

# 8. 表情転写アルゴリズム

## 入力

Source

```text
Person A
Neutral
```

Target

```text
Person B
Angry
```

---

## Latent取得

```python
wA = psp(A)
wB = psp(B)
```

---

## 分離

```python
idA = wA - h(wA)
styB = h(wB)
```

---

## 合成

```python
w_new
=
idA
+
styB
```

---

## 生成

```python
img_new
=
G(w_new)
```

結果

```text
Person A
Angry
```

---

# 9. FERデータ拡張

元データ

```text
A Angry
B Angry
C Happy
D Sad
```

---

生成

```text
A + Angry(B)
A + Happy(C)
A + Sad(D)

B + Happy(C)
B + Sad(D)

...
```

---

期待効果

同一感情

×

多数人物

を人工生成できる。

---

# 10. FER学習

モデル

ViT-B/16

入力

```python
224x224
```

---

比較実験

## Baseline

実画像のみ

---

## Proposed

実画像
+
AFS生成画像

---

評価指標

Accuracy

F1-score

Balanced Accuracy

---

# 11. 実装優先順位

Phase1

pSp inversion

---

Phase2

Style Extractor実装

---

Phase3

Identity/Style分離確認

---

Phase4

表情転写生成

---

Phase5

FERデータ拡張

---

Phase6

ViT学習

---

# 12. 成功判定

成功条件

1. ArcFace類似度維持

2. 表情ラベル保持

3. FER精度向上

特に

```text
FER Accuracy
Baseline

↓

FER Accuracy
+AFS Augmentation
```

で統計的有意な改善が確認できること。