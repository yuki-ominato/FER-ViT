# 感情部分空間の比較実験（k=7, SVM / PCA / ランダム）

W+ 潜在空間（`(18, 512)` = flatten 9216 次元）から取り出す **7 次元の感情部分空間**を、
基底の作り方だけを変えた 3 手法で比較する実験の手順書。

| 手法 | 基底 N (9216×7) の作り方 | ラベル使用 | 分散使用 |
|---|---|---|---|
| **SVM** | 7クラス LinearSVC の `coef_.T`（各クラスの識別方向）を L2 正規化 | ✅（識別境界を直接最適化） | ✗ |
| **PCA** | 全データ PCA の主成分のうち、ラベルとの ANOVA F 値が高い上位 7 個 | △（成分の事後選択に使用） | ✅ |
| **ランダム** | (9216, 7) ガウス乱数を QR 分解した正規直交基底（対照群） | ✗ | ✗ |

**狙い**: 同じ 7 次元でも、教師あり識別方向（SVM）／分散＋相関で選んだ方向（PCA）／
無情報のランダム方向で、感情情報の集約度がどれだけ違うかを測る。ランダムが下限の
対照群になる。

---

## 0. 実験設定の統一（重要）

3 手法を公平に比べるため、**基底構築より後（射影・評価）は全手法で完全に同一のコード**を使う。

| 段階 | 使用スクリプト | 設定 |
|---|---|---|
| 射影 | `project_latents_svm.py`（PCA/ランダムにもそのまま使える） | `w_emo = N @ (pinv(N) @ w)`, `w_res = w − w_emo`（生の w、中心化なし） |
| 評価 | `evaluate_svm_subspace.py`（PCA/ランダムにもそのまま使える） | `LinearSVC(C=1.0, dual=False, max_iter=10000)`, スケーラなし, **train で学習・test で評価** |

- 基底 N はすべて `{'N': Tensor(9216, 7), ...}` の同一フォーマットで保存されるため、
  射影・評価スクリプトは手法を問わず共通で使える。
- 基底構築（SVM/PCA/ランダム）はいずれも **train split のみ**を使う（test/val を使うと
  感情方向が test に最適化されリークするため）。

### 評価器についての注意（過去の食い違いの原因）

本実験の「Baseline / Emotion Only / Residual Only」の数値は、すべて
**`evaluate_*_subspace.py` 内の LinearSVC（線形・1回 fit）を test split で測った値**である。

過去に記録した Baseline **Acc≈0.534**（`learning_logs/cnn_vs_vit/…latent_vit…val_acc.csv`）は
**LatentViT（非線形 Transformer, 60epoch）を val split で測った値**であり、測定器も split も
別物なので一致しない（同じ FER2013 でも `evaluate_*` の LinearSVC-test では Baseline≈0.4944）。
**本比較実験では評価器を LinearSVC に固定**し、ViT の数値とは混在させないこと。

> 注: 上記 0.534 / 0.4944 はいずれも **FER2013** で得た旧数値であり、目安として挙げたもの。
> 本実験は **RAF-DB** で取り直すため、Baseline の絶対値はこれらとは異なる（食い違いの
> 原因＝「測定器・split の違い」という論点だけが RAF-DB でもそのまま当てはまる）。

---

## 1. 前提：latent の用意

`latents/raf-db/{train,test}` に W+ 潜在コード（`{'latent': (18,512), 'label': int}`）が
展開済みであること。無ければ `data/generate_latents.py` で作成する（`document/commands.md` 参照）。

```bash
cd /home/yuki/research2/fer-vit
source /home/yuki/anaconda3/etc/profile.d/conda.sh && conda activate fer-vit
```

---

## 2. 手順

### 2-1. SVM 手法

```bash
# ① LinearSVC を学習（coef_ を基底の元にする）
python latent_analysis/train_svm.py \
    --latent_dir latents/raf-db/train \
    --output_dir latent_analysis/svm_output_raf-db

# ② coef_ から感情基底 N (9216×7) を構築
python latent_analysis/build_svm_projection.py \
    --svm_model  latent_analysis/svm_output_raf-db/svm_model.joblib \
    --output_dir latent_analysis/svm_output_raf-db

# ③ train/test を射影
python latent_analysis/project_latents_svm.py \
    --basis       latent_analysis/svm_output_raf-db/emotion_basis_N.pt \
    --latent_root latents/raf-db \
    --output_root latents_svm_raf-db \
    --splits train test

# ④ 評価（Baseline / Emotion Only / Residual Only）
python latent_analysis/evaluate_svm_subspace.py \
    --latent_root latents_svm_raf-db \
    --train_split train --eval_split test \
    2>&1 | tee latent_analysis/result_svm_k7.log
```

### 2-2. PCA 手法

```bash
# ① PCA 適合＋F値で上位 k=7 成分を選び基底 N を構築（探索プール100）
python latent_analysis/build_pca_projection.py \
    --latent_dir  latents/raf-db/train \
    --output_dir  latent_analysis/pca_output_raf-db \
    --n_components 100 --k 7

# ② train/test を射影（SVM版と同一スクリプトでも可、ここでは PCA 専用ラッパを使用）
python latent_analysis/project_latents_pca.py \
    --basis       latent_analysis/pca_output_raf-db/emotion_basis_pca.pt \
    --latent_root latents/raf-db \
    --output_root latents_pca_raf-db \
    --splits train test

# ③ 評価
python latent_analysis/evaluate_pca_subspace.py \
    --latent_root latents_pca_raf-db \
    --train_split train --eval_split test \
    2>&1 | tee latent_analysis/result_pca_k7.log
```

### 2-3. ランダム手法（対照群）

```bash
# ① 正規直交なランダム 7 次元基底 N を生成（seed 固定で再現可能）
python latent_analysis/build_random_projection.py \
    --output_dir latent_analysis/random_output_raf-db \
    --k 7 --seed 42

# ② train/test を射影（SVM版スクリプトを流用）
python latent_analysis/project_latents_svm.py \
    --basis       latent_analysis/random_output_raf-db/emotion_basis_random.pt \
    --latent_root latents/raf-db \
    --output_root latents_random_raf-db \
    --splits train test

# ③ 評価（SVM版スクリプトを流用）
python latent_analysis/evaluate_svm_subspace.py \
    --latent_root latents_random_raf-db \
    --train_split train --eval_split test \
    2>&1 | tee latent_analysis/result_random_k7.log
```

> ランダム手法の頑健性を見たい場合は `--seed` を変えて数回繰り返し、Emotion Only の
> 平均・分散を取る。

---

## 3. 結果の記録テンプレート

各 `result_*_k7.log` の `Emotion Only` / `Residual Only` 行を転記する。
Baseline は 3 手法とも同じ生 W+ + 同じ LinearSVC なので同値になる（整合性チェックになる）。

| 手法 | Variant | Accuracy | Macro F1 | Bal. Acc |
|---|---|---|---|---|
| （共通） | Baseline（全9216次元） | | | |
| SVM | Emotion Only（7次元） | | | |
| SVM | Residual Only（9209次元） | | | |
| PCA | Emotion Only（7次元） | | | |
| PCA | Residual Only（9209次元） | | | |
| ランダム | Emotion Only（7次元） | | | |
| ランダム | Residual Only（9209次元） | | | |

### 期待される読み取り

- **Emotion Only**: SVM ≫ PCA > ランダム を期待。SVM がランダムを大きく上回れば
  「7 次元の教師あり識別部分空間に感情情報が線形分離可能な形で集約されている」ことを示せる。
  PCA がランダムと大差なければ「分散最大方向には感情情報が乗っていない」ことの傍証。
- **Residual Only**: 理想は「Emotion を抜くと落ちる」だが、7 次元を抜いた残差 9209 次元は
  情報をほぼ保持するため Baseline に近いままになりやすい（`PCA_projection.md` の考察参照）。

---

## 4. 関連ファイル

| 種別 | SVM | PCA | ランダム |
|---|---|---|---|
| 基底構築 | `train_svm.py` → `build_svm_projection.py` | `build_pca_projection.py` | `build_random_projection.py` |
| 射影 | `project_latents_svm.py` | `project_latents_pca.py` | `project_latents_svm.py`（流用） |
| 評価 | `evaluate_svm_subspace.py` | `evaluate_pca_subspace.py` | `evaluate_svm_subspace.py`（流用） |
| 効率版スイープ | — | `sweep_pca_dim.py`（k を一括で振る） | — |

- SVM 版と PCA 版の射影・評価スクリプトは**中身が同一**（基底の作り方だけが違う）。
- `sweep_pca_dim.py` は PCA の k を一括で振って精度曲線を得る効率版（中間 .pt を書かない）。
  本手順書は k=7 固定・3 手法横並びの比較に用いる。
