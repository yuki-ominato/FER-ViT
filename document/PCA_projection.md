# PCA感情部分空間分離 実装設計書

`document/InterFaceGAN_SVM.md`（SVM係数から感情方向を抽出する版）に対応する、
PCAベースの部分空間分離。分離・射影・評価の枠組み自体はSVM版と同一で、
「感情部分空間の基底ベクトルNをどう構築するか」だけが異なる。

| | SVM版 | PCA版 |
|---|---|---|
| 基底の構築元 | LinearSVCの`coef_`（7クラス分の識別方向） | 全データPCAの主成分のうち、ラベルとのANOVA F値が高い上位k個 |
| 基底構築スクリプト | `train_svm.py` → `build_svm_projection.py` | `build_pca_projection.py`（PCA適合＋成分選択を1スクリプトで完結） |
| 射影・評価スクリプト | `project_latents_svm.py` / `evaluate_svm_subspace.py` | `project_latents_pca.py` / `evaluate_pca_subspace.py`（射影の数式・保存フォーマットはSVM版と完全に同一） |
| 学習時オンザフライ射影 | `--svm_basis` / `--svm_projection` | `--pca_basis` / `--pca_projection`（`--svm_basis`と同時指定不可） |

---

## 1. 研究目的

SVM版が「線形分類境界（教師ありの識別方向）」で感情部分空間を定義するのに対し、
PCA版は「データの分散が大きい方向（主成分）のうち、感情ラベルと相関の高いもの」で
感情部分空間を定義する。両者を比較することで、感情情報がW+潜在空間上で

- 識別的な方向（SVM）
- 分散の大きい方向のうち感情に相関するもの（PCA）

のどちらに、どの程度の強さで存在するかを検証できる。

---

## 2. 基底構築のアルゴリズム（`build_pca_projection.py`）

```text
train latents (N, 18, 512) → flatten (N, 9216)
 ↓
PCA(n_components) を適合、上位 n_components 個の主成分を取得
 ↓
各主成分のサンプルごとの射影値 coords (N, n_components) を計算
 ↓
sklearn.feature_selection.f_classif(coords, y) で
各主成分と感情ラベルの ANOVA F値を計算
 ↓
F値が高い上位 k 個（デフォルト 7、SVM版のクラス数と揃える）を選択
 ↓
N = pca.components_[top_k].T   # (9216, k)  各列はすでに単位ベクトル
```

SVM版の `N = clf.coef_.T`（各列をL2正規化）と比べると、PCA版の列は
PCAの直交性により構築時点で自動的に正規直交（Gram行列がほぼ単位行列）になる。

`--n_components`（デフォルト100）は「この中から上位k個を選ぶ」ための探索プールのサイズで、
k（デフォルト7）そのものではない点に注意。

---

## 3. 射影（`project_latents_pca.py`）

SVM版と全く同じ数式（Nが直交していなくても成立する一般形）。

```python
pinv_N = np.linalg.pinv(N)        # (k, 9216)
coords = pinv_N @ w_flat          # (k,)
w_emotion = N @ coords            # (9216,)
w_residual = w_flat - w_emotion   # (9216,)
```

出力フォーマットもSVM版と同一（`latent`/`emotion_latent`/`residual_latent`/`label`）。
そのため実装上は `evaluate_svm_subspace.py` をPCA版の出力にそのまま使うこともできるが、
命名の分かりやすさのため専用の `evaluate_pca_subspace.py` を用意している。

---

## 4. 実行コマンド

```bash
# 1. PCA基底を構築（PCA適合＋ラベル相関による成分選択を1コマンドで実行）
python latent_analysis/build_pca_projection.py \
    --latent_dir latents/train \
    --output_dir latent_analysis/pca_output \
    --n_components 100 \
    --k 7

# 2. 全スプリットを射影
python latent_analysis/project_latents_pca.py \
    --basis latent_analysis/pca_output/emotion_basis_pca.pt \
    --latent_root latents \
    --output_root latents_pca

# 3. 分離品質を評価（Baseline / Emotion Only / Residual Only を LinearSVC で比較）
python latent_analysis/evaluate_pca_subspace.py \
    --latent_root latents_pca \
    --train_split train \
    --eval_split test
```

```bash
# 感情成分のみで LatentViT v2 を学習
python train/train_latent_vit_v2.py \
    --latent_train_dir latents/train \
    --latent_val_dir   latents/val \
    --pca_basis        latent_analysis/pca_output/emotion_basis_pca.pt \
    --pca_projection   emotion

# 非感情成分（アイデンティティ等）のみで学習
python train/train_latent_vit_v2.py \
    --latent_train_dir latents/train \
    --latent_val_dir   latents/val \
    --pca_basis        latent_analysis/pca_output/emotion_basis_pca.pt \
    --pca_projection   residual
```

`--svm_basis` と `--pca_basis` は同時指定不可（`train_latent_vit_v2.py`起動時にエラーになる）。

---

## 5. 評価項目・期待される結果

`document/InterFaceGAN_SVM.md` と同じ指標（Accuracy / Macro F1 / Balanced Accuracy）で
Baseline / Emotion Only / Residual Only を比較する。SVM版との相対比較で、

- PCA版の Emotion Only が SVM版に劣らない性能を示せば、
  「教師なしで見つかる分散最大方向の一部が、既に感情情報と強く相関している」ことを示せる。
- 逆に PCA版の Emotion Only が大きく劣る場合、
  「感情情報は分散の大きい方向には現れず、SVMのような教師あり識別方向でないと
  抽出できない（=支配的な変動要因ではない）」ことを示す傍証になる。

---

## 6. 実行結果と考察（2026-07-23、FER2013）

`build_pca_projection.py`（train, N=24399, n_components=100, k=7）で構築した基底を
`project_latents_pca.py`で全スプリットに射影し、`evaluate_pca_subspace.py`
（train split で学習 / test split で評価, N_test=7178）で評価した結果。

### 基底構築時の統計（参考）

選択された主成分: PC0, 3, 5, 8, 10, 23, 24（探索プール100件中）。F値は224→96と
緩やかに減衰しており明確な「崖」はなく、explained varianceも11.8%（PC0）〜0.74%（PC24）
まで一桁以上の開きがあった（詳細は本ドキュメント作成前のやり取りを参照）。

### 分類精度

| Variant | Accuracy | Macro F1 | Bal. Acc | Baseline比 |
|---|---|---|---|---|
| Baseline（全9216次元） | 0.4944 | 0.4707 | 0.4770 | ± 0 |
| Emotion Only（PCA選択7次元） | 0.3232 | 0.2042 | 0.2385 | **-17.1pt / -26.6pt / -23.9pt** |
| Residual Only（残り9209次元） | 0.4586 | 0.4393 | 0.4518 | -3.6pt / -3.1pt / -2.5pt |

Emotion Onlyのクラス別レポートでは `angry` と `disgust` の precision/recall が **0.00**
となり、7次元では少数派クラスを一切分離できず `happy`（recall 0.76）など一部クラスへの
予測に潰れている。

### 考察

1. **PCA版の部分空間は感情情報をほとんど集約できていない。**
   Emotion Onlyの性能はBaselineから大きく劣化する一方、Residual OnlyはBaselineに
   近い性能を維持した。理想的な分離（Emotion Only ≈ Baseline、Residual Only ≪ Baseline）
   とは正反対の傾向であり、「7次元のPCA基底に感情の識別情報を追い出し、残りには
   ほとんど残らない」という当初の仮説は成立しなかった。

2. **原因は基底構築の手順そのものにある可能性が高い。**
   PCA版は「まずラベルと無関係に分散最大方向を求め、その中からF値で上位k個を
   事後的に選ぶ」という2段階の手続きを取る。真に感情を分離する方向が、たまたま
   上位100個の分散最大方向のいずれとも一致しない（＝データ全体の支配的な分散軸は
   別の要因、例えば個人差やポーズが占めている）場合、PCA基底の中にそもそも
   良い方向が存在せず、F値による事後選択では原理的に発見できない。
   これに対しSVM版は識別境界を直接最適化するため、分散の大小に関わらず
   感情を分離する方向を見つけられる。今回の結果は、本ドキュメント冒頭の仮説2
   （「感情情報は分散の大きい方向には現れず、教師あり識別方向でないと抽出できない」）
   を支持する結果と言える。

3. **k=7という次元数の小ささも寄与している可能性がある。**
   基底構築時点でF値の減衰が緩やかだった（224→96、明確な崖なし）ことから、
   感情と相関する情報は少数の支配的な方向ではなく、多数の弱い方向に分散している
   可能性がある。k=7ではこれらの大部分を捨ててしまうため、kを増やす、あるいは
   PCA成分の線形結合（LDAなど教師ありの次元削減）を試すことで改善する余地はある。

4. **`document/AFS_FER_diagnosis.md`で確認した傾向と整合する。**
   W+潜在空間上で感情情報が少数の明確な方向に線形分離可能な形で存在しているわけ
   ではなく、教師あり学習（分類境界の直接最適化）で初めて取り出せる、分散した・
   非線形的な情報である可能性を示す追加の傍証となっている。

### 次のアクション候補

- SVM版（`evaluate_svm_subspace.py`）の同一設定（FER2013, train/test split）での
  結果を取得し、Emotion Only / Residual Onlyの数値を直接比較する。
- `--k`を増やして（例: 20, 50）Emotion Onlyの性能がBaselineにどこまで近づくか確認し、
  「何次元あれば分散ベースの基底でも感情情報を集約できるか」を検証する。
- PCA成分の単純な上位k選択ではなく、LDA（線形判別分析）など教師あり次元削減を
  基底構築に用いるバリエーションを試す。
