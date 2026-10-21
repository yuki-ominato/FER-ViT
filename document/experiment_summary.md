# 実験記録の整理（2026-10-21 時点）

`experiments/` と `experiments_other/` にある **167 run** の `config.json`、`experiment_summary.json`、TensorBoard ログを機械的に読み出し、実験の目的ごとに並べ直したもの。
全 run の一覧は `document/experiment_runs.csv` にある。再集計するには `python scripts/collect_experiments.py` を実行する。

---

## 0. 読み方と注意

| 項目 | 内容 |
|---|---|
| 表の数値 | 特記がなければ `final_metrics.accuracy`。これは**最終エポックのモデルで評価した val 精度**で、best モデルの値ではない。括弧内は macro F1 |
| FER2013 の評価 | val split（4310 枚）。test（7178 枚）で評価した run は無い |
| RAF-DB の評価 | `val_dir` が **test split**。best の選択と LR スケジューラも test を見ているため、楽観的な値 |
| ラベルスムージング | latent 系の学習スクリプトは config に `label_smoothing` を保存していない。画像系は全 run で 0.1 |
| コードのバージョン | config に git commit が残っていない。例えば E7 と E7_v3 は config も結果も完全に同一（後述） |
| シード | ほぼ全 run が seed 42 の 1 本。同じ config の再実行では結果が完全に一致する（決定論的） |
| FER2013 の latent | `latents/train` と `latents/fer2013/train` の 2 種類の表記がある。10% 学習での精度は 0.399 と 0.400 で、ほぼ同じだが完全には一致しない |

### 重複（同一 config・同一結果。集計では片方だけを数える）
- `experiments/depth`、`dropout_0.3`、`dropout_0.3_depth2` の計 20 run ＝ `experiments_other/` の同名 run のコピー
- `baseline/pSp`、`baseline/e4e` ＝ `lwn-spe-leam/E0/frac100_rafdb`、`E0/frac100_rafdb_e4e`
- `base/latent_vit_..._20251201_114659` と `experiments_other/frac/frac100/latent_vit_..._145701` は、同じ config で再実行して結果も一致した例
- `random_emotion_k7_seed42` ＝ `svm-pca-random/random_emotion_k7`
- `svm/frac100` ＝ `svm/no-id_frac100/20260626_204740`
- `afs/raf-db-a/fer_expr` の 2 run、`fer_expr_2` の `sty_fer2` 系 2 run、`lwn-spe-leam/E3/frac100` の 2 run

### 結果が無い run（summary が無く、中断したとみられる）
`afs/20260625_120301` と `afs/fer2013-a`・`fer2013-b`（いずれも抽出器学習 run で、train_log はある）、`afs/fer2013-a/original` の 2 run、`latent_vit_..._afs_frac100/20260625_204423`、`E7_v3/20260618_175724`、`pretrained-image-cnn/..._frac100` の 2 run（RAF-DB の `231532` と FER2013 の `232509`）、`scratch-image-vit/image_vit_d12_..._frac100`、`svm_emotion_only` の 2 run

---

## 1. データ効率比較（主張 (a) の中心）

### 1-1. FER2013（評価は val）

| 手法 | 設定 | 10% | 25% | 50% | 100% | 出典 |
|---|---|---|---|---|---|---|
| **LatentViT d6**（pSp） | 60ep, cw, plateau | **0.399** (.352) | 0.455 (.415) | 0.513 (.478) | 0.555 (.546) | experiments_other/frac (2025-12-01) |
| LatentViT d6（再実行） | 同上、mixup=0 | 0.393 (.346) | 0.451 (.408) | 0.498 (.466) | 0.553 (.536) | latent/ (2025-12-12) |
| LatentViT d2 | 同上 | 0.399 (.349) | 0.453 (.406) | 0.501 (.458) | 0.539 (.505) | latent/ (2025-12-10) |
| ImageViT d6 スクラッチ、**拡張なし** | 200ep, cw, warmup_cos | 0.229 (.198) | 0.300 (.264) | 0.357 (.316) | 0.462 (.414) | image_scratch/ |
| ImageViT d2 スクラッチ、拡張なし | 200ep | 0.218 | 0.267 | 0.350 | 0.440 | image_scratch/ |
| ImageViT d6 スクラッチ、**拡張あり** | 200ep, cw, warmup_cos | 0.264 (.241) | 0.447 (.396) | **0.550** (.504) | **0.616** (.589) | experiments_other/frac |
| ImageViT ViT-S **ImageNet 事前学習** | 50ep, 拡張なし | 0.475 (.448) | 0.578 (.525) | 0.660 (.627) | 0.698 (.684) | pre_trained/ |
| LatentCNN standard | 60ep, cw, mixup α=1 固定 | 0.139（崩壊） | 0.377 | 0.439 | 0.477 | CNN/frac |
| ResNet18 スクラッチ | 100ep, 拡張なし | 0.444 | – | – | 0.610 | scratch-image-cnn/, 2d-cnn/ |
| ResNet18 事前学習 | 100ep, 拡張なし | 0.546 | – | – | 0.652 | pretrained-image-cnn/, 2d-cnn/ |

補足:
- `pre_trained/` の config には `model_size: custom, depth 6` と記録されているが、`n_parameters` = 21,668,359 は timm の ViT-S/16（7 クラス）と一致する。実体は事前学習済みの ViT-S。
- `eval/plot_data_fraction.py` に手入力されている値は、それぞれ次の run と一致する。
  - Latent ViT → **LatentViT d2**（latent/）
  - Latent CNN → CNN/frac
  - Scratch Image ViT → **ImageViT d6 拡張なし**（image_scratch/）
  - Pre-trained Image ViT → pre_trained/

  LatentViT と画像 ViT で depth が揃っておらず、画像 ViT には拡張ありの run が別にある。

**読み取れること**
- 10% と 25% では、LatentViT がスクラッチの ImageViT（拡張の有無を問わず）を大きく上回る（10% で +13〜17pt）。
- 50% 以上では、**拡張ありのスクラッチ ImageViT が LatentViT を逆転する**（100% で 0.616 対 0.555）。LatentViT は FER2013 で約 55% に頭打ちしている。
- ImageNet 事前学習の ViT と ResNet18（スクラッチ含む）は、全データ量で LatentViT を上回る。

### 1-2. RAF-DB（評価は test。val として兼用しているため楽観値）

| 手法 | 設定 | 10% | 100% | 出典 |
|---|---|---|---|---|
| **LatentViT d6**（pSp） | 60ep, cw | **0.644** (.520) | 0.741 (.629) | latent_vit_..._frac10/…_raf-db, baseline/pSp |
| LatentViTv2 E0（pSp、v2 スクリプト） | 同上 | 0.641 (.506) | 0.741 (.629) | base/frac10, baseline/pSp |
| LatentViT d6（e4e） | 同上 | 0.618 (.496) | 0.728 (.634) | …_frac10/20260724_044137, baseline/e4e |
| ImageViT d6 スクラッチ、拡張なし | 50ep, cw | 0.244 (.201) | 0.630 (.519) | scratch-image-vit/ |
| ImageViT ViT-S スクラッチ（lr 1e-3、拡張あり） | 100ep | 0.045（崩壊） | 結果なし | scratch-image-vit/image_vit_d12_… |
| ImageViT ViT-S **ImageNet 事前学習** | 50ep, 拡張なし | 0.601 (.482) | 0.845 (.753) | pretraind-image-vit/ |
| ResNet18 スクラッチ | 50ep（10%）／100ep（100%）、拡張なし | 0.648 (.459) | 0.817 (.719) | 2d-cnn/raf-db, scratch-image-cnn/ |
| ResNet18 事前学習 | 50ep, 拡張なし | 0.733 (.576) | 0.854 (.770) | 2d-cnn/raf-db |

**読み取れること**
- 10% では、LatentViT（0.644）がスクラッチ ViT（0.244）を大きく上回る。さらに **ImageNet 事前学習の ViT（0.601）も上回る**（macro F1 でも 0.520 対 0.482）。ResNet18 スクラッチとは精度でほぼ同等（F1 は LatentViT が上）、事前学習済みの ResNet18 には負ける。
- 100% では、画像ベースの全ベースライン（スクラッチ ViT を除く）が LatentViT を大きく上回る。
- pSp の方が e4e より 1〜3pt 高い。

---

## 2. LatentViT のハイパーパラメータ（FER2013、評価は val）

| 変更点 | 100% | 10% |
|---|---|---|
| d6, dropout 0.1, 60ep（基準） | 0.555 | 0.399 |
| seed 123 ／ seed 456（d6, 100%） | 0.563 ／ 0.571 | – |
| d3 | 0.551 | 0.396 |
| d2 | 0.538 | 0.389 |
| d6, dropout 0.3 | 0.542 | 0.385 |
| d2, dropout 0.3, 60ep ／ 100ep | 0.520 ／ 0.545 | 0.377 ／ 0.384 |
| エポック数 10 / 20 / 40 / 60 / 100（d6） | 0.401 / 0.527 / 0.550 / 0.555 / 0.553 | – |

→ seed による差は約 1.6pt（0.555〜0.571）。depth と dropout による差は、この seed 差とほぼ同じ大きさ。

---

## 3. LatentViT の拡張

### 3-1. HybridLatentViT（ImageNet の ViT ブロックを latent トークンに転用、FER2013 100%）
| 設定 | 精度 |
|---|---|
| ViT-B 事前学習, lr 1e-3, bs128, cw | 0.529 |
| ViT-B 事前学習, lr 5e-5, bs32 | 0.533 |
| ViT-S 事前学習, lr 5e-5, bs32 | 0.528 |
| ViT-S スクラッチ, lr 1e-3, cosine | 0.477 |

→ どれもスクラッチの LatentViT d6（0.555）に届かない。

### 3-2. LWN / SPE / LEAM のアブレーション（FER2013 10%）
**注意:** この 8 条件は **cw=False, mixup=1.0** で、§1 の LatentViT（cw=True, mixup=0）とは条件が異なる。

| ID | モジュール | 精度 (F1) |
|---|---|---|
| E0 | なし | 0.424 (.367) |
| E1 | LWN | 0.390 (.331) |
| E2 | SPE | 0.426 (.367) |
| E3 | LEAM | 0.422 (.363) |
| E4 | LWN+SPE | 0.387 ／ 0.412（2 run で値が違う。コードの版の違いとみられる） |
| E5 | SPE+LEAM | 0.421 (.364) |
| E6 | LWN+LEAM | 0.393 (.309) |
| E7 | 全部 | **0.251 (.057)＝崩壊**（E7_v3 も同値） |

100% での E0 対 E3（LEAM）: FER2013 e4e は 0.549 対 0.536、RAF-DB pSp は 0.741 対 0.732、RAF-DB e4e は 0.728 対 0.724。

→ どのモジュールも効果なし、または悪化。LWN が入ると下がる傾向がある。ただし、E0（mixup=1, cw なし）は §1 の同条件 LatentViT（0.399）より +2.5pt 高く、**Mixup が 10% 学習で効いている可能性がある**。

---

## 4. 成分分離（LatentViTv2 を射影後の latent で学習）

### 4-1. SVM / PCA / ランダムの 7 次元部分空間
| データ | 入力 | 精度 (F1) |
|---|---|---|
| FER2013 pSp | 射影なし（基準） | 0.555 |
| | SVM emotion（基底 `svm_output`、6/26）／（基底 `svm_output_fer2013`、7/1） | 0.442 ／ 0.472 |
| | SVM residual | 0.543 |
| | SVM emotion, 10% | 0.229 |
| RAF-DB pSp | 射影なし（基準） | 0.741 (.629) |
| | SVM emotion ／ residual | 0.684 (.587) ／ 0.738 (.634) |
| | PCA emotion ／ residual | 0.345 (.259) ／ 0.709 (.605) |
| | ランダム emotion（seed 41/42/43） | 0.092 / 0.087 / 0.071 |
| | ランダム residual | 0.741 |

- ランダム emotion は**チャンスレベル以下**（多数派の Happy だけを予測すれば 38.6%）。情報量を測れているのではなく、学習が失敗しているとみられる。7 次元座標のスケールが小さすぎる可能性がある。対照群としては LinearSVC での評価（`evaluate_svm_subspace.py`）の方が適切。
- residual が基準とほぼ変わらない点は、前回指摘した「感情情報が他の方向にも重複している」という解釈と合う。

### 4-2. AFS
**抽出器の学習 run**（記録に残っているのはすべて論文再現版 `train_style_extractor.py`）

| run | 内容 | 最終エポックの損失 |
|---|---|---|
| afs/20260625_120301 | FER2013 pSp, provider a, 50ep | val id 0.091, lpips 0.204 |
| afs/fer2013-b | FER2013 pSp, provider b, 50ep | val id 0.463, lpips 0.412 |
| extractor/fer2013/e4e | FER2013 e4e, 10ep | val id 0.325, lpips 0.100 |
| extractor/rafdb/e4e | RAF-DB e4e, 10ep | train_log なし |

FER 特化版（`train_fer_extractor.py`）の学習 run は `experiments/` には無い（出力先が `outputs/afs_fer*` のため）。

**抽出済みの latent で LatentViTv2 を学習した結果**
| データ | 入力 | 精度 (F1) |
|---|---|---|
| FER2013 e4e | 基準 ／ w−h(w)（`*_id_fer`）／ h(w)（`*_sty_fer`） | 0.549 ／ 0.545 ／ 0.415 |
| RAF-DB e4e | 基準 | 0.728 (.634) |
| | `sty_fer` | 0.530 ／ 0.627 |
| | `sty_fer_2` | **0.733** (.633) |
| | `sty_fer2`（アンダースコアなし、別物？） | 0.689 |
| | `sty_fer_3` | 0.645 ／ 0.655 |
| | `sty_fer2`, 10% | 0.657（e4e の基準 10% は 0.618） |
| RAF-DB pSp | オンザフライの h(w)（`train_latent_vit_afs.py`, mixup=1） | 0.298 |

→ 100% では、h(w) を入力にしても基準を超えない（最良の `sty_fer_2` でも同等）。10% で +3.9pt という結果が 1 本あるが、seed 1 本で比較相手のスクリプトも違うため、現時点では判断できない。

---

## 5. 主張 (a) に向けて足りないもの

1. **評価プロトコルの統一**
   - RAF-DB は train から val を切り出す（例: 10%、層化抽出）。
   - 全手法で「val で best を選び、test で 1 回だけ評価」に揃える。
   - FER2013 も test split で報告する。
2. **ベースラインの条件を揃える**
   - 画像 ViT のスクラッチ版は、拡張ありを標準にする（拡張なしは弱すぎて、査読で指摘されうる）。
   - エポック数（latent 60 対 画像 200/50）と cw・mixup の条件も揃えるか、各手法ごとに最適化したことを明記する。
3. **比較の位置付け**: pSp と StyleGAN2 は FFHQ で学習済みなので、LatentViT は「事前知識あり」の手法と見なされうる。したがって、主な比較相手は事前学習済みの ViT / ResNet になる。現状は RAF-DB 10% で事前学習 ViT に勝っており、これが最も有望な主張材料。
4. **少データ側を延ばす**: 1% / 5% / 10% / 25%。優位性は少データ側で大きくなる傾向がある。
5. **seed 3 本以上**: seed による差が約 1.6pt あるため、それより小さい差は主張できない。
6. latent 側のデータ拡張: 現状は latent 側に拡張がなく、画像側だけに拡張がある。
