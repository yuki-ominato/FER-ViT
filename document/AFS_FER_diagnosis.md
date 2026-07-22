# AFS_FER 抽出器「表情が視覚的に分離できていない」問題の原因調査

作成日: 2026-07-05
対象: `outputs/afs_fer_raf-db/20260702_013713/checkpoints/best_model.pt`（`train/train_fer_extractor.py` + `afs/fer_losses.py::AFSFERLoss`）
比較対象: 論文再現版 `train/train_style_extractor.py` + `afs/losses.py::AFSLoss`

---

## 0. 観察された事実

- `eval/AFS/cross_class_grid.png`（論文再現版）: 各行（identity）内で列（expression source）を変えると、口の開き方など視覚的な変化が現れる。id_sim 平均 0.75（`eval/AFS/metrics.json`）で識別性も概ね保持。
- `eval/AFS_FER/cross_class_grid.png`（FER特化版）: 各行が7列すべてでほぼ完全に同一画像。expression source をどれに変えても生成画像に変化が出ない。
- `eval/AFS_FER/loss_acc.png`: 表情分類の Test Accuracy は 60〜66%（7クラスランダム ≈14%、多数派クラスのみ ≈30〜35%を上回る）。→ **h(w) には潜在空間上で表情ラベルを判別可能な情報が乗っている**にもかかわらず、画像として可視化すると変化が見えない、という矛盾した状態。

この矛盾を軸に、以下の原因候補を確度別に整理する。

---

## 1. 確度: 高（cross_class_grid + loss_acc.png の比較から直接裏付けられる）

### 1-1. L_expr / L_neutral が生の潜在ベクトルに対してのみ計算されている（最有力）

`afs/fer_losses.py` の `_l_expr` / `_l_neutral` は次の形になっている。

```python
logits_src = self.classifier(h_src)   # h_src = h(w) そのもの
logits_tgt = self.classifier(h_tgt)
```

Generator を一切通さず、`h(w)` という 18×512 の潜在ベクトルに対して直接、正則化なし・weight decay なしの小さな MLP (`ExprClassifier`) で分類している。CrossEntropyLoss は入力の絶対的なスケールを問わないため、**分類器側の重みを大きくして極小の潜在方向を増幅する**ことで、視覚的に無意味な差のまま 60% 台の精度を達成できてしまう。

この結果、h(w) が「分類器には読み取れるが、StyleGAN2 でデコードすると知覚できないほど微小」な方向に収束したと考えられる。論文再現版の `AFSLoss` は L_id/L_feat/L_lpips が **すべて G(w_new) を経由した画像ドメインの損失** であるため、この種の「潜在空間だけで完結する近道」が存在しない。

### 1-2. `document/AFS_FER.md` の設計と実装の乖離

設計ドキュメントが元々提案していたのは、生成画像に対して画像ベースの FER モデルを適用する損失だった。

```
L_expr = 1 - cos( R_FER(G(w_expr + w_rest^target)), R_FER(x_expr_source) )
```

しかし実装は前述の通り、画像を生成せず生の潜在ベクトルにしか触れていない。これは 1-1 の原因を生んだ直接的な設計変更点であり、意図的な簡略化だとしても、結果的に「視覚的に意味のある表情方向を作る」圧力を消してしまっている。

---

## 2. 確度: 中（副次的に効いている可能性が高い、単独では上記の矛盾を説明しきれない）

### 2-1. `ExprClassifier` の mean pooling が信号を希釈する

```python
def forward(self, w):
    return self.fc(w.mean(dim=1))   # 18層全体を平均
```

`EXPR_LAYERS`（4-11）だけでなく、`L_sparse` でゼロに寄せられている非表情10層まで含めて平均している。分類に必要な情報が薄まり、少ない・小さな変化でも損失を下げられる方向に誘導する。

### 2-2. `L_sparse` は非表情層の抑制のみで、表情層側の大きさを保証しない

```python
non_src = h_src[:, self.NON_EXPR_LAYERS, :]
return 0.5 * (non_src.abs().mean() + non_tgt.abs().mean())
```

非表情層（0-3, 12-17）をゼロに寄せる正則化はあるが、表情層（4-11）側に「画像として意味を持つ程度の大きさを持て」という逆方向の圧力は一切ない。結果として全体的に「小さい方が得」という力学に偏っている。

### 2-3. クラス不均衡への対応が一切ない

```python
self.ce = nn.CrossEntropyLoss()
```

RAF-DB は Happy に大きく偏った不均衡データセット。本プロジェクトの `train/train_image_vit.py` は `--use_class_weights` で対処済みだが、`AFSFERLoss` の L_expr/L_neutral には重み付けがない。60〜66%という精度も、多数派クラスへの偏りをある程度含んでいる可能性がある（loss_acc.png だけでは per-class breakdown が不明なため要確認）。

---

## 3. 確度: 低〜不確実（一般的な留意点。今回の現象の主因ではなさそうだが、記録として残す）

### 3-1. AFS論文が明言する「標準的な"再現しにくさ"」

論文 3.4節・図3（[arXiv:2211.10812](https://arxiv.org/abs/2211.10812)）は次のように述べている。

> "The 'style' images seem to have similar identities, and are mainly different in identity-unrelated facial attributes."

つまり **`w_sty` 単体をデコードした際に「互いに似た顔」に見えること自体は、論文でも起こりうる現象として明記されている**。ただし論文再現版の `eval/AFS/cross_class_grid.png` では列ごとの変化がちゃんと視認できているため、今回の FER 版の「完全に無変化」はこの一般論だけでは説明できず、上記1節の原因が主因と判断する。

### 3-2. e4e で RAF-DB（FFHQ外ドメイン）を反転していることによる画質劣化

両方の cross_class_grid とも、画像がぼやけて色味が不自然（オレンジ/ピンクがかる、毛皮の帽子のような偽アーティファクトが乗る）。RAF-DB はポーズ・表情が強く FFHQ の学習分布から外れているため、e4e 反転の品質が本来のFFHQ写真より低下している可能性がある。信号対雑音比を下げる方向には作用しうるが、1節の問題（潜在空間では分離できているのに画像に一切出ない）を単独で説明するものではない。

### 3-3. epoch数・batch_sizeが少ない（論文デフォルト: 10 epoch, batch_size 4）

学習不足の可能性は常にあるが、`loss_acc.png` の Train Loss は既に十分小さく収束しているように見えるため、単純な学習不足というより「間違った目的関数に対して収束している」と考える方が整合的。

---

## 4. 論文再現版（train_style_extractor.py）は大丈夫か

`eval/AFS/cross_class_grid.png` と `metrics.json`（id_sim 平均 0.75, cons 平均 0.065）を見る限り、少なくとも「列を変えると画像が変わる」という最低限の分離は機能している。1節で挙げた「潜在空間だけで完結する近道」が存在しないアーキテクチャ（L_id/L_feat/L_lpipsが全てG(w_new)経由）であるため、FER版と同じ形の破綻は起きにくいと考えられる。

---

## 5. 画像ベースFERモデルで評価する場合、論文再現版との相違点

「G(w_new) を生成 → 画像ベースの FER モデルでラベル一致を評価する」という修正を導入した場合、論文再現版（AFSLoss）との構造的な違いは以下の通り。

| 観点 | 論文再現版 (AFSLoss) | 画像ベースFER損失を導入した場合 |
|---|---|---|
| 教師信号の単位 | **インスタンス単位**。`L_feat`/`L_lpips` は特定の target 画像 `x^t` / `G(w^t)` そのものに画素・特徴レベルで一致させる | **クラス単位**。ターゲットの具体的な写真ではなく、感情ラベル `label_tgt` に一致するかどうかだけを問う |
| 使うモデル | ArcFace（identity）、StyleGAN2中間特徴（層5, 32×32）、LPIPS(AlexNet) — いずれも「誰か」「何に似ているか」を測るモデル | 画像ベースのFER分類器（本プロジェクトなら `train/train_image_vit.py` の学習済みViT等）を **ArcFaceと同じ立ち位置で** 追加する必要がある |
| 何を disentangle しようとしているか | 「identity」 vs 「identity以外すべて(pose・照明・髪型・表情を含む)」という大きな二分法 | 論文の「style」サブスペースを **さらに「表情」と「表情以外(pose・照明・髪型)」に細分化** しようとしている。これは論文が扱っていない、より難しい追加の disentangle 課題 |
| 勾配の通り道 | 既に `G(w_new)` を経由しており、`feat_hook` の使い回しなど配線は完成済み | 同様に `G(w_new)` の出力をFERモデルに通す新しい経路が必要。ArcFaceExtractor同様、**frozen・微分可能・eval固定**なFERモデルのラッパーを追加する必要がある（現状の `_l_expr`/`_l_neutral` はここを素通りして潜在ベクトルに直接アクセスしている） |
| 前処理の一致 | ArcFace: crop `[35:223, 32:220]` → 112×112。LPIPS: 256×256のまま | FERモデルの入力仕様（例えば ViT は 224×224、ImageNet正規化）に合わせた別系統の前処理が必要。ArcFace/LPIPSの256px `[-1,1]` 正規化とは異なる可能性が高い |
| ラベル情報の要否 | 不要（教師なし。identity/style の分離に感情ラベルは一切使わない） | 必須（`label_tgt` に対する CrossEntropy、または FER埋め込みのcos類似度など） |
| 既存の L_sparse / EXPR_LAYERS 制約 | 存在しない | 「表情は中間解像度層に集中する」という仮説に基づく制約だが、画像ベース損失を導入するなら、この制約が本当に必要か（画像ドメインの圧力だけで十分に表情層へ収束するか）を再検証すべき |
| 計算コスト | Generator 2回 + ArcFace 2回 + LPIPS 1回 | 上記に加えて FERモデルの forward が最低1回（`G(w_new)` に対して）追加。理想的には `G(w_src)` に対しても計算し、ターゲットラベルとの一致だけでなく「ソースの表情が消えているか」も見たいなら計算量がさらに増える |

### まとめ

論文再現版は「識別対象＝身元」を固定した上で「それ以外すべて」を画像インスタンス単位で一致させる、比較的単純な二値disentangleである。FER版でやろうとしているのは、その「それ以外すべて」の中からさらに「表情」だけを取り出すという、論文が想定していない一段階難しい課題であり、そのための教師信号（画像ベースFERモデル）を導入すると、ArcFaceと同格の「もう一つの frozen 参照モデル」を新設し、既存の潜在空間のみで完結していた `_l_expr`/`_l_neutral` をこちらに置き換える（あるいは併用する）大きな設計変更になる。
