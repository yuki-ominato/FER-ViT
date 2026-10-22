# fer-vit: Claude Code 向けプロジェクトメモ

このファイルは Claude Code がセッション開始時に自動で読み込む。複数の端末で同じ前提を共有するためにリポジトリに置いている。
進行中の作業と次の手順は `document/session_handoff.md` に書く（作業の区切りごとに更新する）。

## やり取り
- ユーザーは日本語で書く。返答も日本語。
- 長いファイルの読み込みや全体把握のための調査は歓迎されている。

## 研究の目的
- 表情認識（FER2013, RAF-DB）で ViT がデータを大量に必要とする問題を緩和する。
- 画像を pSp / e4e で StyleGAN2 の W+（18×512）に変換し、18 層のベクトルをトークンとして ViT に入れる（LatentViT / LatentViTv2）。
- **主張したいこと**: データが少ない条件で LatentViT が画像ベースの ViT に勝つ。
- AFS（表情と個人性の分離）、SVM / PCA による成分分解、LWN / SPE / LEAM は、LatentViT を改善する試みとして行ったもの。いずれも明確な改善は出ていない。
- 結果はまだどこにも発表していない。

## 環境
| | 開発機（この端末） | 実行機 |
|---|---|---|
| パス | `/home/yuki/research2/fer-vit` | `/home/yulab-pc2/Desktop/yuki/fer-vit` |
| Python | `/home/yuki/anaconda3/envs/fer-vit/bin/python` | `/home/yulab-pc2/anaconda3/envs/fer-vit/bin/python` |
| GPU | GTX 1660 Ti 6GB（小さい学習・確認用） | 本番の学習用（他の人と共用で、空きが無いことがある） |
| データ | `latents/raf-db/{train,test}`（RAF-DB pSp）と smoke 用データのみ | pSp / e4e × FER2013 / RAF-DB の全 latent、全実験の出力 |

- torch 2.5.1, timm 1.0.17。
- `latents/`, `outputs/`, `pretrained_models/`, `third_party/`, `eval_results/`, `experiments_other/` は git 管理外。`experiments/` には config やサマリだけをコピーしている。
- 決定的アルゴリズムを使うため、学習時は `CUBLAS_WORKSPACE_CONFIG=:16:8` を設定する（実行用スクリプトは設定済み）。

## データ
- FER2013: train / val / test = 24399 / 4310 / 7178。
- RAF-DB: train / test = 12271 / 3068。公式の val split は無い。
- RAF-DB の検証用には `splits/raf-db_holdout10_seed0.json` を使う。train から 10% をクラス比を保って切り出したもので、`data/holdout_split.py` で作った。分割はファイル名で持つので pSp / e4e / AFS 抽出後のどれにも使える。

## 評価で注意すること
- `--holdout_split` を使っていない過去の RAF-DB の run は、test で best 選択と LR 調整をしている。そのため数値は楽観的に出ている。今後の比較は holdout で選び、test は最後に 1 度だけ評価する。
- 学習スクリプトの最終レポート（`final_metrics.accuracy`）は最終エポックのモデルの値。holdout 使用時の `test_*` は best モデルの値。
- ベースライン間で条件が揃っていない。LatentCNN は mixup α=1 が固定になっている。`train_image_vit.py` の `--use_pretrained` はアーキテクチャ指定の引数を無視する。
- seed による精度のばらつきは約 1.6pt ある。差を主張するときは 3 seed 以上で比べる。
- SVM の "Emotion Only ≈ Baseline" は、決定関数が射影成分にしか依存しないので、構成上必ずそうなる。
- AFS について:
  - FER 特化版の抽出器（`train/train_fer_extractor.py`）は崩壊していた（h(w) がほぼ定数）。
  - 論文再現版は、表情ではなく帽子・色・背景を転写していた。
  - `sty_fer*` の latent はどの抽出器で作ったか不明なので、やり直す予定。
- 層番号は 0 始まりで統一する。過去の `eval/baseline/pSp/layer_importance.png` は 1 始まりの目盛りで描かれている。
- その他の既知の問題:
  - `sefa/factorize.py` は mapping の `fc0` を使っているが、これは SeFa として誤り。
  - `LatentDecomposer` は直交していない方向への射影を足し合わせている。

## 作成元の記録（provenance）
- `afs/provenance.py` の仕組みで、次の情報を記録している:
  - 抽出器のチェックポイント
  - 抽出・生成した latent のディレクトリ（`provenance.json`）
  - 学習 config の `data.*_provenance`
- 既存の出力ディレクトリに別の作成元の .pt があると、書き込みを止める。

## 主なファイル
- 実験一覧: `document/experiment_summary.md`。`document/experiment_runs.csv` は `scripts/collect_experiments.py` で再生成する。
- AFS の再実験: `scripts/run_afs_pipeline.sh`
- 層の分析と LayerMix の比較: `scripts/run_layer_experiments.sh`。各段階で使うスクリプトは次のとおり:
  - holdout 分割: `data/holdout_split.py`
  - 層ごとの線形分類: `eval/layer_probe.py`
  - 学習: `train/train_latent_vit_v2.py` の `--holdout_split` と `--layer_mix_*`
  - 層アブレーション: `eval/analyze_layer_importance.py`

## 作業の進め方
- コミットはユーザーに頼まれたときだけ行う。作業ブランチは `feat/AFS`（main は `main`）。
- 実行機で長い学習を回すときは、まずローカルで smoke 用データ（`latents/*_smoke`）と `DRY_RUN=1` で動作を確認する。
