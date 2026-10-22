# セッション引き継ぎメモ

別の端末や新しい Claude Code セッションで作業を続けるためのメモ。作業の区切りごとに更新する。
プロジェクト全体の前提と注意点は、リポジトリ直下の `CLAUDE.md` にまとめてある。

最終更新: 2026-10-22（開発機）

## これまでの作業（2026-10-21〜22）
1. リポジトリ全体を読み、評価上の問題点を洗い出した（内容は `CLAUDE.md` の「評価で注意すること」）。
2. 167 run を整理した（`document/experiment_summary.md`, `document/experiment_runs.csv`）。
3. AFS 再実験の準備をした（コミット d74cefa）。
   - 作成元を記録する仕組み（`afs/provenance.py`）を入れた。
   - 一括実行スクリプト `scripts/run_afs_pipeline.sh` を作った。
4. 層の分析と LayerMix 拡張を実装した（今回）。

## 未完了・次の手順
### AFS 再実験（保留中）
- 実行機で `scripts/run_afs_pipeline.sh` を実行したところ、抽出器学習の最初のバッチで `NVML ... INTERNAL ASSERT FAILED` が出て止まった。
- 原因は GPU の使用率が高く、メモリを確保できなかったため（他の処理と共用）。
- GPU が空いたら再実行する。途中で止まった run の `outputs/extractor/raf-db_e4e/paper/20261021_160918/` にはチェックポイントが無いので、無視してよい。

### 層の分析と LayerMix（実装済み・GPU での学習は未実行）
- 背景: 過去の層アブレーション（`eval/baseline/pSp/layer_importance.json`、RAF-DB pSp、test で測定）では、層 9（0 始まり）を潰すと精度が 15.6pt 下がった。層 4–11 以外を潰したときの低下はほぼ 0 だった。
- 実装したもの:
  - `data/holdout_split.py`: train から検証用データを切り出す。RAF-DB 用の分割は作成済み（`splits/raf-db_holdout10_seed0.json`）。
  - `eval/layer_probe.py`: 層ごとの線形分類。
  - `train/train_latent_vit_v2.py` に次の 2 つを追加した:
    - `--holdout_split`: holdout で best を選び、test は最後に 1 度だけ評価する。
    - `--layer_mix_prob` / `--layer_mix_keep`: 層 4–11 を残し、それ以外の層をバッチ内の別サンプルと入れ替える。
  - `eval/analyze_layer_importance.py` に `--holdout_split` を追加した。
- 次にやること（実行機で）:
  ```bash
  bash scripts/run_layer_experiments.sh                                   # RAF-DB pSp
  ENCODER=e4e LATENT_ROOT=latents/raf-db_e4e bash scripts/run_layer_experiments.sh
  ```
  見るもの: 次の 2 つを 3 seed の平均で比べる。データ 10% での差を重視する。
  - `experiments/layer_mix/.../experiment_summary.json` の `final_metrics.test_accuracy`
  - 同じファイルの `best_f1_macro`（holdout での値）
- その後の候補:
  - 使う層を 4–11 に絞った学習
  - LayerMix で作った latent を generator で画像に戻し、表情が保たれているかを目で確認する
  - AFS の分解を中間層に限定する

## 層ごとの線形分類の結果（RAF-DB pSp、holdout 1226 枚で評価）
- 実行日と実行場所: 2026-10-22、開発機の CPU。
- 実行したコマンド:
  ```bash
  python eval/layer_probe.py --latent_dir latents/raf-db/train --holdout_split splits/raf-db_holdout10_seed0.json \
      --output_dir eval_results/layer_probe/raf-db_psp --data_fraction {1.0,0.1}
  ```
- 出力先は `eval_results/layer_probe/raf-db_psp/`（git 管理外）。
- 分類器は StandardScaler + ロジスティック回帰（C=0.1、class_weight=balanced）。
- 多数派クラスだけを当てた場合の精度は 0.389。

| 入力 | 精度（train 100%, 11045 枚） | F1（100%） | 精度（train 10%, 1102 枚） | F1（10%） |
|---|---|---|---|---|
| 層 0–3 を連結 | .587 | .477 | .507 | .375 |
| **層 4–11 を連結** | **.681** | **.571** | **.599** | **.475** |
| 層 12–17 を連結 | .587 | .484 | .495 | .370 |
| 全 18 層を連結 | .707 | .600 | .596 | .455 |

- 層単体で最も精度が高いのは層 9（100% で .587、10% で .508）。次いで層 5–7 が高い。層 0 から 9 にかけて精度は上がり、層 12 以降は単調に下がる。
- **データ 10% では、層 4–11 だけの方が全層より F1 が高い**（.475 と .455）。全層を使うと次元が増えて過学習するためと考えられ、層を絞る案を支持する結果。
- 層 0–3 と 12–17 も単体で多数派クラスの精度を超えており、表情の情報がまったく無いわけではない。LayerMix で入れ替えると、この分の情報が別サンプル由来になる（ラベルとずれる）。p=0.5 だけでなく、p を下げた条件も比べる価値がある。
