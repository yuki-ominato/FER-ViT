# 実行チェックリスト（ViT + pSp + StyleGAN2 for FER2013）

本チェックリストは、潜在生成 → 学習 → 評価 までを滞りなく回すための実務的な残タスクをまとめたものです。必要箇所を順に満たしてください。

---

## 0) 前提・環境
- [ ] conda 環境を作成・有効化（既存 `environment.yml`）
  - `conda env create -f environment.yml -n fer-vit && conda activate fer-vit`
- [ ] 追加依存を導入（requirements）
  - `pip install -r FER-ViT/requirements.txt`
- [ ] GPU/CUDA が利用可能（少なくとも 8–12GB 推奨）

---

## 1) 外部リポジトリと重み
- [ ] pixel2style2pixel を配置済み（例: `/home/yuki/research2/pixel2style2pixel`）
  - 既に `FER-ViT/models/encoder_wrapper.py` で絶対パスを自動追加済み
- [ ] `pixel2style2pixel/configs/paths_config.py` の重みパスを設定
  - StyleGAN2の `stylegan_weights`、IR-SE50の `ir_se50`、（必要なら）追加重み
- [ ] pSp のエンコーダ重みを配置
  - 例: `FER-ViT/pretrained_models/psp_ffhq_encode.pt`

---

## 2) データと前処理
- [ ] FER2013 を以下の階層で用意
```
dataset/fer2013/
  train/{angry,disgust,fear,happy,neutral,sad,surprise}/*.png
  val/{...}
  test/{...}
```
- [ ] 顔アライン（任意だが推奨）：MTCNN/dlib 等で 256x256 へ安定化
- [ ] 画像は RGB 化される前提（スクリプト側で ToTensor + Normalize 実施）

---

## 3) 潜在コード生成（w+ キャッシュ）
- [ ] train/val/test それぞれで latent を生成
```
PYTHONPATH=. python FER-ViT/scripts/generate_latents.py \
  --data_root dataset/fer2013/train \
  --latent_out FER-ViT/latents/train \
  --encoder_model FER-ViT/pretrained_models/psp_ffhq_encode.pt \
  --encoder_type psp \
  --batch_size 4

PYTHONPATH=. python FER-ViT/scripts/generate_latents.py \
  --data_root dataset/fer2013/val \
  --latent_out FER-ViT/latents/val \
  --encoder_model FER-ViT/pretrained_models/psp_ffhq_encode.pt \
  --encoder_type psp \
  --batch_size 4

PYTHONPATH=. python FER-ViT/scripts/generate_latents.py \
  --data_root dataset/fer2013/test \
  --latent_out FER-ViT/latents/test \
  --encoder_model FER-ViT/pretrained_models/psp_ffhq_encode.pt \
  --encoder_type psp \
  --batch_size 4
```
- [ ] 出力ファイル数が画像数と一致することを確認
- [ ] 失敗画像があればログを確認し、再実行

---

## 4) 学習（LatentViT）
- [ ] 最初の一周を小規模で動かし配線を確認（各クラス 20–50 枚など）
- [ ] 本学習を実行（`seq_len` は自動推定に対応）
```
PYTHONPATH=. python FER-ViT/train/train_latent_vit.py \
  --latent_train_dir FER-ViT/latents/train \
  --latent_val_dir FER-ViT/latents/val \
  --epochs 60 \
  --batch_size 64 \
  --lr 1e-4 \
  --use_class_weights \
  --scheduler plateau
```
- [ ] TensorBoard でログ確認：`tensorboard --logdir experiments/`
- [ ] ベストモデル保存のメッセージを確認（F1 マクロ）

---

## 5) 評価・可視化
- [ ] テスト評価＆可視化
```
PYTHONPATH=. python FER-ViT/eval/evaluate_model.py \
  --checkpoint_path experiments/<実験ディレクトリ>/checkpoints/best_model.pt \
  --latent_test_dir FER-ViT/latents/test \
  --output_dir FER-ViT/eval_results
```
- [ ] 生成物：混同行列、クラス別メトリクス、学習曲線、簡易Attention可視化、JSONレポート

---

## 6) よくある落とし穴（チェック）
- [ ] `paths_config.py` 未設定で pSp 初期化に失敗（StyleGAN2/IR-SE50 のパス）
- [ ] OOM：潜在生成バッチ・学習バッチを小さく（AMP未使用の場合特に）
- [ ] `seq_len` 不一致：latent の L と `LatentViT` の `seq_len` を一致（本実装は自動推定対応）
- [ ] クラス不均衡：`--use_class_weights` を有効化、必要なら FocalLoss へ拡張
- [ ] 絶対パス依存：他マシンで使う場合は `encoder_wrapper.py` 冒頭の `_ABS_PSP` を変更

---

## 7) 次の改善候補（任意）
- [ ] 早期停止（F1 マクロ監視）
- [ ] Mixed Precision（`torch.cuda.amp`）
- [ ] Warmup + Cosine スケジューラ
- [ ] 厳密な Attention 抽出（Transformer 層を改造して weights 取得）
- [ ] 潜在補間（Mixup/CutMix 代替）

---

## 8) サニティ・スクリプト（任意）
- [ ] サンプル潜在1件を読み `(L, D)` 形状/値域を print する短いスクリプト
- [ ] `LatentViT` にダミー `(B, L, D)` を通し出力 `(B, 7)` を確認

このチェックリストを満たせば、ViT + pSp + StyleGAN2 による FER2013 分類を安定稼働できます。必要に応じて、各ステップの自動化スクリプトも追加可能です。
