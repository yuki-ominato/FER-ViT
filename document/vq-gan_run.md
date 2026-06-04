[cite_start]ご要望に合わせて、論文 "Taming Transformers for High-Resolution Image Synthesis" (CVPR 2021) [cite: 5] の内容に基づいた実験手順書を作成しました。
この論文の公式実装（`CompVis/taming-transformers`）を想定した構成になっています。

以下の内容をコピーして、Markdownファイル（例: `experiment_guide.md`）として保存し、使用してください。

-----

# VQGAN + Transformer 実験手順書

[cite_start]本ドキュメントは、CVPR 2021で発表された "Taming Transformers for High-Resolution Image Synthesis" [cite: 5] に基づく、高解像度画像合成モデルの学習および評価の手順書です。

## 概要

[cite_start]本手法は2段階のアプローチで構成されています [cite: 59]。

1.  [cite_start]**Stage 1 (VQGAN):** CNNを用いて、画像を知覚的にリッチな離散コードブック（Vocabulary）に圧縮するモデルを学習します [cite: 17, 33]。
2.  [cite_start]**Stage 2 (Transformer):** Stage 1で得られたコードブックのインデックス列の構成（Composition）を、Transformerを用いて自己回帰的に学習します [cite: 17, 34]。

## 0\. 環境構築

まず、必要なリポジトリと依存ライブラリをセットアップします。

```bash
# リポジトリのクローン (論文のProject page記載のリンクに基づく)
git clone https://github.com/CompVis/taming-transformers
cd taming-transformers

# 仮想環境の作成 (Conda推奨)
conda env create -f environment.yaml
conda activate taming

# 必要な場合、PyTorchを環境に合わせて再インストール
# pip install torch torchvision ...
```

-----

## 1\. データセットの準備

[cite_start]学習に使用する画像データセットを準備します。論文ではImageNetやFacesHQなどが使用されています [cite: 190, 221]。ここでは一般的なカスタムデータセットの使用を想定します。

### ディレクトリ構成

画像を1つのフォルダにまとめるか、学習用(`train`)と検証用(`validation`)のテキストファイルリストを作成します。

```text
data/
  └── my_dataset/
      ├── train/       # 学習用画像 (.jpg, .png)
      └── validation/  # 検証用画像
```

### 設定ファイル (Config) の作成

`configs/custom_vqgan.yaml` を作成し、データパスを指定します。

```yaml
model:
  base_learning_rate: 4.5e-6
  target: taming.models.vqgan.VQModel
  params:
    embed_dim: 256
    [cite_start]n_embed: 1024  # コードブックのサイズ |Z|=1024 [cite: 175]
    ddconfig:
      double_z: False
      z_channels: 256
      resolution: 256
      in_channels: 3
      out_ch: 3
      ch: 128
      ch_mult: [ 1,1,2,2,4]  # ダウンサンプリング設定
      num_res_blocks: 2
      attn_resolutions: [16]
      dropout: 0.0
    lossconfig:
      target: taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator
      params:
        disc_conditional: False
        disc_in_channels: 3
        disc_start: 10000
        disc_weight: 0.8
        codebook_weight: 1.0

data:
  target: main.DataModuleFromConfig
  params:
    [cite_start]batch_size: 4  # GPUメモリに合わせて調整 (GPT2-mediumと併用時は12GB VRAMでbatch 16程度が限界 [cite: 175])
    num_workers: 4
    train:
      target: taming.data.custom.CustomTrain
      params:
        training_images_list_file: "data/my_dataset/train.txt" # またはフォルダパス
        size: 256
    validation:
      target: taming.data.custom.CustomTest
      params:
        test_images_list_file: "data/my_dataset/val.txt"
        size: 256
```

-----

## 2\. Stage 1: VQGANの学習

CNNベースのエンコーダ・デコーダと、敵対的損失（GAN Loss）を用いてコードブックを学習します。

### 学習のポイント

  * [cite_start]**目的:** 画像を効率的に圧縮し、再構成品質を保つこと [cite: 121]。
  * [cite_start]**損失関数:** 知覚的損失（Perceptual Loss）とパッチベースのDiscriminatorによる敵対的損失を組み合わせます [cite: 135]。
  * [cite_start]**ダウンサンプリング率 ($f$):** 論文では $f=16$ が推奨されています（画像を$1/16$のサイズに圧縮）[cite: 236]。

### 実行コマンド

```bash
# VQGANの学習開始
python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,
```

学習が完了すると、チェックポイント（例: `logs/YYYY-MM-DD.../checkpoints/last.ckpt`）が保存されます。これがStage 2で必要になります。

-----

## 3\. Stage 2: Transformerの学習

Stage 1で学習したVQGANを固定し、その潜在表現（インデックス列）の並び方をTransformerに学習させます。

### 設定ファイル (Config) の作成

`configs/custom_transformer.yaml` を作成します。ここで**Stage 1の重みへのパス**を指定することが重要です。

```yaml
model:
  base_learning_rate: 4.5e-6
  target: taming.models.cond_transformer.Net2NetTransformer
  params:
    first_stage_key: image
    cond_stage_key: class_label # 条件付き生成の場合。無条件なら不要またはdummy
    transformer_config:
      target: taming.modules.transformer.mingpt.GPT
      params:
        vocab_size: 1024 # VQGANのn_embedと一致させる
        block_size: 256  # 16*16=256 (シーケンス長)
        n_layer: 24
        n_head: 16
        n_embd: 1024
    first_stage_config:
      target: taming.models.vqgan.VQModel
      params:
        ckpt_path: "logs/YOUR_VQGAN_LOG_DIR/checkpoints/last.ckpt" # Stage 1の重みパス
        # ... (Stage 1と同じパラメータ) ...
```

### 学習のポイント

  * [cite_start]**処理:** 画像をコードブックのインデックス $s \in \{0, ..., |\mathcal{Z}|-1\}^{h \times w}$ に変換し、自己回帰的に次を予測します ($p(s_i|s_{<i})$) [cite: 149, 153]。
  * [cite_start]**計算コスト:** シーケンス長に対して二次関数的に計算コストが増えるため、画像全体ではなくクロップしたパッチ（例: $16 \times 16$ latent codes）で学習します [cite: 168]。

### 実行コマンド

```bash
# Transformerの学習開始
python main.py --base configs/custom_transformer.yaml -t True --gpus 0,
```

-----

## 4\. 推論・画像生成 (Sampling)

学習済みモデルを使用して画像を生成します。

### サンプリング手法

  * [cite_start]**Top-k Sampling:** 確率分布の上位 $k$ 個からサンプリングすることで、多様性と品質のバランスを取ります（論文では $k=100$ 等を使用 [cite: 262]）。
  * [cite_start]**Sliding Window:** 学習時よりも大きな解像度（メガピクセル級）を生成する場合、Transformerをスライディングウィンドウ方式で適用します [cite: 169]。

### 実行コマンド

```bash
# サンプリングスクリプトの実行（例）
python scripts/make_samples.py \
  --base configs/custom_transformer.yaml \
  --resume logs/YOUR_TRANSFORMER_LOG_DIR/checkpoints/last.ckpt \
  --outdir outputs/samples \
  --top_k 100 \
  --temperature 1.0
```

-----

## 5\. 評価 (Evaluation)

生成された画像の品質を定量的に評価します。

### 指標

  * [cite_start]**FID (Fréchet Inception Distance):** 生成画像と実画像の分布の距離を測ります。値が小さいほど高品質です [cite: 239, 247]。
  * [cite_start]**再構成FID:** Stage 1 (VQGAN) の性能を確認するため、画像をエンコード→デコードした際の劣化を測定します [cite: 272]。

### 評価コマンド例

`pytorch-fid` 等の外部ライブラリを使用するのが一般的です。

```bash
# 必要なライブラリのインストール
pip install pytorch-fid

# FIDの計算 (実画像フォルダ と 生成画像フォルダ を比較)
python -m pytorch_fid data/my_dataset/validation outputs/samples
```

## 参考文献

  * [cite_start]**Paper:** Esser et al., "Taming Transformers for High-Resolution Image Synthesis", CVPR 2021. [cite: 5]
  * [cite_start]**Project Page:** [https://git.io/JLlvY](https://git.io/JLlvY) [cite: 19]