# FER-ViT: Facial Expression Recognition via StyleGAN Latent Space

**StyleGANの潜在空間を活用した、高効率・データ節約型の感情認識Vision Transformer**

## 📖 概要
従来のVision Transformer (ViT) は、学習に膨大なデータと計算リソースを必要とします。本研究では、この課題を解決するために **「LatentViT」** アーキテクチャを提案します。

画像を生のピクセルとして扱うのではなく、**StyleGAN2の潜在空間（Latent Space $w+$）** にマッピングしてからTransformerに入力することで、入力情報を「意味的に圧縮」し、モデルの学習効率とデータ効率を向上させることを目的としています。

### アーキテクチャの比較
* **従来手法 (Standard ViT):** 画像 $\rightarrow$ パッチ分割 $\rightarrow$ Transformer $\rightarrow$ 分類
* **提案手法 (LatentViT):** 画像 $\rightarrow$ **pSp Encoder ($w+$空間)** $\rightarrow$ **Hybrid Transformer** $\rightarrow$ 分類

## 🚀 特徴
* **Semantic Compression:** StyleGANのエンコーダを用いることで、表情の「意味」を保持したまま情報を圧縮。
* **Hybrid Architecture:** ImageNetで事前学習済みのViT（`timm`ライブラリ）を、潜在空間入力用に適応させる独自構造を実装。
* **Parameter Efficiency:** Adapter層を導入し、Transformerの重みを凍結したまま少数のパラメータのみで学習が可能。

## 🛠️ 環境構築

```bash
# conda環境の作成
conda env create -f environment.yml
conda activate fer-vit

# 依存ライブラリのインストール
pip install -r requirements.txt