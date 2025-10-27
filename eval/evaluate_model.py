"""
評価・可視化スクリプト（改善版）
学習済みモデルの詳細評価とAttention可視化を行う
"""

import os
import sys
import argparse
import json
import matplotlib
matplotlib.use('Agg')  # GUIなし環境対応
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.latent_dataset import LatentFERDataset
from models_fer_vit.latent_vit import LatentViT


def load_model(checkpoint_path: str, device: str = "cuda"):
    """学習済みモデルをロード"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # PyTorch 2.6以降の互換性対応
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        # PyTorch 2.5以前の場合
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 設定を取得（複数のキー名に対応）
    if 'config' in checkpoint:
        config = checkpoint['config']
        model_config = config.get('model', config)
    elif 'args' in checkpoint:
        # train_latent_vit_simple.pyからのチェックポイント
        args = checkpoint['args']
        model_config = {
            'latent_dim': args.latent_dim,
            'seq_len': args.seq_len,
            'embed_dim': args.embed_dim,
            'depth': args.depth,
            'heads': args.heads,
            'mlp_dim': args.mlp_dim,
            'num_classes': args.num_classes,
            'dropout': args.dropout,
        }
    else:
        # デフォルト設定
        print("Warning: Config not found in checkpoint, using default values")
        model_config = {
            'latent_dim': 512,
            'seq_len': 18,
            'embed_dim': 512,
            'depth': 6,
            'heads': 8,
            'mlp_dim': 2048,
            'num_classes': 7,
            'dropout': 0.1,
        }
    
    # モデル初期化
    model = LatentViT(**model_config).to(device)
    
    # state_dictのキー名に対応
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        raise KeyError("Model state dict not found in checkpoint")
    
    model.eval()
    
    # エポック情報を取得
    epoch_info = checkpoint.get('epoch', 'unknown')
    print(f"Loaded model from epoch {epoch_info}")
    if 'metrics' in checkpoint:
        print(f"Checkpoint metrics: {checkpoint['metrics']}")
    
    return model, model_config


def evaluate_model(model, data_loader, device):
    """モデルの詳細評価"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for latents, labels in data_loader:
            latents = latents.to(device)
            labels = labels.to(device)
            
            logits = model(latents)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """混同行列を可視化"""
    cm = confusion_matrix(y_true, y_pred)
    
    # 正規化版と生データ版の両方を表示
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 正規化版
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Normalized)')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # 生データ版
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Counts)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def plot_class_metrics(y_true, y_pred, class_names, save_path=None):
    """クラス別メトリクスを可視化"""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # 各クラスのprecision, recall, f1-scoreを抽出
    metrics = ['precision', 'recall', 'f1-score']
    data = {metric: [] for metric in metrics}
    classes = []
    
    for class_name in class_names:
        if class_name in report:
            classes.append(class_name)
            for metric in metrics:
                data[metric].append(report[class_name][metric])
    
    # プロット
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, data[metric], width, label=metric.capitalize())
    
    ax.set_xlabel('Emotion Classes')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class metrics saved to: {save_path}")
    plt.close()


def visualize_attention(model, sample_latent, device, save_path=None):
    """Attention重みを可視化（簡易版）"""
    model.eval()
    
    with torch.no_grad():
        sample_latent = sample_latent.unsqueeze(0).to(device)  # (1, L, D)
        
        # モデルの内部処理を再現
        x = model.input_proj(sample_latent)  # (1, L, embed_dim)
        b = x.size(0)
        cls = model.cls_token.expand(b, -1, -1)  # (1, 1, embed_dim)
        x_with_cls = torch.cat([cls, x], dim=1)  # (1, L+1, embed_dim)
        x_with_pos = x_with_cls + model.pos_emb
        
        # Transformer処理
        x_out = model.transformer(x_with_pos)  # (1, L+1, embed_dim)
        
        # CLSトークンと各潜在トークンの類似度を計算
        cls_token = x_out[:, 0]  # (1, embed_dim)
        latent_tokens = x_out[:, 1:]  # (1, L, embed_dim)
        
        # コサイン類似度
        similarities = torch.cosine_similarity(
            cls_token.unsqueeze(1), latent_tokens, dim=2
        ).squeeze(0)  # (L,)
        
        # 可視化
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 棒グラフ
        ax1.bar(range(len(similarities)), similarities.cpu().numpy())
        ax1.set_xlabel('Latent Token Index (StyleGAN Layer)')
        ax1.set_ylabel('Cosine Similarity with CLS Token')
        ax1.set_title('Attention-like Visualization: CLS Token Similarity')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # ヒートマップ（トークン間の類似度）
        token_similarities = torch.cosine_similarity(
            latent_tokens.unsqueeze(2), latent_tokens.unsqueeze(1), dim=3
        ).squeeze(0).cpu().numpy()
        
        sns.heatmap(token_similarities, cmap='coolwarm', center=0, ax=ax2,
                   xticklabels=range(len(similarities)), 
                   yticklabels=range(len(similarities)))
        ax2.set_title('Token-to-Token Similarity Matrix')
        ax2.set_xlabel('Token Index')
        ax2.set_ylabel('Token Index')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention visualization saved to: {save_path}")
        plt.close()
        
        return similarities.cpu().numpy()


def plot_prediction_confidence(probabilities, labels, predictions, save_path=None):
    """予測信頼度の分布を可視化"""
    # 正解/不正解ごとの信頼度
    correct_mask = labels == predictions
    correct_conf = np.max(probabilities[correct_mask], axis=1)
    incorrect_conf = np.max(probabilities[~correct_mask], axis=1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ヒストグラム
    ax1.hist(correct_conf, bins=20, alpha=0.7, label='Correct', color='green')
    ax1.hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', color='red')
    ax1.set_xlabel('Prediction Confidence')
    ax1.set_ylabel('Count')
    ax1.set_title('Prediction Confidence Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ボックスプロット
    ax2.boxplot([correct_conf, incorrect_conf], labels=['Correct', 'Incorrect'])
    ax2.set_ylabel('Prediction Confidence')
    ax2.set_title('Confidence Comparison')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confidence plot saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained LatentViT model")
    parser.add_argument("--checkpoint_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--latent_test_dir", required=True, help="Path to test latent files")
    parser.add_argument("--output_dir", default="eval_results", help="Output directory for results")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--visualize_samples", type=int, default=5, 
                       help="Number of samples to visualize attention for")
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # デバイス設定
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # モデルロード
    print("\n" + "="*60)
    print("Loading model...")
    print("="*60)
    model, model_config = load_model(args.checkpoint_path, device)
    print(f"Model configuration: {json.dumps(model_config, indent=2)}")
    
    # データローダー作成
    print("\n" + "="*60)
    print("Loading test dataset...")
    print("="*60)
    test_dataset = LatentFERDataset(args.latent_test_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4)
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # 評価実行
    print("\n" + "="*60)
    print("Evaluating model...")
    print("="*60)
    results = evaluate_model(model, test_loader, device)
    
    # クラス名
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    # 基本メトリクス
    accuracy = np.mean(results['predictions'] == results['labels'])
    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 分類レポート
    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    print(classification_report(results['labels'], results['predictions'], 
                              target_names=emotion_names, digits=4))
    
    # 可視化
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)
    
    # 混同行列
    plot_confusion_matrix(
        results['labels'], results['predictions'], emotion_names,
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # クラス別メトリクス
    plot_class_metrics(
        results['labels'], results['predictions'], emotion_names,
        save_path=os.path.join(args.output_dir, 'class_metrics.png')
    )
    
    # 予測信頼度
    plot_prediction_confidence(
        results['probabilities'], results['labels'], results['predictions'],
        save_path=os.path.join(args.output_dir, 'prediction_confidence.png')
    )
    
    # Attention可視化（サンプル）
    if len(test_dataset) > 0:
        print(f"\nVisualizing attention for {args.visualize_samples} samples...")
        for i in range(min(args.visualize_samples, len(test_dataset))):
            sample_latent, sample_label = test_dataset[i]
            similarities = visualize_attention(
                model, sample_latent, device,
                save_path=os.path.join(args.output_dir, f'attention_sample_{i}.png')
            )
    
    # 結果をJSONで保存
    results_summary = {
        'accuracy': float(accuracy),
        'classification_report': classification_report(
            results['labels'], results['predictions'], 
            target_names=emotion_names, output_dict=True
        ),
        'model_config': model_config,
        'checkpoint_path': args.checkpoint_path,
        'test_dataset_size': len(test_dataset),
    }
    
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Evaluation completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Summary: {results_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()