"""
評価・可視化スクリプト
学習済みモデルの詳細評価とAttention可視化を行う
"""

import os
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from data.latent_dataset import LatentFERDataset
from models.latent_vit import LatentViT


def load_model(checkpoint_path: str, device: str = "cuda"):
    """学習済みモデルをロード"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 設定を取得
    if 'config' in checkpoint:
        config = checkpoint['config']
        model_config = config['model']
    else:
        # デフォルト設定
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
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
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
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


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
        ax.bar(x + i * width, data[metric], width, label=metric)
    
    ax.set_xlabel('Emotion Classes')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_attention(model, sample_latent, device, save_path=None):
    """Attention重みを可視化"""
    model.eval()
    
    # 注意：実際のAttention重みを取得するには、モデルを改造する必要があります
    # ここでは簡易版として、CLSトークンと各潜在トークンの類似度を計算
    
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
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(similarities)), similarities.cpu().numpy())
        plt.xlabel('Latent Token Index')
        plt.ylabel('Cosine Similarity with CLS Token')
        plt.title('Attention-like Visualization: CLS Token Similarity to Latent Tokens')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return similarities.cpu().numpy()


def plot_training_curves(checkpoint_dir, save_path=None):
    """学習曲線を可視化"""
    # チェックポイントから学習履歴を復元
    train_losses = []
    val_accuracies = []
    val_f1_scores = []
    epochs = []
    
    for epoch in range(1, 100):  # 最大100エポックまで確認
        ckpt_path = os.path.join(checkpoint_dir, f"latent_vit_epoch{epoch}.pt")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            train_losses.append(checkpoint.get('train_loss', 0))
            val_metrics = checkpoint.get('val_metrics', {})
            val_accuracies.append(val_metrics.get('accuracy', 0))
            val_f1_scores.append(val_metrics.get('f1_macro', 0))
            epochs.append(epoch)
        else:
            break
    
    if not epochs:
        print("No training checkpoints found.")
        return
    
    # プロット
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 学習損失
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 検証精度
    ax2.plot(epochs, val_accuracies, 'g-', label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # F1スコア
    ax3.plot(epochs, val_f1_scores, 'r-', label='Val F1 Macro')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('Validation F1 Macro')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained LatentViT model")
    parser.add_argument("--checkpoint_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--latent_test_dir", required=True, help="Path to test latent files")
    parser.add_argument("--output_dir", default="eval_results", help="Output directory for results")
    parser.add_argument("--device", default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # デバイス設定
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # モデルロード
    print("Loading model...")
    model, model_config = load_model(args.checkpoint_path, device)
    print(f"Model loaded from {args.checkpoint_path}")
    
    # データローダー作成
    test_dataset = LatentFERDataset(args.latent_test_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # 評価実行
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device)
    
    # クラス名
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    # 基本メトリクス
    accuracy = np.mean(results['predictions'] == results['labels'])
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # 分類レポート
    print("\nClassification Report:")
    print(classification_report(results['labels'], results['predictions'], 
                              target_names=emotion_names))
    
    # 可視化
    print("\nGenerating visualizations...")
    
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
    
    # Attention可視化（サンプル）
    if len(test_dataset) > 0:
        sample_latent, _ = test_dataset[0]
        similarities = visualize_attention(
            model, sample_latent, device,
            save_path=os.path.join(args.output_dir, 'attention_visualization.png')
        )
        print(f"Attention similarities: {similarities}")
    
    # 学習曲線（チェックポイントディレクトリから）
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    plot_training_curves(
        checkpoint_dir,
        save_path=os.path.join(args.output_dir, 'training_curves.png')
    )
    
    # 結果をJSONで保存
    results_summary = {
        'accuracy': float(accuracy),
        'classification_report': classification_report(
            results['labels'], results['predictions'], 
            target_names=emotion_names, output_dict=True
        ),
        'model_config': model_config,
    }
    
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nEvaluation completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
