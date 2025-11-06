"""
画像ViTモデルの評価スクリプト
"""

import os
import sys
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.image_dataset import ImageFERDataset, get_val_transforms
from models_fer_vit.image_vit import ImageViT


def load_model(checkpoint_path: str, device: str = "cuda"):
    """学習済みモデルをロード"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 設定を取得
    if 'config' in checkpoint:
        config = checkpoint['config']
        model_config = config.get('model', config)
    else:
        print("Warning: Config not found, using default values")
        model_config = {
            'img_size': 224,
            'patch_size': 16,
            'embed_dim': 384,
            'depth': 12,
            'heads': 6,
            'mlp_dim': 1536,
            'num_classes': 7,
            'dropout': 0.1,
        }
    
    # モデル初期化
    model = ImageViT(
        img_size=model_config.get('img_size', 224),
        patch_size=model_config.get('patch_size', 16),
        in_channels=3,
        embed_dim=model_config.get('embed_dim', 384),
        depth=model_config.get('depth', 12),
        heads=model_config.get('heads', 6),
        mlp_dim=model_config.get('mlp_dim', 1536),
        num_classes=model_config.get('num_classes', 7),
        dropout=model_config.get('dropout', 0.1),
    ).to(device)
    
    # state_dictロード
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        raise KeyError("Model state dict not found in checkpoint")
    
    model.eval()
    
    epoch_info = checkpoint.get('epoch', 'unknown')
    print(f"Loaded model from epoch {epoch_info}")
    if 'metrics' in checkpoint:
        print(f"Checkpoint metrics: {checkpoint['metrics']}")
    
    return model, model_config


def evaluate_model(model, data_loader, device):
    """モデルの評価"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
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
    """混同行列の可視化"""
    cm = confusion_matrix(y_true, y_pred)
    
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
    """クラス別メトリクスの可視化"""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    metrics = ['precision', 'recall', 'f1-score']
    data = {metric: [] for metric in metrics}
    classes = []
    
    for class_name in class_names:
        if class_name in report:
            classes.append(class_name)
            for metric in metrics:
                data[metric].append(report[class_name][metric])
    
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


def plot_prediction_confidence(probabilities, labels, predictions, save_path=None):
    """予測信頼度の分布"""
    correct_mask = labels == predictions
    correct_conf = np.max(probabilities[correct_mask], axis=1)
    incorrect_conf = np.max(probabilities[~correct_mask], axis=1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(correct_conf, bins=20, alpha=0.7, label='Correct', color='green')
    ax1.hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', color='red')
    ax1.set_xlabel('Prediction Confidence')
    ax1.set_ylabel('Count')
    ax1.set_title('Prediction Confidence Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
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
    parser = argparse.ArgumentParser(description="Evaluate trained ImageViT model")
    parser.add_argument("--checkpoint_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--test_dir", required=True, help="Path to test images")
    parser.add_argument("--output_dir", default="eval_results", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # モデルロード
    print("\n" + "="*60)
    print("Loading model...")
    print("="*60)
    model, model_config = load_model(args.checkpoint_path, device)
    print(f"Model configuration: {json.dumps(model_config, indent=2)}")
    
    # データセット
    print("\n" + "="*60)
    print("Loading test dataset...")
    print("="*60)
    test_dataset = ImageFERDataset(
        args.test_dir,
        transform=get_val_transforms(args.img_size)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # 評価
    print("\n" + "="*60)
    print("Evaluating model...")
    print("="*60)
    results = evaluate_model(model, test_loader, device)
    
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    accuracy = np.mean(results['predictions'] == results['labels'])
    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    print(classification_report(results['labels'], results['predictions'],
                              target_names=emotion_names, digits=4))
    
    # 可視化
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)
    
    plot_confusion_matrix(
        results['labels'], results['predictions'], emotion_names,
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    plot_class_metrics(
        results['labels'], results['predictions'], emotion_names,
        save_path=os.path.join(args.output_dir, 'class_metrics.png')
    )
    
    plot_prediction_confidence(
        results['probabilities'], results['labels'], results['predictions'],
        save_path=os.path.join(args.output_dir, 'prediction_confidence.png')
    )
    
    # 結果保存
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