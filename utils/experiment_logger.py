"""
実験管理とロギング機能
TensorBoardロギングと実験設定管理を行う
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class ExperimentLogger:
    """実験管理とロギングを行うクラス"""
    
    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.experiment_dir = os.path.join(base_dir, experiment_name)
        
        # タイムスタンプ付きの実行ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.experiment_dir, f"{timestamp}")
        
        # ディレクトリ作成
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "logs"), exist_ok=True)
        
        # TensorBoardライター初期化
        self.writer = SummaryWriter(os.path.join(self.run_dir, "logs"))
        
        # 実験設定
        self.config = {}
        self.start_time = time.time()
        
    def log_config(self, config: Dict[str, Any]):
        """実験設定をログ"""
        self.config = config
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {config_path}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """メトリクスをログ"""
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
    
    def log_learning_curves(self, train_loss: float, val_metrics: Dict[str, float], epoch: int):
        """学習曲線をログ"""
        # 学習損失
        self.writer.add_scalar("Loss/Train", train_loss, epoch)
        
        # 検証メトリクス
        for key, value in val_metrics.items():
            if key in ['accuracy', 'f1_macro', 'f1_weighted']:
                self.writer.add_scalar(f"Validation/{key}", value, epoch)
    
    def log_model_architecture(self, model: torch.nn.Module, input_shape: tuple):
        """モデルアーキテクチャをログ"""
        # ダミー入力でモデルグラフを記録
        dummy_input = torch.randn(1, *input_shape)
        self.writer.add_graph(model, dummy_input)
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """ハイパーパラメータと最終メトリクスをログ"""
        self.writer.add_hparams(hparams, metrics)
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        class_names: list, epoch: int):
        """混同行列をログ"""
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # 正規化
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # プロットを生成
        fig = self._plot_confusion_matrix(cm_normalized, class_names)
        
        # Noneでない場合のみTensorBoardに記録
        if fig is not None:
            self.writer.add_figure(
                f"Confusion_Matrix/Epoch_{epoch}",
                fig,
                epoch
            )
        else:
            print(f"Skipping confusion matrix visualization for epoch {epoch}")

    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: list):
        """混同行列のプロット"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # GUIなし環境対応
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError as e:
            print(f"Warning: Required visualization library not available ({e}), skipping confusion matrix plot")
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error creating confusion matrix plot: {e}")
            return None
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """チェックポイントを保存（ベストと最新のみを残す）"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'run_id': self.run_dir,
        }
        
        # 1. 最新モデル (last_model.pt)
        # 毎回上書き保存することで、学習終了時には「最終エポック」のモデルのみが残ります
        last_path = os.path.join(self.run_dir, "checkpoints", "last_model.pt")
        
        # PyTorch 2.6対応: 古い形式との互換性維持
        torch.save(checkpoint, last_path, _use_new_zipfile_serialization=False)
        
        # 2. ベストモデル (best_model.pt)
        # 精度更新時のみ保存
        if is_best:
            best_path = os.path.join(self.run_dir, "checkpoints", "best_model.pt")
            torch.save(checkpoint, best_path, _use_new_zipfile_serialization=False)
            print(f"Best model saved at epoch {epoch}")

    
    def log_attention_weights(self, attention_weights: np.ndarray, epoch: int, 
                            sample_idx: int = 0):
        """Attention重みをログ"""
        # ヒートマップとして可視化
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(attention_weights, cmap='viridis', aspect='auto')
        ax.set_title(f'Attention Weights - Sample {sample_idx}')
        ax.set_xlabel('Latent Token Index')
        ax.set_ylabel('Attention Head')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        self.writer.add_figure(f"Attention/Sample_{sample_idx}", fig, epoch)
        plt.close(fig)
    
    def log_gradients(self, model: torch.nn.Module, epoch: int):
        """勾配の分布をログ"""
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f"Gradients/{name}", param.grad, epoch)
                self.writer.add_scalar(f"Gradient_Norm/{name}", 
                                     param.grad.norm().item(), epoch)
    
    def log_learning_rate(self, optimizer: torch.optim.Optimizer, epoch: int):
        """学習率をログ"""
        for i, param_group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f"Learning_Rate/Group_{i}", 
                                 param_group['lr'], epoch)
    
    def log_parameters(self, model: torch.nn.Module, epoch: int):
        """パラメータの分布をログ"""
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"Parameters/{name}", param, epoch)
    
    def log_images(self, images: torch.Tensor, labels: torch.Tensor, 
                  predictions: torch.Tensor, epoch: int, max_images: int = 8):
        """画像と予測結果をログ"""
        # 注意：潜在ベクトルなので、実際の画像可視化は困難
        # 代わりに潜在ベクトルの統計情報をログ
        self.writer.add_histogram("Latent_Statistics/Mean", 
                                images.mean(dim=(1, 2)), epoch)
        self.writer.add_histogram("Latent_Statistics/Std", 
                                images.std(dim=(1, 2)), epoch)
    
    def log_experiment_summary(self, final_metrics: Dict[str, float]):
        """実験サマリーをログ"""
        end_time = time.time()
        duration = end_time - self.start_time
        
        summary = {
            'experiment_name': self.experiment_name,
            'run_id': self.run_dir,
            'duration_seconds': duration,
            'final_metrics': final_metrics,
            'config': self.config,
        }
        
        summary_path = os.path.join(self.run_dir, "experiment_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Experiment summary saved to {summary_path}")
        print(f"Total duration: {duration:.2f} seconds")
    
    def close(self):
        """ロガーを閉じる"""
        self.writer.close()
    
    def get_experiment_path(self) -> str:
        """実験ディレクトリのパスを取得"""
        return self.run_dir


def create_experiment_name(model_config: Dict[str, Any], 
                          training_config: Dict[str, Any],
                          is_latent: bool = True,
                          is_pretrained: bool = False) -> str:
                          
    """実験名を自動生成"""
    # モデル設定からキー情報を抽出
    # depth, heads, dropoutを使用
    if is_latent:
        model_name = f"latent_vit_d{model_config.get('depth', 6)}_h{model_config.get('heads', 8)}_do{model_config.get('dropout', 0.1)}"
    else:
        model_name = f"image_vit_d{model_config.get('depth', 6)}_h{model_config.get('heads', 8)}_do{model_config.get('dropout', 0.1)}"
    
    # 学習設定からキー情報を抽出
    lr = training_config.get('lr', 1e-4)
    batch_size = training_config.get('batch_size', 64)
    epochs = training_config.get('epochs', 60)
    mixup = training_config.get('mixup', 1.0)
    
    if is_latent:
        training_name = f"lr{lr}_bs{batch_size}_ep{epochs}_Mixup{mixup}"
    elif is_pretrained:
        training_name = f"lr{lr}_bs{batch_size}_ep{epochs}_pretrained"
    else:
        training_name = f"lr{lr}_bs{batch_size}_ep{epochs}"

    
    # エンコーダ情報（設定に含まれている場合）
    encoder_info = ""
    if 'encoder_type' in training_config:
        encoder_info = f"_{training_config['encoder_type']}"
    
    return f"{model_name}_{training_name}{encoder_info}"


def load_experiment_config(experiment_path: str) -> Dict[str, Any]:
    """実験設定をロード"""
    config_path = os.path.join(experiment_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")


def compare_experiments(experiment_dirs: list, metric: str = "f1_macro") -> Dict[str, float]:
    """複数の実験結果を比較"""
    results = {}
    
    for exp_dir in experiment_dirs:
        summary_path = os.path.join(exp_dir, "experiment_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                exp_name = summary.get('experiment_name', os.path.basename(exp_dir))
                final_metrics = summary.get('final_metrics', {})
                results[exp_name] = final_metrics.get(metric, 0.0)
    
    return results
