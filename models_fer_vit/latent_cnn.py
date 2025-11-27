"""
潜在空間ベクトル用CNNモデル
入力: StyleGAN w+ latent codes (B, 18, 512)
出力: 感情分類 (B, 7)

ViTと同じ入力形式で、CNNベースの処理を行う
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentConv1D(nn.Module):
    """
    1D畳み込みブロック
    系列データ（潜在コード）に対する畳み込み
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=1, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             stride, padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x):
        """
        Args:
            x: (B, C, L) - バッチ、チャンネル、系列長
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class LatentResBlock1D(nn.Module):
    """
    1D残差ブロック
    系列の局所パターンを学習
    """
    def __init__(self, channels, kernel_size=3, dropout=0.2):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out, inplace=True)
        return out


class LatentCNN(nn.Module):
    """
    潜在空間用CNN（1D畳み込み）
    
    アーキテクチャ:
    (B, 18, 512) → Reshape → (B, 512, 18)
    → Conv1D blocks → Global pooling
    → FC → (B, 7)
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        seq_len: int = 18,
        num_classes: int = 7,
        hidden_dims: list = [512, 512, 512, 512],
        dropout: float = 0.3,
        use_residual: bool = True,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.use_residual = use_residual
        
        # 1D畳み込み層（チャンネル方向の特徴抽出）
        layers = []
        in_ch = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(LatentConv1D(in_ch, hidden_dim, kernel_size=3, dropout=dropout))
            in_ch = hidden_dim
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 残差ブロック（オプション）
        if use_residual:
            self.res_blocks = nn.ModuleList([
                LatentResBlock1D(hidden_dims[-1], kernel_size=3, dropout=dropout)
                for _ in range(2)
            ])
        
        # グローバルプーリング（決定論的）
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        # MaxPoolの代わりに平均プーリングのみ使用（決定論的）
        self.use_max_pool = False  # 決定論的実行のため無効化
        
        # 分類ヘッド（入力次元を調整）
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dims[-1], 512),  # avg poolingのみ
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """重みの初期化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (B, seq_len, latent_dim) = (B, 18, 512)
        Returns:
            (B, num_classes) ロジット
        """
        # (B, L, D) → (B, D, L) - Conv1Dの入力形式
        x = x.transpose(1, 2)  # (B, 512, 18)
        
        # 畳み込み処理
        x = self.conv_layers(x)
        
        # 残差ブロック
        if self.use_residual:
            for res_block in self.res_blocks:
                x = res_block(x)
        
        # グローバルプーリング（avgのみ - 決定論的）
        avg_pool = self.global_avg_pool(x)  # (B, C, 1)
        
        # 分類
        x = self.classifier(avg_pool)
        return x


class LatentCNNDeep(nn.Module):
    """
    より深い潜在空間CNN
    階層的な特徴抽出
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        seq_len: int = 18,
        num_classes: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        
        # 初期投影（次元削減）
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
        )
        
        # 1D畳み込みブロック（階層的）
        self.conv_block1 = nn.Sequential(
            LatentConv1D(256, 256, kernel_size=3, dropout=dropout),
            LatentResBlock1D(256, kernel_size=3, dropout=dropout),
        )
        
        self.conv_block2 = nn.Sequential(
            LatentConv1D(256, 384, kernel_size=3, dropout=dropout),
            LatentResBlock1D(384, kernel_size=3, dropout=dropout),
        )
        
        self.conv_block3 = nn.Sequential(
            LatentConv1D(384, 512, kernel_size=3, dropout=dropout),
            LatentResBlock1D(512, kernel_size=3, dropout=dropout),
            LatentResBlock1D(512, kernel_size=3, dropout=dropout),
        )
        
        # アテンションプーリング（重要な位置に注目）
        self.attention = nn.Sequential(
            nn.Conv1d(512, 1, kernel_size=1),
            nn.Softmax(dim=2),
        )
        
        # 分類ヘッド
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (B, seq_len, latent_dim) = (B, 18, 512)
        """
        B = x.size(0)
        
        # 次元削減
        x = self.input_proj(x)  # (B, 18, 256)
        
        # Conv1D形式に変換
        x = x.transpose(1, 2)  # (B, 256, 18)
        
        # 階層的畳み込み
        x = self.conv_block1(x)  # (B, 256, 18)
        x = self.conv_block2(x)  # (B, 384, 18)
        x = self.conv_block3(x)  # (B, 512, 18)
        
        # アテンションプーリング
        attention_weights = self.attention(x)  # (B, 1, 18)
        x = torch.sum(x * attention_weights, dim=2)  # (B, 512)
        
        # 分類
        x = self.classifier(x)
        return x


class LatentCNNLight(nn.Module):
    """
    軽量版潜在空間CNN
    高速実験用
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        seq_len: int = 18,
        num_classes: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # シンプルな構造
        self.encoder = nn.Sequential(
            # (B, L, D) → (B, D, L)
            nn.Conv1d(latent_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Conv1d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, D) = (B, 18, 512)
        """
        x = x.transpose(1, 2)  # (B, 512, 18)
        x = self.encoder(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


class LatentCNN2D(nn.Module):
    """
    2D畳み込みアプローチ
    潜在ベクトルを2D画像として扱う
    
    (18, 512) → Reshape → (1, 18, 512) → Conv2D
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        seq_len: int = 18,
        num_classes: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # 2D畳み込み（画像として処理）
        self.features = nn.Sequential(
            # (B, 1, 18, 512)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 0.5),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),  # (B, 128, 9, 256)
            nn.Dropout2d(dropout * 0.5),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),  # (B, 256, 4, 128)
            nn.Dropout2d(dropout),
        )
        
        # Global pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 分類器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (B, 18, 512)
        """
        # 2D画像として扱う
        x = x.unsqueeze(1)  # (B, 1, 18, 512)
        
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


def create_latent_cnn(
    model_type: str = 'standard',
    latent_dim: int = 512,
    seq_len: int = 18,
    num_classes: int = 7,
    dropout: float = 0.3,
):
    """
    潜在空間CNNのファクトリー関数
    
    Args:
        model_type: 'light', 'standard', 'deep', '2d'
        latent_dim: 潜在コードの次元
        seq_len: 系列長（w+の層数）
        num_classes: 出力クラス数
        dropout: ドロップアウト率
    """
    if model_type == 'light':
        return LatentCNNLight(latent_dim, seq_len, num_classes, dropout)
    elif model_type == 'standard':
        return LatentCNN(latent_dim, seq_len, num_classes, dropout=dropout, use_residual=True)
    elif model_type == 'deep':
        return LatentCNNDeep(latent_dim, seq_len, num_classes, dropout)
    elif model_type == '2d':
        return LatentCNN2D(latent_dim, seq_len, num_classes, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    print("="*60)
    print("Latent CNN Models Test")
    print("="*60)
    
    # テスト入力（ViTと同じ形式）
    batch_size = 4
    seq_len = 18
    latent_dim = 512
    x = torch.randn(batch_size, seq_len, latent_dim)
    
    models = {
        'Light': create_latent_cnn('light'),
        'Standard (1D)': create_latent_cnn('standard'),
        'Deep': create_latent_cnn('deep'),
        '2D Conv': create_latent_cnn('2d'),
    }
    
    print(f"\nInput shape: {x.shape}")
    print("="*60)
    
    for name, model in models.items():
        output = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        
        print(f"\n{name}:")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {n_params:,}")
        print(f"  Size (MB): {n_params * 4 / 1024 / 1024:.2f}")
        
        # 勾配フローの確認
        loss = output.sum()
        loss.backward()
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        print(f"  Gradient norm: {grad_norm:.2f}")