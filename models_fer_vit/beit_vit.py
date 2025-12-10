import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

class TokenViT(nn.Module):
    def __init__(self, 
                 vocab_size=16384,   # VQ-GANのコードブックサイズ
                 seq_len=256,        # 16x16パッチなど
                 embed_dim=512,
                 depth=6,
                 heads=8,
                 num_classes=7,
                 dropout=0.1,
                 use_mim=False):     # MIM学習用モードかどうか
        super().__init__()
        
        self.num_classes = num_classes
        self.use_mim = use_mim
        
        # 1. Token Embedding (Discrete -> Continuous)
        self.token_emb = nn.Embedding(vocab_size + 1, embed_dim) # +1 は [MASK] トークン用
        self.mask_token_id = vocab_size 
        
        # 2. Position Embedding
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 3. Transformer Encoder (timmのBlockを再利用するか、自作)
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=heads, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # 4. Heads
        if use_mim:
            # MIM用ヘッド: マスクされた部分の元のトークンIDを予測
            self.mim_head = nn.Linear(embed_dim, vocab_size)
        else:
            # 分類用ヘッド
            self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, bool_masked_pos=None):
        """
        x: (B, Seq_Len) - 離散トークンID
        bool_masked_pos: (B, Seq_Len) - マスク位置を示すブール値 (MIM用)
        """
        B, N = x.shape
        
        # MIMの場合、入力トークンをMASKトークンに置き換え
        if self.use_mim and bool_masked_pos is not None:
            x = x.clone()
            x[bool_masked_pos] = self.mask_token_id
            
        # Embedding
        x = self.token_emb(x) # (B, N, D)
        
        # CLS token append
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, N+1, D)
        
        # Add Pos Emb
        x = x + self.pos_emb
        
        # Transformer
        x = self.blocks(x)
        x = self.norm(x)
        
        if self.use_mim:
            # CLSを除いて、元のシーケンス部分のみ出力
            return self.mim_head(x[:, 1:]) 
        else:
            # 分類: CLSトークンのみ使用
            return self.head(x[:, 0])