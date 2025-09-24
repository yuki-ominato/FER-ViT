import torch
import torch.nn as nn


class LatentViT(nn.Module):
    def __init__(
        self,
        latent_dim: int = 512,
        seq_len: int = 18,
        embed_dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 2048,
        num_classes: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len

        self.input_proj = nn.Linear(latent_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, latent_dim)
        x = self.input_proj(x)
        b = x.size(0)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_emb
        x = self.transformer(x)
        cls_out = x[:, 0]
        out = self.mlp_head(cls_out)
        return out


