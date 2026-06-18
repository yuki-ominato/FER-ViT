import torch
import torch.nn as nn


class HighwayLayer(nn.Module):
    """
    Single Highway layer as used in the AFS paper (style_extraction.py).

    y = gate ⊙ nonlinear(x) + (1 − gate) ⊙ linear(x)

      gate      = sigmoid(W_gate · x)
      nonlinear = act( BN( W_nonlinear · x ) )   ← BatchNorm1d before activation
      linear    = W_linear · x                    ← learned carry (NOT identity)

    Note: the carry path uses a separate learned Linear, not the identity.
    This differs from the classic Highway Network (Srivastava et al., 2015)
    where the carry is the identity.
    """

    def __init__(self, dim: int, act: str = "lrelu", momentum: float = 0.1) -> None:
        super().__init__()
        self.nonlinear = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim, momentum=momentum),
        )
        self.linear = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

        if act == "relu":
            self.act = nn.ReLU()
        elif act == "lrelu":
            self.act = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise ValueError(f"Unknown activation '{act}': choose 'relu' or 'lrelu'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(x))
        n = self.act(self.nonlinear(x))
        l = self.linear(x)
        return g * n + (1.0 - g) * l


class StyleBlock(nn.Module):
    """
    Per-layer style extraction block.

    Linear(in_dim → mid_dim)
    HighwayLayer × num_highway
    Linear(mid_dim → in_dim)

    Matches StyleExtractionNet in AFS (fc1 → Highway(num_layers) → fc2).
    """

    def __init__(
        self,
        in_dim: int = 512,
        mid_dim: int = 256,
        num_highway: int = 2,
        act: str = "lrelu",
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.down = nn.Linear(in_dim, mid_dim)
        self.highways = nn.ModuleList(
            [HighwayLayer(mid_dim, act=act, momentum=momentum) for _ in range(num_highway)]
        )
        self.up = nn.Linear(mid_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        for hw in self.highways:
            x = hw(x)
        return self.up(x)


class StyleExtractor(nn.Module):
    """
    AFS Style Extractor  h : W+ → w_sty

    Each of the n_layers W+ codes is processed by an independent StyleBlock,
    one per StyleGAN2 resolution layer.

    Input : [B, n_layers, latent_dim]
    Output: [B, n_layers, latent_dim]   ← w_sty

    Identity component:
        w_id = w - h(w)

    Default hyperparameters match the AFS inference call:
        StyleExtractionNet(size=256, n_latent=18, num_layers=2, act="lrelu")
    """

    def __init__(
        self,
        n_layers: int = 18,
        latent_dim: int = 512,
        mid_dim: int = 256,
        num_highway: int = 2,
        act: str = "lrelu",
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.blocks = nn.ModuleList(
            [
                StyleBlock(latent_dim, mid_dim, num_highway, act, momentum)
                for _ in range(n_layers)
            ]
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # w : [B, n_layers, latent_dim]
        return torch.stack(
            [self.blocks[i](w[:, i, :]) for i in range(self.n_layers)],
            dim=1,
        )
