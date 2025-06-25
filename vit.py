import torch
import torch.nn as nn
import torch.nn.functional as F

class VitInputLayer(nn.Module):
    def __init__(self,
    in_channels:int=3,
    emb_dim:int=3,
    num_patch_row:int=2,
    image_size:int=32
    ):

        super(VitInputLayer, self).__init__()
        self.in_channels=in_channels
        self.emb_dim=emb_dim
        self.num_patch_row=num_patch_row
        self.image_size=image_size

        # num of patch
        self.num_patch = self.num_patch_row**2

        # scale of patch
        self.patch_size = int(self.image_size // self.num_patch_row)

        # 入力画像のパッチへの分割 & パッチの埋め込みを行う層
        self.patch_emb_layer = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # クラストークン
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, emb_dim)
        )

        # 位置埋め込み
        self.pos_emb = nn.Parameter(
            torch.randn(1, self.num_patch+1, emb_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        引数:
            x: 入力画像。形状は、(B, C, H, W)
                B: バッチサイズ、C: チャンネル数、H: 高さ、W:幅
        
        返り値:
            z_0: Vitへの入力。形状は、(B, N, D)
                B: バッチサイズ、N:トークン数、D:埋め込みベクトルの長さ
        """
        # 入力画像xをパッチに埋め込む
        z_0 = self.patch_emb_layer(x)   # (B,C,H,W) -> (B,D,H/P,W/P)
        z_0 = z_0.flatten(2)            # (B,D,H/P,W/P) -> (B,D,Np)
        z_0 = z_0.transpose(1,2)        # (B,D,Np) -> (B,Np,D)
        
        # クラストークン結合 (Np + 1 = N)
        # (B,Np,D) -> (B,N,D)
        # cls_tokenの形状は(1,1,D)であるため、repeat関数によって(B,1,D)に変換
        z_0 = torch.cat([self.cls_token.repeat(repeats=(x.size(0),1,1)), z_0], dim=1)
        
        # 位置埋め込み
        z_0 += self.pos_emb

        return z_0

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
    emb_dim:int=384,
    head:int=3,
    dropout:float=0.
    ):

        super(MultiHeadSelfAttention, self).__init__()

class VitEncoderBlock(nn.Module):
    def __init__(
        self,
        emb_dim:int=384,
        head:int=8,
        hidden_dim:int=384*4,
        dropout:float=0.
    ):

        super(VitEncoderBlock, self).__init__()
        self.ln1 = nn.LayerNorm(emb_dim)

        # Multi Head Self-Attention
        self.msa = MultiHeadSelfAttention()

class Vit(nn.Module):
    def __init__(self,
    in_channels:int=3,
    num_classes:int=10,
    emb_dim:int=384,
    num_patch_row:int=2,
    image_size:int=32,
    num_blocks:int=7,
    head:int=8,
    hidden_dim:int=384*4,
    dropout:float=0
    ):
        
        super(Vit, self).__init__()

        # Input Layer
        self.input_layer = VitInputLayer(
            in_channels,
            emb_dim,
            num_patch_row,
            image_size
        )

        # Encoder
        self.encoder = nn.Sequential(*[
            VitEncoderBlock(
                emb_dim=emb_dim,
                head=head,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            for _ in range(num_blocks)])

batch_size, channel, height, width = 2, 3, 32, 32
x = torch.randn(batch_size, channel, height, width)
input_layer = VitInputLayer(num_patch_row=2)
z_0 = input_layer(x)
print(x.shape)
print(z_0.shape)