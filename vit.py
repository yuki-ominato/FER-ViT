import torch
import torch.nn as nn
import torch.nn.functional as F

class VitInputLayer(nn.Module):
    def __init__(self,
    in_channels:int=3,
    emb_dim:int=384,
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
        self.head = head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // head
        self.sqrt_dh = self.head_dim**0.5

        # 入力をq,k,vに埋め込むための線形層
        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        self.w_o = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, num_patch, _ = z.size()

        # 埋め込み
        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)

        # q, k, vをヘッドに分ける
        # (B, N, D -> (B, N, h, D//h)
        q = q.view(batch_size, num_patch, self.head, self.head_dim)
        k = k.view(batch_size, num_patch, self.head, self.head_dim)
        v = v.view(batch_size, num_patch, self.head, self.head_dim)

        # Self-Attetion用に形状変更
        # (B, N, h, D//h) -> (B, h, N, D\\h)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)


        # 内積
        # (B, h, N, D//h) x (B, h, D//h, N) -> (B, h, N, N)
        k_T = k.transpose(2, 3) # 転置
        dots = (q @ k_T) / self.sqrt_dh # パッチ同士の類似度
        attn = F.softmax(dots, dim=-1)
        attn = self.attn_drop(attn)

        # 加重和
        # (B, h, N, N) x (B, h, N, D//h) -> (B, h, N, D//h)
        out = attn @ v
        # (B, h, N, D//h) -> (B, N, h, D//h)
        out = out.transpose(1, 2)
        # (B, N, h, D//h) -> (B, N, D)
        out = out.reshape(batch_size, num_patch, self.emb_dim)

        # 出力層
        out = self.w_o(out)
        return out

class VitEncoderBlock(nn.Module):
    def __init__(
        self,
        emb_dim:int=384,
        head:int=8,
        hidden_dim:int=384*4,   # 原論文に従い、emb_dimの4倍
        dropout:float=0.
    ):

        super(VitEncoderBlock, self).__init__()
        # 1つ目のLayer Normalization
        self.ln1 = nn.LayerNorm(emb_dim)

        # Multi Head Self-Attention
        self.msa = MultiHeadSelfAttention(
            emb_dim=emb_dim,
            head=head,
            dropout = dropout,
        )

        # 2つ目のLayer Normalization
        self.ln2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, z:torch.Tensor) -> torch.Tensor:
        """
        引数：
            z: Encoder Blockへの入力。形状は(B, N, D)

        返り値
            out: Encoder Blockへの出力。形状は(B, N, D)
        """
        
        # Encoder Block
        out = self.msa(self.ln1(z)) + z      # 前半
        out = self.mlp(self.ln2(z)) + out   # 後半
        return out

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
        # 内包表記でnum_block個のEncoderを定義
        self.encoder = nn.Sequential(*[
            VitEncoderBlock(
                emb_dim=emb_dim,
                head=head,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            for _ in range(num_blocks)])

        # MLP Head
        # クラス分類(確率)を出力するレイヤー
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        引数
            x: ViTへの入力画像。形状は(B, C, H, W)
        返り値
            out: ViTへの出力。形状は(B, M)
                M: クラス数
        """

        out = self.input_layer(x)   # Input Layer (B, C, H, W) -> (B, N, D)
        out = self.encoder(out)     # Encoder     (B, N, D) -> (B, N, D)
        cls_token = out[:, 0]       # (B, N, D) -> (B, D)
        pred = self.mlp_head(cls_token)
        return pred            


# VitInputLayer 動作確認
batch_size, channel, height, width = 2, 3, 32, 32
x = torch.randn(batch_size, channel, height, width)
input_layer = VitInputLayer(num_patch_row=2)
z_0 = input_layer(x)
print(x.shape)
print(z_0.shape)

# MultiHeadSelf-Attention 動作確認
mhsa = MultiHeadSelfAttention()
out = mhsa(z_0)
print(out.shape)

# Encoder 動作確認
vit_enc = VitEncoderBlock()
z_1 = vit_enc(z_0)
print(z_1.shape)

# Vision Transfomer 動作確認
num_classes = 10
vit = Vit(in_channels=channel, num_classes=num_classes)
pred = vit(x)
print(pred.shape)
