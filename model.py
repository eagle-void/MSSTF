import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import List
from efficientnet_pytorch.model import MemoryEfficientSwish


def get_model_name():
    return 'MSSTF'


# 卷积encoder模块 (b, d, h, w)
class AttnMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            MemoryEfficientSwish(),
            nn.Conv2d(dim, dim, 1, 1, 0)
            # nn.Identity()
        )

    def forward(self, x):
        return self.act_block(x)


class AttnConvBlock(nn.Module):
    def __init__(self, input_dim, dim, group_split: List[int], kernel_sizes: List[int], num_heads=8, window_size=8, attn_drop=0.1,
                 proj_drop=0.1, qkv_bias=True):
        super().__init__()
        assert sum(group_split) == num_heads
        assert len(kernel_sizes) + 1 == len(group_split)
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scalor = self.dim_head ** -0.5
        self.kernel_sizes = kernel_sizes
        self.window_size = window_size
        self.group_split = group_split
        self.embedding = nn.Conv2d(in_channels=input_dim, out_channels=dim, kernel_size=1, padding=0, padding_mode='circular', bias=False)
        convs = []
        act_blocks = []
        qkvs = []
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            group_head = group_split[i]
            if group_head == 0:
                continue
            convs.append(nn.Conv2d(3 * self.dim_head * group_head, 3 * self.dim_head * group_head, kernel_size,
                                   1, kernel_size // 2, groups=3 * self.dim_head * group_head, padding_mode='replicate'))
            act_blocks.append(AttnMap(self.dim_head * group_head))
            qkvs.append(nn.Conv2d(dim, 3 * group_head * self.dim_head, 1, 1, 0, bias=qkv_bias))
            # projs.append(nn.Linear(group_head*self.dim_head, group_head*self.dim_head, bias=qkv_bias))
        if group_split[-1] != 0:
            self.global_q = nn.Conv2d(dim, group_split[-1] * self.dim_head, 1, 1, 0, bias=qkv_bias)
            self.global_kv = nn.Conv2d(dim, group_split[-1] * self.dim_head * 2, 1, 1, 0, bias=qkv_bias)
            # self.global_proj = nn.Linear(group_split[-1]*self.dim_head, group_split[-1]*self.dim_head, bias=qkv_bias)
            self.avgpool = nn.AvgPool2d(window_size, window_size) if window_size != 1 else nn.Identity()

        self.convs = nn.ModuleList(convs)
        self.act_blocks = nn.ModuleList(act_blocks)
        self.qkvs = nn.ModuleList(qkvs)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def high_fre_attntion(self, x, to_qkv: nn.Module, mixer: nn.Module, attn_block: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()
        qkv = to_qkv(x)  # (b (3 m d) h w)
        qkv = mixer(qkv).reshape(b, 3, -1, h, w).transpose(0, 1).contiguous()  # (3 b (m d) h w)
        q, k, v = qkv  # (b (m d) h w)
        attn = attn_block(q.mul(k)).mul(self.scalor)
        attn = self.attn_drop(torch.tanh(attn))
        res = attn.mul(v)  # (b (m d) h w)
        return res

    def low_fre_attention(self, x: torch.Tensor, to_q: nn.Module, to_kv: nn.Module, avgpool: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()

        q = to_q(x).reshape(b, -1, self.dim_head, h * w).transpose(-1, -2).contiguous()  # (b m (h w) d)
        kv = avgpool(x)  # (b c h w)
        kv = to_kv(kv).view(b, 2, -1, self.dim_head, (h * w) // (self.window_size ** 2)).permute(1, 0, 2, 4,
                                                                                                 3).contiguous()  # (2 b m (H W) d)
        k, v = kv  # (b m (H W) d)
        attn = self.scalor * q @ k.transpose(-1, -2)  # (b m (h w) (H W))
        attn = self.attn_drop(attn.softmax(dim=-1))
        res = attn @ v  # (b m (h w) d)
        res = res.transpose(2, 3).reshape(b, -1, h, w).contiguous()
        return res

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        x = self.embedding(x)
        res = []
        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue
            res.append(self.high_fre_attntion(x, self.qkvs[i], self.convs[i], self.act_blocks[i]))
        if self.group_split[-1] != 0:
            res.append(self.low_fre_attention(x, self.global_q, self.global_kv, self.avgpool))
        return self.proj_drop(self.proj(torch.cat(res, dim=1)))


# Transformer
class PositionalEmbedding(nn.Module):  # x 'b t d -> 1 t d'
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=8, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        attn = self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
        attn = self.dropout(attn)

        attn = torch.matmul(attn, v)
        attn = rearrange(attn, 'b h n d -> b n (h d)')
        return self.to_out(attn)


class Transformer(nn.Module):
    def __init__(self, d_model, depth=1, heads=8, dim_head=8, mlp_dim=64, dropout=0.1):
        super().__init__()
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

        self.pos_embedding = PositionalEmbedding(d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(d_model, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(d_model, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        b, t, _ = x.shape
        x = self.to_patch_embedding(x) + self.pos_embedding(x)
        x = self.dropout(x)

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MSSTF(nn.Module):
    def __init__(self, win_size, input_dim, d_model=16):
        super().__init__()
        self.win_size = win_size
        self.local_conv = AttnConvBlock(input_dim=input_dim, dim=d_model, group_split=[2, 2, 2, 2], kernel_sizes=[3, 3, 3])
        self.global_conv = AttnConvBlock(input_dim=input_dim, dim=d_model, group_split=[2, 2, 2, 2], kernel_sizes=[7, 5, 3])
        self.ltt = Transformer(d_model=d_model)
        self.stt = Transformer(d_model=d_model)
        self.feature_dim = 3 * d_model
        self.out_dim = 1
        self.featurefusion = nn.Sequential(
            Rearrange('b d t h w -> b t h w d'),
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.out_dim),
            Rearrange('b t h w d -> b d t h w'),
            nn.GELU()
            #nn.Sigmoid()
        )

    def to_piece(self, x):  # x: b d t h w
        x_list = []
        for i in range(x.shape[2] - self.win_size + 1):
            x_list.append(x[:, :, i:i + self.win_size])
        x_list = torch.stack(x_list)
        n, _, _, _, _, _ = x_list.shape
        x_list = rearrange(x_list, 'n b d t h w -> (n b) d t h w')
        return x_list, n

    def to_unpiece(self, x, n):
        x = rearrange(x, '(n b) d t h w -> n b d t h w', n=n)
        x = torch.cat((x[0, :, :, :-1], rearrange(x[:, :, :, -1], 't b d h w -> b d t h w')), dim=2)
        return x

    def model_solve(self, x, model, mode):
        b, d, t, h, w = x.shape
        if mode == 'conv':
            x = rearrange(x, 'b d t h w -> (b t) d h w')
            x = model(x)
            x = rearrange(x, '(b t) d h w -> b d t h w', b=b, t=t)
        elif mode == 'attn':
            x = rearrange(x, 'b d t h w -> (b h w) t d')
            x = model(x)
            x = rearrange(x, '(b h w) t d -> b d t h w', b=b, h=h, w=w)
        return x

    def forward(self, x):
        x_piece, n = self.to_piece(x)

        # small conv and long-term attn
        x = self.model_solve(x, self.local_conv, 'conv')
        x = self.model_solve(x, self.ltt, 'attn')

        # large-small conv and short-term attn
        x_piece_local = self.model_solve(x_piece, self.local_conv, 'conv')
        x_piece_global = self.model_solve(x_piece, self.global_conv, 'conv')
        x_piece_local = self.model_solve(x_piece_local, self.stt, 'attn')
        x_piece_global = self.model_solve(x_piece_global, self.stt, 'attn')

        x_piece_local = self.to_unpiece(x_piece_local, n)
        x_piece_global = self.to_unpiece(x_piece_global, n)
        x = torch.stack((x, x_piece_local, x_piece_global))
        x = rearrange(x, 'n b d t h w -> b (n d) t h w')
        x = self.featurefusion(x)
        return x


