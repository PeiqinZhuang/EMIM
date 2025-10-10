#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from slowfast.models.common import DropPath, Mlp
from spatial_correlation_sampler import SpatialCorrelationSampler
from mmcv.ops import Correlation
from einops import rearrange
import math
import pdb
from timm.models.layers import trunc_normal_






class CostvolumeAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        up_rate=None,
        attn_type='MultiScaleAttn',
        temporal_atten=False
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)


        self.attn_type = attn_type
        if self.attn_type == 'CostVolumeAttention2':
            self.attn = CostVolumeAttention2(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate
            )
        elif self.attn_type == 'CostVolumeAttention3':
            self.attn = CostVolumeAttention2(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate
            )
        elif self.attn_type == 'CostVolumeAttention4':
            self.attn = CostVolumeAttention2(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate
            )
        elif self.attn_type == 'CostVolumeAttention5':
            self.attn = CostVolumeAttention5(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate
            )
        elif self.attn_type == 'CostVolumeAttention6':
            self.attn = CostVolumeAttention6(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate
            )
        elif self.attn_type == 'CostVolumeAttention7':
            self.attn = CostVolumeAttention7(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate
            )
        elif self.attn_type == 'CostVolumeAttention8':
            self.attn = CostVolumeAttention8(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate
            )
        elif self.attn_type == 'CostVolumeAttention9':
            self.attn = CostVolumeAttention9(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate
            )
        elif self.attn_type == 'CostVolumeAttention10':
            self.attn = CostVolumeAttention10(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate
            )
        elif self.attn_type == 'CostVolumeAttention11':
            self.attn = CostVolumeAttention11(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate
            )
        elif self.attn_type == 'CostVolumeAttention12':
            self.attn = CostVolumeAttention12(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate
            )
        elif self.attn_type == 'CostVolumeAttention13':
            self.attn = CostVolumeAttention13(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate
            )
        elif self.attn_type == 'CostVolumeAttention14':
            self.attn = CostVolumeAttention14(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate
            )
        elif self.attn_type == 'CostVolumeAttention15':
            self.attn = CostVolumeAttention15(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate
            )
        elif self.attn_type == 'CostVolumeAttention16':
            self.attn = CostVolumeAttention16(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate
            )
        elif self.attn_type == 'CostVolumeAttention17':
            self.attn = CostVolumeAttention17(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate
            )
        elif self.attn_type == 'CostVolumeAttention18':
            self.attn = CostVolumeAttention18(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate
            )
        elif self.attn_type == 'CostVolumeAttention19':
            self.attn = CostVolumeAttention19(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate
            )
        elif self.attn_type == 'CostVolumeAttention20':
            self.attn = CostVolumeAttention20(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate
            )



        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # TODO: check the use case for up_rate, and merge the following lines
        if up_rate is not None and up_rate > 1:
            mlp_dim_out = dim * up_rate
        else:
            mlp_dim_out = dim_out
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
            drop_rate=drop_rate,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        self.temporal_atten = temporal_atten

        if self.temporal_atten:
            print('Use temporal attention!', flush=True)
            self.t_norm = norm_layer(dim)
            self.t_attn = Attention(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop_rate
            )


 

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'B C T H W -> B (T H W) C')

        x_block = self.attn(self.norm1(x))
  
        x = x + self.drop_path(x_block)

        if not self.temporal_atten:
            x_norm = self.norm2(x)
            x_mlp = self.mlp(x_norm)
            if self.dim != self.dim_out:
                x = self.proj(x_norm)
            x = x + self.drop_path(x_mlp)

            x = rearrange(x, 'B (T H W) C -> B C T H W', T=T, H=H, W=W)
            return x

        else:
            x = rearrange(x, 'B (T H W) C -> (B H W) T C', T=T, H=H, W=W)
            x = x + self.drop_path(self.t_attn(self.t_norm(x)))
            
            x_norm = self.norm2(x)
            x_mlp = self.mlp(x_norm)
            if self.dim != self.dim_out:
                x = self.proj(x_norm)
            x = x + self.drop_path(x_mlp)

            x = rearrange(x, '(B H W) T C -> B C T H W', T=T, H=H, W=W)
            return x





class CostVolumeAttention2(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
 

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)


        # self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=2*3+1,
        #             stride=1, padding=0, dilation=1)
        self.corr = Correlation(kernel_size=1, max_displacement=3, 
                        stride=1, padding=0, dilation=1)

    def forward(self, x):
        B, N, C = x.shape
        T = int(8) 
        H = int(math.sqrt((N) // T))
        W = H

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2] #[1, self.num_heads, T * H * W, C // self.num_heads]


        q = rearrange(q, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        k = rearrange(k, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        v = rearrange(v, 'b n (t h w) d -> (b n t) (h w) d', t=T, h=H, w=W)

        q_prefix = q.clone()
        k_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        k_postfix[:, :-1, ...] = k[:, 1:, ...]
        k_postfix[:, -1, ...] = k[:, -1, ...]

        q_prefix = rearrange(q_prefix, 'b t d h w -> (b t) d h w')
        q_prefix = F.normalize(q_prefix, dim=1)
        k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
        k_postfix = F.normalize(k_postfix, dim=1)

        corr = self.corr(q_prefix.contiguous(), k_postfix.contiguous()) #[B, K, K, H, W]
        corr = rearrange(corr, 'b u v h w -> b (u v) (h w)')

        motion_attn = torch.bmm(corr.transpose(-2, -1), corr) #[B, H*W, H*W]
        motion_attn = motion_attn.softmax(dim=-1)
        x = torch.bmm(motion_attn, v) 
        x = rearrange(x, '(b n t) (h w) d -> b n (t h w) d', b=B, n=self.num_heads, t=T, h=H, w=W)
        x = rearrange(x, 'b h n d -> b n (h d)', h=self.num_heads)

        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x



class CostVolumeAttention3(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5


        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)


        # self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=2*7+1,
        #             stride=1, padding=0, dilation=1)
        self.corr = Correlation(kernel_size=1, max_displacement=3, 
                        stride=1, padding=0, dilation=1)

    def forward(self, x):
        B, N, C = x.shape
        T = int(8) 
        H = int(math.sqrt((N) // T))
        W = H

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2] #[1, self.num_heads, T * H * W, C // self.num_heads]


        q = rearrange(q, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        k = rearrange(k, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        v = rearrange(v, 'b n (t h w) d -> (b n t) (h w) d', t=T, h=H, w=W)

        q_prefix = q.clone()
        k_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        k_postfix[:, :-1, ...] = k[:, 1:, ...]
        k_postfix[:, -1, ...] = k[:, -1, ...]

        q_prefix = rearrange(q_prefix, 'b t d h w -> (b t) d h w')
        q_prefix = F.normalize(q_prefix, dim=1)
        k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
        k_postfix = F.normalize(k_postfix, dim=1)

        corr = self.corr(q_prefix.contiguous(), k_postfix.contiguous()) #[B, K, K, H, W]
        corr = rearrange(corr, 'b u v h w -> b (u v) (h w)')

        motion_attn = torch.bmm(corr.transpose(-2, -1), corr) #[B, H*W, H*W]
        motion_attn = motion_attn.softmax(dim=-1)
        x = torch.bmm(motion_attn, v) 
        x = rearrange(x, '(b n t) (h w) d -> b n (t h w) d', b=B, n=self.num_heads, t=T, h=H, w=W)
        x = rearrange(x, 'b h n d -> b n (h d)', h=self.num_heads)

        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x


class CostVolumeAttention4(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5


        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

        # self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=2*3+1,
        #             stride=1, padding=0, dilation=1)
        self.corr = Correlation(kernel_size=1, max_displacement=3, 
                        stride=1, padding=0, dilation=1)

    def forward(self, x):
        B, N, C = x.shape
        T = int(8) 
        H = int(math.sqrt((N) // T))
        W = H

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2] #[1, self.num_heads, T * H * W, C // self.num_heads]

        q = rearrange(q, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        k = rearrange(k, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        v = rearrange(v, 'b n (t h w) d -> (b n t) (h w) d', t=T, h=H, w=W)

        q_prefix = q.clone()
        k_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        k_postfix[:, :-1, ...] = k[:, 1:, ...]
        k_postfix[:, -1, ...] = k[:, -1, ...]

        q_prefix = rearrange(q_prefix, 'b t d h w -> (b t) d h w')
        q_prefix = F.normalize(q_prefix, dim=1)
        k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
        k_postfix = F.normalize(k_postfix, dim=1)

        corr = self.corr(q_prefix.contiguous(), k_postfix.contiguous()) #[B, K, K, H, W]
        corr = F.relu(corr)
        corr = rearrange(corr, 'b u v h w -> b (u v) (h w)')

        motion_attn = torch.bmm(corr.transpose(-2, -1), corr) #[B, H*W, H*W]
        motion_attn = motion_attn.softmax(dim=-1)
        x = torch.bmm(motion_attn, v) 
        x = rearrange(x, '(b n t) (h w) d -> b n (t h w) d', b=B, n=self.num_heads, t=T, h=H, w=W)
        x = rearrange(x, 'b h n d -> b n (h d)', h=self.num_heads)

        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x




class CostVolumeAttention5(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)



        # self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=2*3+1,
        #             stride=1, padding=0, dilation=1)
        self.corr = Correlation(kernel_size=1, max_displacement=5, 
                        stride=1, padding=0, dilation=1)

    def forward(self, x):
        B, N, C = x.shape
        T = int(8) 
        H = int(math.sqrt((N) // T))
        W = H

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2] #[1, self.num_heads, T * H * W, C // self.num_heads]


        q = rearrange(q, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        k = rearrange(k, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        v = rearrange(v, 'b n (t h w) d -> (b n t) (h w) d', t=T, h=H, w=W)

        q_prefix = q.clone()
        k_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        k_postfix[:, :-1, ...] = k[:, 1:, ...]
        k_postfix[:, -1, ...] = k[:, -1, ...]

        q_prefix = rearrange(q_prefix, 'b t d h w -> (b t) d h w')
        q_prefix = F.normalize(q_prefix, dim=1)
        k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
        k_postfix = F.normalize(k_postfix, dim=1)

        corr = self.corr(q_prefix.contiguous(), k_postfix.contiguous()) #[B, K, K, H, W]
        corr = rearrange(corr, 'b u v h w -> b (u v) (h w)')

        motion_attn = torch.bmm(corr.transpose(-2, -1), corr) #[B, H*W, H*W]
        motion_attn = motion_attn.softmax(dim=-1)
        x = torch.bmm(motion_attn, v) 
        x = rearrange(x, '(b n t) (h w) d -> b n (t h w) d', b=B, n=self.num_heads, t=T, h=H, w=W)
        x = rearrange(x, 'b h n d -> b n (h d)', h=self.num_heads)

        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x



class CostVolumeAttention6(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
   

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)



        # self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=2*3+1,
        #             stride=1, padding=0, dilation=1)
        self.corr = Correlation(kernel_size=1, max_displacement=7, 
                        stride=1, padding=0, dilation=1)

    def forward(self, x):
        B, N, C = x.shape
        T = int(8) 
        H = int(math.sqrt((N) // T))
        W = H

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2] #[1, self.num_heads, T * H * W, C // self.num_heads]


        q = rearrange(q, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        k = rearrange(k, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        v = rearrange(v, 'b n (t h w) d -> (b n t) (h w) d', t=T, h=H, w=W)

        q_prefix = q.clone()
        k_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        k_postfix[:, :-1, ...] = k[:, 1:, ...]
        k_postfix[:, -1, ...] = k[:, -1, ...]

        q_prefix = rearrange(q_prefix, 'b t d h w -> (b t) d h w')
        q_prefix = F.normalize(q_prefix, dim=1)
        k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
        k_postfix = F.normalize(k_postfix, dim=1)

        corr = self.corr(q_prefix.contiguous(), k_postfix.contiguous()) #[B, K, K, H, W]
        corr = rearrange(corr, 'b u v h w -> b (u v) (h w)')

        motion_attn = torch.bmm(corr.transpose(-2, -1), corr) #[B, H*W, H*W]
        motion_attn = motion_attn.softmax(dim=-1)
        x = torch.bmm(motion_attn, v) 
        x = rearrange(x, '(b n t) (h w) d -> b n (t h w) d', b=B, n=self.num_heads, t=T, h=H, w=W)
        x = rearrange(x, 'b h n d -> b n (h d)', h=self.num_heads)

        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x

class CostVolumeAttention7(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5


        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

        # self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=2*3+1,
        #             stride=1, padding=0, dilation=1)
        self.corr = Correlation(kernel_size=1, max_displacement=3, 
                        stride=1, padding=0, dilation=1)

    def forward(self, x):
        B, N, C = x.shape
        T = int(8) 
        H = int(math.sqrt((N) // T))
        W = H

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2] #[1, self.num_heads, T * H * W, C // self.num_heads]


        q = rearrange(q, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        k = rearrange(k, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        v = rearrange(v, 'b n (t h w) d -> (b n t) (h w) d', t=T, h=H, w=W)

        q_prefix = q.clone()
        k_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        k_postfix[:, :-1, ...] = k[:, 1:, ...]
        k_postfix[:, -1, ...] = k[:, -1, ...]

        q_prefix = rearrange(q_prefix, 'b t d h w -> (b t) d h w')
        q_prefix = F.normalize(q_prefix, dim=1)
        k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
        k_postfix = F.normalize(k_postfix, dim=1)

        corr = self.corr(q_prefix.contiguous(), k_postfix.contiguous()) #[B, K, K, H, W]
        corr = rearrange(corr, 'b u v h w -> b (u v) (h w)')

        motion_attn = torch.bmm(corr.transpose(-2, -1), corr) #[B, H*W, H*W]
        motion_attn = motion_attn.softmax(dim=-1)
        motion_x = torch.bmm(motion_attn, v) 
        motion_x = rearrange(motion_x, '(b n t) (h w) d -> b n (t h w) d', b=B, n=self.num_heads, t=T, h=H, w=W)
        motion_x = rearrange(motion_x, 'b h n d -> b n (h d)', h=self.num_heads)

        q_prefix = rearrange(q_prefix, 'b d h w -> b (h w) d')
        k_postfix = rearrange(k_postfix, 'b d h w -> b d (h w)')
        appearance_attn = torch.bmm(q_prefix, k_postfix)
        appearance_attn = appearance_attn.softmax(dim=-1)
        appearance_x = torch.bmm(appearance_attn, v)
        appearance_x = rearrange(appearance_x, '(b n t) (h w) d -> b n (t h w) d', b=B, n=self.num_heads, t=T, h=H, w=W)
        appearance_x = rearrange(appearance_x, 'b h n d -> b n (h d)', h=self.num_heads)

        x = 0.5 * appearance_x + 0.5 * motion_x

        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x


class CostVolumeAttention8(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
      

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)


        # self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=2*3+1,
        #             stride=1, padding=0, dilation=1)
        self.corr = Correlation(kernel_size=1, max_displacement=3, 
                        stride=1, padding=0, dilation=1)

    def forward(self, x):
        B, N, C = x.shape
        T = int(8) 
        H = int(math.sqrt((N) // T))
        W = H

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2] #[1, self.num_heads, T * H * W, C // self.num_heads]


        q = rearrange(q, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        k = rearrange(k, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        v = rearrange(v, 'b n (t h w) d -> (b n t) (h w) d', t=T, h=H, w=W)

        q_prefix = q.clone()
        k_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        k_postfix[:, :-1, ...] = k[:, 1:, ...]
        k_postfix[:, -1, ...] = k[:, -1, ...]

        q_prefix = rearrange(q_prefix, 'b t d h w -> (b t) d h w')
        q_prefix = F.normalize(q_prefix, dim=1)
        k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
        k_postfix = F.normalize(k_postfix, dim=1)

        corr = self.corr(q_prefix.contiguous(), k_postfix.contiguous()) #[B, K, K, H, W]
        corr = rearrange(corr, 'b u v h w -> b (u v) (h w)')

        motion_attn = torch.bmm(corr.transpose(-2, -1), corr) #[B, H*W, H*W]
        motion_attn = motion_attn.softmax(dim=-1)
        x = torch.bmm(motion_attn, v) 
        x = rearrange(x, '(b n t) (h w) d -> b n (t h w) d', b=B, n=self.num_heads, t=T, h=H, w=W)

        
        cls_q = rearrange(cls_q, 'b h n d -> (b h) n d')
        q = rearrange(q, 'b t d h w -> b d (t h w)')
        v = rearrange(v, '(b n t) (h w) d -> (b n) (t h w) d', b=B, n=self.num_heads, t=T, h=H, w=W)
        cls_attn = torch.bmm(cls_q, q) #[b, 1, thw]
        cls_attn = cls_attn.softmax(dim=-1)
        cls_tok = torch.bmm(cls_attn, v)
        cls_tok = rearrange(cls_tok, '(b h) n d -> b h n d', b=B)

        x = torch.cat([cls_tok, x], dim=2)
        x = rearrange(x, 'b h n d -> b n (h d)', h=self.num_heads)

        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x




class CostVolumeAttention9(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
  
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

        # self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=2*3+1,
        #             stride=1, padding=0, dilation=1)
        self.corr = Correlation(kernel_size=1, max_displacement=3, 
                        stride=1, padding=0, dilation=1)

        self.pos_embed = nn.Parameter(torch.zeros(
            1, 7 * 7 , 14 * 14
        ))
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, N, C = x.shape
        T = int(8) 
        H = int(math.sqrt((N) // T))
        W = H

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2] #[1, self.num_heads, T * H * W, C // self.num_heads]

        q = rearrange(q, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        k = rearrange(k, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        v = rearrange(v, 'b n (t h w) d -> (b n t) (h w) d', t=T, h=H, w=W)

        q_prefix = q.clone()
        k_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        k_postfix[:, :-1, ...] = k[:, 1:, ...]
        k_postfix[:, -1, ...] = k[:, -1, ...]

        q_prefix = rearrange(q_prefix, 'b t d h w -> (b t) d h w')
        q_prefix = F.normalize(q_prefix, dim=1)
        k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
        k_postfix = F.normalize(k_postfix, dim=1)

        corr = self.corr(q_prefix.contiguous(), k_postfix.contiguous()) #[B, K, K, H, W]
        corr = rearrange(corr, 'b u v h w -> b (u v) (h w)')
        corr = corr + self.pos_embed

        motion_attn = torch.bmm(corr.transpose(-2, -1), corr) #[B, H*W, H*W]
        motion_attn = motion_attn.softmax(dim=-1)
        x = torch.bmm(motion_attn, v) 
        x = rearrange(x, '(b n t) (h w) d -> b n (t h w) d', b=B, n=self.num_heads, t=T, h=H, w=W)
        x = rearrange(x, 'b h n d -> b n (h d)', h=self.num_heads)

        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x


class CostVolumeAttention10(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5



        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

     

        # self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=2*3+1,
        #             stride=1, padding=0, dilation=1)
        self.corr = Correlation(kernel_size=1, max_displacement=3, 
                        stride=1, padding=0, dilation=1)

        self.pos_embed = nn.Parameter(torch.zeros(
            1, 7 * 7 , 14 * 14
        ))
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, N, C = x.shape
        T = int(8) 
        H = int(math.sqrt((N) // T))
        W = H

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2] #[1, self.num_heads, T * H * W, C // self.num_heads]


        q = rearrange(q, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        k = rearrange(k, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        v = rearrange(v, 'b n (t h w) d -> (b n t) (h w) d', t=T, h=H, w=W)

        q_prefix = q.clone()
        k_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        k_postfix[:, :-1, ...] = k[:, 1:, ...]
        k_postfix[:, -1, ...] = k[:, -1, ...]

        q_prefix = rearrange(q_prefix, 'b t d h w -> (b t) d h w')
        q_prefix = F.normalize(q_prefix, dim=1)
        k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
        k_postfix = F.normalize(k_postfix, dim=1)

        corr = self.corr(q_prefix.contiguous(), k_postfix.contiguous()) #[B, K, K, H, W]
        corr = rearrange(corr, 'b u v h w -> b (u v) (h w)')

        corr = corr + self.pos_embed
        motion_attn = torch.bmm(corr.transpose(-2, -1), corr) #[B, H*W, H*W]
        motion_attn = motion_attn.softmax(dim=-1)
        motion_x = torch.bmm(motion_attn, v) 
        motion_x = rearrange(motion_x, '(b n t) (h w) d -> b n (t h w) d', b=B, n=self.num_heads, t=T, h=H, w=W)
        motion_x = rearrange(motion_x, 'b h n d -> b n (h d)', h=self.num_heads)

        q_prefix = rearrange(q_prefix, 'b d h w -> b (h w) d')
        k_postfix = rearrange(k_postfix, 'b d h w -> b d (h w)')
        appearance_attn = torch.bmm(q_prefix, k_postfix)
        appearance_attn = appearance_attn.softmax(dim=-1)
        appearance_x = torch.bmm(appearance_attn, v)
        appearance_x = rearrange(appearance_x, '(b n t) (h w) d -> b n (t h w) d', b=B, n=self.num_heads, t=T, h=H, w=W)
        appearance_x = rearrange(appearance_x, 'b h n d -> b n (h d)', h=self.num_heads)

        x = 0.5 * appearance_x + 0.5 * motion_x

        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x



class CostVolumeAttention11(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5


        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)



        # self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=2*3+1,
        #             stride=1, padding=0, dilation=1)
        self.corr = Correlation(kernel_size=1, max_displacement=3, 
                        stride=1, padding=0, dilation=1)
        
        self.proj_corr = nn.Linear(7 * 7, head_dim)

    def forward(self, x):
        B, N, C = x.shape
        T = int(8) 
        H = int(math.sqrt((N) // T))
        W = H

        qk = (
            self.qk(x)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k = qk[0], qk[1] #[1, self.num_heads, T * H * W, C // self.num_heads]


        q = rearrange(q, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        k = rearrange(k, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)

        q_prefix = q.clone()
        k_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        k_postfix[:, :-1, ...] = k[:, 1:, ...]
        k_postfix[:, -1, ...] = k[:, -1, ...]

        q_prefix = rearrange(q_prefix, 'b t d h w -> (b t) d h w')
        q_prefix = F.normalize(q_prefix, dim=1)
        k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
        k_postfix = F.normalize(k_postfix, dim=1)

        corr = self.corr(q_prefix.contiguous(), k_postfix.contiguous()) #[B, K, K, H, W]
        corr = rearrange(corr, 'b u v h w -> b (h w) (u v)')
        v = self.proj_corr(corr)

        x = rearrange(v, '(b n t) (h w) d -> b n (t h w) d', b=B, n=self.num_heads, t=T, h=H, w=W)
        x = rearrange(x, 'b h n d -> b n (h d)', h=self.num_heads)

        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x


class CostVolumeAttention12(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5


        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)


        self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=2*3+1,
                    stride=1, padding=0, dilation=1)
        # self.corr = Correlation(kernel_size=1, max_displacement=3, 
        #                 stride=1, padding=0, dilation=1)

    def forward(self, x):
        B, N, C = x.shape
        T = int(8) 
        H = int(math.sqrt((N) // T))
        W = H

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2] #[1, self.num_heads, T * H * W, C // self.num_heads]
  

        q = rearrange(q, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        k = rearrange(k, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        v = rearrange(v, 'b n (t h w) d -> (b n t) (h w) d', t=T, h=H, w=W)

        q_prefix = q.clone()
        k_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        k_postfix[:, :-1, ...] = k[:, 1:, ...]
        k_postfix[:, -1, ...] = k[:, -1, ...]

        q_prefix = rearrange(q_prefix, 'b t d h w -> (b t) d h w')
        q_prefix = F.normalize(q_prefix, dim=1)
        k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
        k_postfix = F.normalize(k_postfix, dim=1)

        corr = self.corr(q_prefix.contiguous(), k_postfix.contiguous()) #[B, K, K, H, W]
        corr = rearrange(corr, 'b u v h w -> b (u v) (h w)')

        motion_attn = torch.bmm(corr.transpose(-2, -1), corr) #[B, H*W, H*W]
        motion_attn = motion_attn.softmax(dim=-1)
        x = torch.bmm(motion_attn, v) 
        x = rearrange(x, '(b n t) (h w) d -> b n (t h w) d', b=B, n=self.num_heads, t=T, h=H, w=W)
        x = rearrange(x, 'b h n d -> b n (h d)', h=self.num_heads)

        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x


class CostVolumeAttention13(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
 

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)


        self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=2*3+1,
                    stride=1, padding=0, dilation=1)
        # self.corr = Correlation(kernel_size=1, max_displacement=3, 
        #                 stride=1, padding=0, dilation=1)
        
        self.max_d = 3
        self.proj_corr = nn.Linear((2 * self.max_d + 1 )**2, dim // num_heads)


    def forward(self, x):
        B, N, C = x.shape
        T = int(8) 
        H = int(math.sqrt((N) // T))
        W = H

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2] #[1, self.num_heads, T * H * W, C // self.num_heads]


        q = rearrange(q, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        k = rearrange(k, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        v = rearrange(v, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)

        q_prefix = q.clone()
        k_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        k_postfix[:, :-1, ...] = k[:, 1:, ...]
        k_postfix[:, -1, ...] = k[:, -1, ...]

        v_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        v_postfix[:, :-1, ...] = v[:, 1:, ...]
        v_postfix[:, -1, ...] = v[:, -1, ...]

        q_prefix = rearrange(q_prefix, 'b t d h w -> (b t) d h w')
        q_prefix = F.normalize(q_prefix, dim=1)
        k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
        k_postfix = F.normalize(k_postfix, dim=1)

        v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
        v_postfix_local = torch.zeros_like(v_postfix).reshape(-1, C // self.num_heads, H*W, 1)
        v_postfix_local = v_postfix_local.repeat(1, 1, 1, (2 * self.max_d + 1)**2)
        for i in torch.range(-self.max_d, self.max_d):
            x_offset = torch.arange(0, H) + i
            x_offset = torch.clip(x_offset, min=0, max=H-1)

            for j in torch.range(-self.max_d, self.max_d):
                y_offset = torch.arange(0, H) + j
                y_offset = torch.clip(y_offset, min=0, max=W-1)

                offset = torch.meshgrid(x_offset, y_offset)
                offset = H * offset[0] + offset[1]
                offset = offset.reshape(-1).long().cuda()

                offset_tensor = torch.index_select(
                    v_postfix.reshape(-1, C // self.num_heads, H*W), 
                    dim=-1, index=offset)

                offset_index = (2 * self.max_d + 1) * (i + self.max_d) + (j + self.max_d)
                offset_index = offset_index.long()
                v_postfix_local[:,:,:, offset_index] = offset_tensor


        corr = self.corr(q_prefix.contiguous(), k_postfix.contiguous()) #[B, K, K, H, W]
        corr = rearrange(corr, 'b u v h w -> b (h w) (u v)')


        motion_feature = self.proj_corr(corr)
        motion_feature = rearrange(motion_feature, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W).contiguous()

        local_attn = corr.softmax(dim=-1)
        local_feature = torch.einsum('b s u, b d s u -> b s d', corr, v_postfix_local)
        local_feature = rearrange(local_feature, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W).contiguous()

        x = motion_feature + local_feature

        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x

class CostVolumeAttention14(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
 

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)


        self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=2*3+1,
                    stride=1, padding=0, dilation=1)
        # self.corr = Correlation(kernel_size=1, max_displacement=3, 
        #                 stride=1, padding=0, dilation=1)
        
        self.max_d = 3
        # self.proj_corr = nn.Linear((2 * self.max_d + 1 )**2, dim // num_heads)


    def forward(self, x):
        B, N, C = x.shape
        T = int(8) 
        H = int(math.sqrt((N) // T))
        W = H

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2] #[1, self.num_heads, T * H * W, C // self.num_heads]


        q = rearrange(q, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        k = rearrange(k, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        v = rearrange(v, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)

        q_prefix = q.clone()
        k_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        k_postfix[:, :-1, ...] = k[:, 1:, ...]
        k_postfix[:, -1, ...] = k[:, -1, ...]

        v_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        v_postfix[:, :-1, ...] = v[:, 1:, ...]
        v_postfix[:, -1, ...] = v[:, -1, ...]

        q_prefix = rearrange(q_prefix, 'b t d h w -> (b t) d h w')
        q_prefix = F.normalize(q_prefix, dim=1)
        k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
        k_postfix = F.normalize(k_postfix, dim=1)

        v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
        v_postfix_local = torch.zeros_like(v_postfix).reshape(-1, C // self.num_heads, H*W, 1)
        v_postfix_local = v_postfix_local.repeat(1, 1, 1, (2 * self.max_d + 1)**2)
        for i in torch.range(-self.max_d, self.max_d):
            x_offset = torch.arange(0, H) + i
            x_offset = torch.clip(x_offset, min=0, max=H-1)

            for j in torch.range(-self.max_d, self.max_d):
                y_offset = torch.arange(0, H) + j
                y_offset = torch.clip(y_offset, min=0, max=W-1)

                offset = torch.meshgrid(x_offset, y_offset)
                offset = H * offset[0] + offset[1]
                offset = offset.reshape(-1).long().cuda()

                offset_tensor = torch.index_select(
                    v_postfix.reshape(-1, C // self.num_heads, H*W), 
                    dim=-1, index=offset)

                offset_index = (2 * self.max_d + 1) * (i + self.max_d) + (j + self.max_d)
                offset_index = offset_index.long()
                v_postfix_local[:,:,:, offset_index] = offset_tensor


        corr = self.corr(q_prefix.contiguous(), k_postfix.contiguous()) #[B, K, K, H, W]
        corr = rearrange(corr, 'b u v h w -> b (h w) (u v)')


        local_attn = corr.softmax(dim=-1)
        local_feature = torch.einsum('b s u, b d s u -> b s d', corr, v_postfix_local)
        local_feature = rearrange(local_feature, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W).contiguous()

        x = local_feature

        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x


class CostVolumeAttention15(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
 

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)


        self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=2*3+1,
                    stride=1, padding=0, dilation=1)
        # self.corr = Correlation(kernel_size=1, max_displacement=3, 
        #                 stride=1, padding=0, dilation=1)
        
        self.max_d = 3
        self.proj_corr = nn.Linear((2 * self.max_d + 1 )**2, dim // num_heads)


    def forward(self, x):
        B, N, C = x.shape
        T = int(8) 
        H = int(math.sqrt((N) // T))
        W = H

        qk = (
            self.qkv(x)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k = qk[0], qk[1] #[1, self.num_heads, T * H * W, C // self.num_heads]


        q = rearrange(q, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        k = rearrange(k, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)

        q_prefix = q.clone()
        k_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        k_postfix[:, :-1, ...] = k[:, 1:, ...]
        k_postfix[:, -1, ...] = k[:, -1, ...]

        v_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        v_postfix[:, :-1, ...] = v[:, 1:, ...]
        v_postfix[:, -1, ...] = v[:, -1, ...]

        q_prefix = rearrange(q_prefix, 'b t d h w -> (b t) d h w')
        q_prefix = F.normalize(q_prefix, dim=1)
        k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
        k_postfix = F.normalize(k_postfix, dim=1)


        corr = self.corr(q_prefix.contiguous(), k_postfix.contiguous()) #[B, K, K, H, W]
        corr = rearrange(corr, 'b u v h w -> b (h w) (u v)')


        motion_feature = self.proj_corr(corr)
        motion_feature = rearrange(motion_feature, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W).contiguous()
        x = motion_feature 

        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x


class CostVolumeAttention16(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5



        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

     

        # self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=2*3+1,
        #             stride=1, padding=0, dilation=1)
        self.corr = Correlation(kernel_size=1, max_displacement=3, 
                        stride=1, padding=0, dilation=1)

        self.pos_embed = nn.Parameter(torch.zeros(
            1, 7 * 7 , 14 * 14
        ))
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, N, C = x.shape
        T = int(8) 
        H = int(math.sqrt((N) // T))
        W = H

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2] #[1, self.num_heads, T * H * W, C // self.num_heads]


        q = rearrange(q, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        k = rearrange(k, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        v = rearrange(v, 'b n (t h w) d -> (b n t) (h w) d', t=T, h=H, w=W)

        q_prefix = q.clone()
        k_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        k_postfix[:, :-1, ...] = k[:, 1:, ...]
        k_postfix[:, -1, ...] = k[:, -1, ...]

        q_prefix = rearrange(q_prefix, 'b t d h w -> (b t) d h w')
        q_prefix = F.normalize(q_prefix, dim=1)
        k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
        k_postfix = F.normalize(k_postfix, dim=1)

        corr = self.corr(q_prefix.contiguous(), k_postfix.contiguous()) #[B, K, K, H, W]
        corr = rearrange(corr, 'b u v h w -> b (u v) (h w)')

        corr = corr + self.pos_embed
        motion_attn = torch.bmm(corr.transpose(-2, -1), corr) #[B, H*W, H*W]
        motion_attn = motion_attn.softmax(dim=-1)
        motion_x = torch.bmm(motion_attn, v) 
        motion_x = rearrange(motion_x, '(b n t) (h w) d -> b n (t h w) d', b=B, n=self.num_heads, t=T, h=H, w=W)
        motion_x = rearrange(motion_x, 'b h n d -> b n (h d)', h=self.num_heads)

        q_prefix = rearrange(q_prefix, 'b d h w -> b (h w) d')
        appearance_attn = torch.einsum('b h d, b w d -> b h w', q_prefix, q_prefix)
        appearance_attn = appearance_attn.softmax(dim=-1)
        appearance_x = torch.bmm(appearance_attn, v)
        appearance_x = rearrange(appearance_x, '(b n t) (h w) d -> b n (t h w) d', b=B, n=self.num_heads, t=T, h=H, w=W)
        appearance_x = rearrange(appearance_x, 'b h n d -> b n (h d)', h=self.num_heads)

        x = 0.5 * appearance_x + 0.5 * motion_x

        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x



class CostVolumeAttention17(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.head_dim = dim // num_heads


        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

     

        # self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=2*3+1,
        #             stride=1, padding=0, dilation=1)
        self.corr = Correlation(kernel_size=1, max_displacement=3, 
                        stride=1, padding=0, dilation=1)

        self.window_size = 7

        self.pos_embed1 = nn.Parameter(torch.zeros(
            1, self.window_size ** 2, 7 * 7
        ))

        self.pos_embed2 = nn.Parameter(torch.zeros(
            1, self.window_size ** 2, self.head_dim
        ))

        trunc_normal_(self.pos_embed1, std=0.02)
        trunc_normal_(self.pos_embed2, std=0.02)
        
    def forward(self, x):
        B, N, C = x.shape
        T = int(8) 
        H = int(math.sqrt((N) // T))
        W = H

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2] #[1, self.num_heads, T * H * W, C // self.num_heads]


        q = rearrange(q, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        k = rearrange(k, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        v = rearrange(v, 'b n (t h w) d -> (b n t) (h w) d', t=T, h=H, w=W)

        q_prefix = q.clone()
        k_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        k_postfix[:, :-1, ...] = k[:, 1:, ...]
        k_postfix[:, -1, ...] = k[:, -1, ...]

        q_prefix = rearrange(q_prefix, 'b t d h w -> (b t) d h w')
        q_prefix = F.normalize(q_prefix, dim=1)
        k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
        k_postfix = F.normalize(k_postfix, dim=1)

        corr = self.corr(q_prefix.contiguous(), k_postfix.contiguous()) #[B, K, K, H, W]
        corr = rearrange(corr, 'b u v h w -> b h w (u v)')

        corr = corr.reshape(-1, self.window_size, H // self.window_size,
                                self.window_size, W // self.window_size, 7*7)
        corr = rearrange(corr, 'b h nh w nw d -> (b nh nw) (h w) d')

        corr = corr + self.pos_embed1
        motion_attn = torch.bmm(corr, corr.transpose(-2, -1))
        motion_attn = motion_attn.softmax(dim=-1)

        motion_v = v.reshape(-1, self.window_size, H // self.window_size, 
                                self.window_size, W // self.window_size, self.head_dim)
        motion_v = rearrange(motion_v, 'b h nh w nw d -> (b nh nw) (h w) d')
        motion_x = torch.bmm(motion_attn, motion_v)
        motion_x = rearrange(motion_x, '(b nh nw) (h w) d -> b (h nh) (w nw) d', nh=H // self.window_size, nw=W // self.window_size,
                                h=self.window_size, w=self.window_size)
      

        q_prefix = rearrange(q_prefix, 'b d h w -> b h w d')
        q_prefix = q_prefix.reshape(-1, self.window_size, H // self.window_size,
                                       self.window_size, W // self.window_size, self.head_dim)
        q_prefix = rearrange(q_prefix, 'b h nh w nw d -> (b nh nw) (h w) d')
        q_prefix = q_prefix + self.pos_embed2

        appearance_attn = torch.bmm(q_prefix, q_prefix.transpose(-2, -1))
        appearance_attn = appearance_attn.softmax(dim=-1)

        appearance_v = motion_v.clone()
        appearance_x = torch.bmm(appearance_attn, appearance_v)
        appearance_x = rearrange(appearance_x, '(b nh nw) (h w) d -> b (h nh) (w nw) d', nh=H // self.window_size, nw=W // self.window_size,
                                h=self.window_size, w=self.window_size)

        x = 0.5 * appearance_x + 0.5 * motion_x
        x = rearrange(x, '(b n t) h w d -> b (t h w) (n d)', b=B, t=T, n=self.num_heads, h=H, w=W)


        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CostVolumeAttention18(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5



        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

     

        # self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=2*3+1,
        #             stride=1, padding=0, dilation=1)
        self.corr = Correlation(kernel_size=1, max_displacement=3, 
                        stride=1, padding=0, dilation=1)

 

        self.sigmma = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x):
        B, N, C = x.shape
        T = int(8) 
        H = int(math.sqrt((N) // T))
        W = H

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2] #[1, self.num_heads, T * H * W, C // self.num_heads]


        q = rearrange(q, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        k = rearrange(k, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        v = rearrange(v, 'b n (t h w) d -> (b n t) (h w) d', t=T, h=H, w=W)

        q_prefix = q.clone()
        k_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        k_postfix[:, :-1, ...] = k[:, 1:, ...]
        k_postfix[:, -1, ...] = k[:, -1, ...]

        q_prefix = rearrange(q_prefix, 'b t d h w -> (b t) d h w')
        q_prefix = F.normalize(q_prefix, dim=1)
        k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
        k_postfix = F.normalize(k_postfix, dim=1)

        corr = self.corr(q_prefix.contiguous(), k_postfix.contiguous()) #[B, K, K, H, W]
        corr = rearrange(corr, 'b u v h w -> b (u v) (h w)')

        motion_attn = torch.bmm(corr.transpose(-2, -1), corr) #[B, H*W, H*W]
        motion_attn = motion_attn.softmax(dim=-1)
        motion_x = torch.bmm(motion_attn, v) 
        motion_x = rearrange(motion_x, '(b n t) (h w) d -> b n (t h w) d', b=B, n=self.num_heads, t=T, h=H, w=W)
        motion_x = rearrange(motion_x, 'b h n d -> b n (h d)', h=self.num_heads)

        q_prefix = rearrange(q_prefix, 'b d h w -> b (h w) d')
        k_postfix = rearrange(k_postfix, 'b d h w -> b d (h w)')
        appearance_attn = torch.bmm(q_prefix, k_postfix)
        appearance_attn = appearance_attn.softmax(dim=-1)
        appearance_x = torch.bmm(appearance_attn, v)
        appearance_x = rearrange(appearance_x, '(b n t) (h w) d -> b n (t h w) d', b=B, n=self.num_heads, t=T, h=H, w=W)
        appearance_x = rearrange(appearance_x, 'b h n d -> b n (h d)', h=self.num_heads)

        x = torch.sigmoid(self.sigmma) * appearance_x + (1 - torch.sigmoid(self.sigmma)) * motion_x

        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x


class CostVolumeAttention19(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
 

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)


        self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=2*3+1,
                    stride=1, padding=0, dilation=1)
        # self.corr = Correlation(kernel_size=1, max_displacement=3, 
        #                 stride=1, padding=0, dilation=1)
        
        self.max_d = 3
        self.proj_corr = nn.Linear((2 * self.max_d + 1 )**2, dim // num_heads)

        self.sigmma = nn.Parameter(torch.Tensor([0.0]))


    def forward(self, x):
        B, N, C = x.shape
        T = int(8) 
        H = int(math.sqrt((N) // T))
        W = H

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2] #[1, self.num_heads, T * H * W, C // self.num_heads]


        q = rearrange(q, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        k = rearrange(k, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        v = rearrange(v, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)

        q_prefix = q.clone()
        k_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        k_postfix[:, :-1, ...] = k[:, 1:, ...]
        k_postfix[:, -1, ...] = k[:, -1, ...]

        v_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        v_postfix[:, :-1, ...] = v[:, 1:, ...]
        v_postfix[:, -1, ...] = v[:, -1, ...]

        q_prefix = rearrange(q_prefix, 'b t d h w -> (b t) d h w')
        q_prefix = F.normalize(q_prefix, dim=1)
        k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
        k_postfix = F.normalize(k_postfix, dim=1)

        v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
        v_postfix_local = torch.zeros_like(v_postfix).reshape(-1, C // self.num_heads, H*W, 1)
        v_postfix_local = v_postfix_local.repeat(1, 1, 1, (2 * self.max_d + 1)**2)
        for i in torch.range(-self.max_d, self.max_d):
            x_offset = torch.arange(0, H) + i
            x_offset = torch.clip(x_offset, min=0, max=H-1)

            for j in torch.range(-self.max_d, self.max_d):
                y_offset = torch.arange(0, H) + j
                y_offset = torch.clip(y_offset, min=0, max=W-1)

                offset = torch.meshgrid(x_offset, y_offset)
                offset = H * offset[0] + offset[1]
                offset = offset.reshape(-1).long().cuda()

                offset_tensor = torch.index_select(
                    v_postfix.reshape(-1, C // self.num_heads, H*W), 
                    dim=-1, index=offset)

                offset_index = (2 * self.max_d + 1) * (i + self.max_d) + (j + self.max_d)
                offset_index = offset_index.long()
                v_postfix_local[:,:,:, offset_index] = offset_tensor


        corr = self.corr(q_prefix.contiguous(), k_postfix.contiguous()) #[B, K, K, H, W]
        corr = rearrange(corr, 'b u v h w -> b (h w) (u v)')


        motion_feature = self.proj_corr(corr)
        motion_feature = rearrange(motion_feature, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W).contiguous()

        local_attn = corr.softmax(dim=-1)
        local_feature = torch.einsum('b s u, b d s u -> b s d', corr, v_postfix_local)
        local_feature = rearrange(local_feature, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W).contiguous()

        x = torch.sigmoid(self.sigmma) * motion_feature + (1 - torch.sigmoid(self.sigmma)) * local_feature

        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x



class CostVolumeAttention20(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5



        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

     

        # self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=2*3+1,
        #             stride=1, padding=0, dilation=1)
        self.corr = Correlation(kernel_size=1, max_displacement=3, 
                        stride=1, padding=0, dilation=1)


    def forward(self, x):
        B, N, C = x.shape
        T = int(8) 
        H = int(math.sqrt((N) // T))
        W = H

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2] #[1, self.num_heads, T * H * W, C // self.num_heads]


        q = rearrange(q, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        k = rearrange(k, 'b n (t h w) d -> (b n) t d h w', t=T, h=H, w=W)
        v = rearrange(v, 'b n (t h w) d -> (b n t) (h w) d', t=T, h=H, w=W)

        q_prefix = q.clone()
        k_postfix = torch.zeros_like(q_prefix, device=q_prefix.device)
        k_postfix[:, :-1, ...] = k[:, 1:, ...]
        k_postfix[:, -1, ...] = k[:, -1, ...]

        q_prefix = rearrange(q_prefix, 'b t d h w -> (b t) d h w')
        q_prefix = F.normalize(q_prefix, dim=1)
        k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
        k_postfix = F.normalize(k_postfix, dim=1)

        corr = self.corr(q_prefix.contiguous(), k_postfix.contiguous()) #[B, K, K, H, W]
        corr = rearrange(corr, 'b u v h w -> b (u v) (h w)')

        motion_attn = torch.bmm(corr.transpose(-2, -1), corr) #[B, H*W, H*W]
        motion_attn = motion_attn.softmax(dim=-1)
        motion_x = torch.bmm(motion_attn, v) 
        motion_x = rearrange(motion_x, '(b n t) (h w) d -> b n (t h w) d', b=B, n=self.num_heads, t=T, h=H, w=W)
        motion_x = rearrange(motion_x, 'b h n d -> b n (h d)', h=self.num_heads)

        q_prefix = rearrange(q_prefix, 'b d h w -> b (h w) d')
        k_postfix = rearrange(k_postfix, 'b d h w -> b d (h w)')
        appearance_attn = torch.bmm(q_prefix, k_postfix)
        appearance_attn = appearance_attn.softmax(dim=-1)
        appearance_x = torch.bmm(appearance_attn, v)
        appearance_x = rearrange(appearance_x, '(b n t) (h w) d -> b n (t h w) d', b=B, n=self.num_heads, t=T, h=H, w=W)
        appearance_x = rearrange(appearance_x, 'b h n d -> b n (h d)', h=self.num_heads)

        x = appearance_x +  motion_x

        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x