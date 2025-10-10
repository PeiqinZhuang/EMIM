from math import ceil, sqrt
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from functools import partial
from timm.models.vision_transformer import _cfg
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
import os
from mmcv.ops import Correlation
from einops import rearrange
import torch.nn.functional as F

# import slowfast.utils.logging as logging
# from slowfast.models.costvolume_attention import CostvolumeAttention
import math

import sys
sys.path.append('/mnt/cache/zhuangpeiqin.vendor/workspace/Transformer/uniformer_3d/video_classification/slowfast')
from nat_with_rpb.natten import NATTENAVFunction_With_RPB as NATTENAVFunction_With_RPB
from nat_with_rpb.natten import NATTENQKRPBFunction_With_RPB as NATTENQKRPBFunction_With_RPB
from nat_without_rpb.natten import NATTENAVFunction_Without_RPB as NATTENAVFunction_Without_RPB
from nat_without_rpb.natten import NATTENQKRPBFunction_Without_RPB as NATTENQKRPBFunction_Without_RPB


# logger = logging.get_logger(__name__)

model_path = 'path_to_models'
model_path = {
    'uniformer_small_in1k': os.path.join(model_path, 'uniformer_small_in1k.pth'),
    'uniformer_small_in1k_dim32': os.path.join(model_path, 'uniformer_small_in1k_dim32.pth'),
    'uniformer_small_k400_8x8': os.path.join(model_path, 'uniformer_small_k400_8x8.pth'),
    'uniformer_small_k400_16x4': os.path.join(model_path, 'uniformer_small_k400_16x4.pth'),
    'uniformer_small_k400_16x4_no_temporal_reduction': os.path.join(model_path, 'uniformer_small_k400_16x4_no_temporal_reduction.pyth'),
    'uniformer_small_k400_16x4_no_temporal_reduction_0875': os.path.join(model_path, 'uniformer_small_k400_16x4_no_temporal_reduction_0875.pyth'),
    'uniformer_small_k400_16x4_no_temporal_reduction2': os.path.join(model_path, 'uniformer_small_k400_16x4_no_temporal_reduction2.pyth'),
    'uniformer_small_k600_16x4': os.path.join(model_path, 'uniformer_small_k600_16x4.pth'),
    'uniformer_base_in1k': os.path.join(model_path, 'uniformer_base_in1k.pth'),
    'uniformer_base_k400_8x8': os.path.join(model_path, 'uniformer_base_k400_8x8.pth'),
    'uniformer_base_k400_16x4': os.path.join(model_path, 'uniformer_base_k400_16x4.pth'),
    'uniformer_base_k400_32x4': os.path.join(model_path, 'uniformer_base_k400_32x4.pth'),
    'uniformer_base_k600_16x4': os.path.join(model_path, 'uniformer_base_k600_16x4.pth'),
    'uniformer_small_k400_mypretrained_lr_8e-5': os.path.join(model_path, 'uniformer_s16x4_k400_uniformer_extra_attn35_ATTN_ATTN_7_7_LG_DROPPATH_01_LR_8e-5.pyth'),
    'uniformer_small_in1k_mypretrained': os.path.join(model_path, 'uniformer_small_in1k_mypretrained.pth'),
    's16x4_k400_uniformer64_01_mypretrained_multinode': os.path.join(model_path, 's16x4_k400_uniformer64_01_mypretrained_multinode.pyth'),
    'uniformer_small_design': os.path.join(model_path, 'uniformer_small_design.pyth'),
    'uniformer_small_k400_16x8': os.path.join(model_path, 'uniformer_small_k400_16x8.pth')
}


def conv_3xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (1, stride, stride), (1, 0, 0), groups=groups)
    
def conv_1xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, 0, 0), groups=groups)

def conv_3xnxn_std(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (1, stride, stride), (1, 0, 0), groups=groups)

def conv_1x1x1(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=groups)

def conv_3x3x3(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (3, 3, 3), (1, 1, 1), (1, 1, 1), groups=groups)

def conv_5x5x5(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (5, 5, 5), (1, 1, 1), (2, 2, 2), groups=groups)

def bn_3d(dim):
    return nn.BatchNorm3d(dim)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., spatial_resolution=None,
                    attn_type=None, drop_path=0., max_disp=7):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_type = attn_type

        print('Attention Type:', self.attn_type, flush=True)
        if self.attn_type is None:
            pass

        elif self.attn_type == 'extra_attn26':
                self.max_disp = max_disp
                self.motion_conv = nn.Sequential(
                        nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                        nn.ReLU(),
                        nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
                self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

                self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
                trunc_normal_(self.rpb, std=0.02)
        

        elif self.attn_type == 'extra_attn35':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn52':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads - (self.num_heads // 2), self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

        elif self.attn_type == 'extra_attn36':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn37':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=False),
                    nn.BatchNorm2d(self.max_disp**2 * 4),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn38':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

            self.layer_norm = nn.LayerNorm(self.max_disp**2)

        elif self.attn_type == 'extra_attn39':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

            self.reweight = Mlp(dim, dim // 2, dim*2)

        

        elif self.attn_type == 'extra_attn40':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)


        elif self.attn_type == 'extra_attn41':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn42':
            self.max_disp = max_disp
            self.motion_conv_1 = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU())
            self.motion_conv_2 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
            
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        

        elif self.attn_type == 'extra_attn33':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            # self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            # trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn34':
            self.max_disp = max_disp
            self.motion_conv_1 = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU())
            self.motion_conv_2 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
            
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            # self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            # trunc_normal_(self.rpb, std=0.02)

        elif self.attn_type == 'extra_attn32':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            # self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            # trunc_normal_(self.rpb, std=0.02)
        

        elif self.attn_type == 'extra_attn31':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            with torch.no_grad():
                nn.init.xavier_uniform_(self.motion_conv[0].weight)
                nn.init.zeros_(self.motion_conv[-1].weight)
                nn.init.zeros_(self.motion_conv[-1].bias)
        
        elif self.attn_type == 'extra_attn30':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn43':
            self.max_disp = max_disp
            self.motion_conv_1 = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            
            self.motion_conv_2 = nn.Linear(dim, dim, bias=qkv_bias)
            self.motion_norm = nn.LayerNorm(dim)

            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn44' or self.attn_type == 'extra_attn45' or self.attn_type == 'extra_attn46':
            self.max_disp = max_disp

            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn48':
            self.max_disp = max_disp
            self.motion_conv_1 = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU())
            self.motion_conv_2 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
            
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

            self.weight = nn.Parameter(torch.Tensor([0.0]))

        elif self.attn_type == 'extra_attn49':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=3, padding=1, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=3, padding=1, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        

        elif self.attn_type == 'extra_attn50':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    Linear(self.max_disp**2, self.max_disp**2 * 4),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

        
        elif self.attn_type == 'extra_attn53':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

        elif self.attn_type == 'extra_attn54':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

            self.weight = nn.Parameter(torch.Tensor([1.0]))
        
        elif self.attn_type == 'extra_attn55':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

            self.weight = nn.Parameter(torch.Tensor([1e-6]))
        
        elif self.attn_type == 'extra_attn56':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

            self.weight = nn.Parameter(torch.Tensor([0.1]))
        
        elif self.attn_type == 'extra_attn57':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

            self.weight = nn.Parameter(torch.Tensor([0.01]))
        
        elif self.attn_type == 'extra_attn58':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn59' or self.attn_type == 'extra_attn62':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

            self.weight = nn.Parameter(torch.Tensor([0.0]))
        
        elif self.attn_type == 'extra_attn60':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

            self.weight = nn.Parameter(torch.Tensor([-1e6]))
        
        elif self.attn_type == 'extra_attn61':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

            self.weight = nn.Parameter(torch.Tensor([-1e6]))
        
        elif self.attn_type == 'extra_attn63':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    QuickGELU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn64':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn64_softmax':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn64_non_sliding':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            # self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            # trunc_normal_(self.rpb, std=0.02)
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.max_disp -1)**2, num_heads)
            )

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.max_disp)
            coords_w = torch.arange(self.max_disp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.max_disp - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.max_disp - 1
            relative_coords[:, :, 0] *= 2 * self.max_disp - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)
        
        elif self.attn_type == 'extra_attn64_non_sliding_bias':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            # self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            # trunc_normal_(self.rpb, std=0.02)
            # self.relative_position_bias_table = nn.Parameter(
            #     torch.zeros((2 * self.max_disp -1)**2, num_heads)
            # )

            # # get pair-wise relative position index for each token inside the window
            # coords_h = torch.arange(self.max_disp)
            # coords_w = torch.arange(self.max_disp)
            # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            # coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            # relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            # relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            # relative_coords[:, :, 0] += self.max_disp - 1  # shift to start from 0
            # relative_coords[:, :, 1] += self.max_disp - 1
            # relative_coords[:, :, 0] *= 2 * self.max_disp - 1
            # relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            # self.register_buffer("relative_position_index", relative_position_index)

            # trunc_normal_(self.relative_position_bias_table, std=.02)


        
        elif self.attn_type == 'extra_attn64_without_bias':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

     
        
        elif self.attn_type == 'extra_attn64_without_motion':
            self.max_disp = max_disp
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn64_skip':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
            
        elif self.attn_type == 'extra_attn64_up_down':
            self.max_disp = max_disp

            self.motion_down = nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=2, stride=2)
            self.motion_norm = nn.LayerNorm(self.max_disp**2 * 4)
            self.motion_up = nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=3, padding=1)

        
        
        
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn64_fb':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn64_fb2':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb_prefix = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb_prefix, std=0.02)

            self.rpb_postfix = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb_postfix, std=0.02)
        
        elif self.attn_type == 'extra_attn64_fb3':
            self.max_disp = max_disp
            self.motion_conv_prefix = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            
            self.motion_conv_postfix = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))

            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb_prefix = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb_prefix, std=0.02)

            self.rpb_postfix = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb_postfix, std=0.02)
        
        elif self.attn_type == 'extra_attn64_fb4':
            self.max_disp = max_disp
            self.motion_conv_prefix = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            
            self.motion_conv_postfix = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))

            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb_prefix = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb_prefix, std=0.02)

            self.rpb_postfix = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb_postfix, std=0.02)
        
        elif self.attn_type == 'extra_attn64_fb5':
            self.max_disp = max_disp
            self.motion_conv_prefix = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            
            self.motion_conv_postfix = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))

            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb_prefix = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb_prefix, std=0.02)

            self.rpb_postfix = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb_postfix, std=0.02)

            self.weight = nn.Parameter(torch.Tensor([0.0]))
        
        elif self.attn_type == 'extra_attn64_identical':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        

        elif self.attn_type == 'extra_attn64_sandwich':
            self.max_disp = max_disp

            self.motion_conv1 = nn.Linear(self.max_disp**2, self.head_dim * 2)
            self.motion_conv2 = nn.Conv2d(self.head_dim * 2, self.head_dim * 2, kernel_size=3, padding=1, groups=self.head_dim*2)
            self.motion_conv3 = nn.Linear(self.head_dim * 2, self.head_dim)
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

        elif self.attn_type == 'extra_attn64_relu':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.head_dim * 4),
                    nn.ReLU(),
                    nn.Linear(self.head_dim * 4, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

        
        elif self.attn_type == 'extra_attn64_double':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 8),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 8, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn64_relu6':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 6),
                    nn.ReLU(),
                    nn.Linear(self.max_disp**2 * 6, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn64_relu_iter':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2),
                    nn.ReLU(),
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 2),
                    nn.ReLU(),
                    nn.Linear(self.max_disp**2 * 2, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

            # self.adapter = nn.Sequential(
            #     nn.Linear(dim, dim // 4),
            #     QuickGELU(),
            #     nn.Linear(dim // 4, dim)
            # )
        
        elif self.attn_type == 'extra_attn64_conv':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2 * self.num_heads, self.max_disp**2 * 4 * self.num_heads, kernel_size=1, stride=1, padding=0, groups=self.num_heads),
                    QuickGELU(),
                    nn.Conv2d(self.max_disp**2 * 4 * self.num_heads, dim, kernel_size=1, stride=1, padding=0))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        

        elif self.attn_type == 'extra_attn64_iter':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 2),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 2, self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn128':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

        elif self.attn_type == 'extra_attn129':
            self.max_disp = max_disp

            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)
            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

        
        
        elif self.attn_type == 'extra_attn65':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    nn.ReLU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn66':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

            self.weight = nn.Parameter(torch.zeros((dim)))
            self.sigmoid = nn.Sigmoid()
        
        elif self.attn_type == 'extra_attn67':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    QuickGELU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

            self.weight = nn.Parameter(torch.zeros((dim)))
            self.sigmoid = nn.Sigmoid()
        
        elif self.attn_type == 'extra_attn68':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn69':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.num_heads * self.max_disp**2, self.num_heads * self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.num_heads * self.max_disp**2 * 4, dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

            self.weight = nn.Parameter(torch.zeros((dim)))
            self.sigmoid = nn.Sigmoid()
        
        elif self.attn_type == 'extra_attn70':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.num_heads * self.max_disp**2, self.num_heads * self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.num_heads * self.max_disp**2 * 4, dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

            self.weight = nn.Parameter(torch.zeros((dim)))
            self.sigmoid = nn.Sigmoid()
        
        elif self.attn_type == 'extra_attn71' or self.attn_type == 'extra_attn72':
            self.max_disp = max_disp
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn73':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Dropout(0.5),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn74':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.num_heads * self.max_disp**2, self.num_heads * self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.num_heads * self.max_disp**2 * 4, dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

           # self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
           # trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn75':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    nn.GELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

        elif self.attn_type == 'extra_attn76':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4, bias=False),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim, bias=False))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn77':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4, bias=False),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn78':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    nn.GELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

            self.norm = nn.LayerNorm(self.max_disp**2)
        
        elif self.attn_type == 'extra_attn80':
            self.max_disp = max_disp
            # self.motion_conv = nn.Sequential(
            #         nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
            #         QuickGELU(),
            #         nn.Linear(self.max_disp**2 * 4, self.head_dim))
            self.motion_conv = conv_3x3x3(self.max_disp**2 * self.num_heads, self.num_heads * self.head_dim, groups=self.num_heads)
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        elif self.attn_type == 'extra_attn81' or self.attn_type == 'extra_attn82':
            self.max_disp = max_disp
            # self.motion_conv = nn.Sequential(
            #         nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
            #         QuickGELU(),
            #         nn.Linear(self.max_disp**2 * 4, self.head_dim))
            self.motion_conv = nn.Sequential(
                    conv_3x3x3(self.max_disp**2 * self.num_heads, self.num_heads * self.max_disp**2 * 4, groups=self.num_heads),
                    QuickGELU(),
                    conv_1x1x1(self.max_disp**2 * self.num_heads * 4, dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

        
        elif self.attn_type == 'extra_attn84':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

        elif self.attn_type == 'extra_attn85':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)

        elif self.attn_type == 'extra_attn86':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.max_disp**2 * 4, self.head_dim))
            self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
            trunc_normal_(self.rpb, std=0.02)
        
        
        

        
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        


    def forward(self, x):
        if self.attn_type is None or self.attn_type == 'None':
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

            attn_original = (q @ k.transpose(-2, -1))
            attn = (q @ k.transpose(-2, -1)) * self.scale
            with torch.cuda.amp.autocast(enabled=False):
                attn = attn.float().softmax(dim=-1)
            attn = self.attn_drop(attn)

            attn_x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(attn_x)
            x = self.proj_drop(x)

  
 
            return x
        
        elif self.attn_type == 'extra_attn26':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t s -> (b n t) d s', b=B)

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1) #[b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)

            k_patches = F.unfold(k_postfix, kernel_size=(self.max_disp, self.max_disp),  stride=1)
            k_patches = rearrange(k_patches, 'b (d p) s -> b d p s', d=self.head_dim, p=self.max_disp ** 2)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)

            v_patches = F.unfold(v_postfix, kernel_size=(self.max_disp, self.max_disp),  stride=1)
            v_patches = rearrange(v_patches, 'b (d p) s -> b d p s', d=self.head_dim, p=self.max_disp ** 2)

            attn = torch.einsum('b d s, b d p s -> b s p', q_prefix, k_patches)
            attn = attn * self.scale
            attn = rearrange(attn, '(b n t) s p -> (b t) n s p', b=B, n=self.num_heads, t=T)
            rpb = self.rpb.reshape(1, self.num_heads, 1, self.max_disp**2)
            attn = attn + rpb
            attn = rearrange(attn,'(b t) n s p -> (b n t) s p', b=B, n=self.num_heads, t=T)

            motion_attn = attn
            motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = torch.einsum('b s p, b d p s -> b s d', context_attn, v_patches)
            context_x = rearrange(context_x, '(b n t) s d -> b (t s) (n d)', b=B, t=T)

            x = context_x + motion_x

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x

        elif self.attn_type == 'extra_attn35' or self.attn_type == 'extra_attn37' or self.attn_type =='extra_attn49':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            print('q shape:', q.shape, flush=True)
            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x
        
        elif self.attn_type == 'extra_attn52':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            num_heads_global = self.num_heads // 2
            num_heads_local = self.num_heads - num_heads_global
            q_global, k_global, v_global = q[:, :num_heads_global, ...], k[:, :num_heads_global, ...], v[:, :num_heads_global, ...]
            q_local, k_local, v_local = q[:, num_heads_global:, ...], k[:, num_heads_global:, ...], v[:, num_heads_global:, ...]

            attn_global = (q_global @ k_global.transpose(-2, -1)) * self.scale
            attn_global = attn_global.softmax(dim=-1)
            attn_global = self.attn_drop(attn_global)

            x_global = (attn_global @ v_global).transpose(1, 2).reshape(B, N, -1)



            q_local = rearrange(q_local, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q_local[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=num_heads_local, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k_local = rearrange(k_local, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k_local[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=num_heads_local)

            v_local = rearrange(v_local, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v_local[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=num_heads_local)


            attn_local = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn_local, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=num_heads_local, h=H, w=W)
            motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=num_heads_local, t=T, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=num_heads_local, t=T, h=H, w=W)


            context_attn = attn_local
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x_local = context_x + motion_x

            x = torch.cat([x_global, x_local], dim=-1)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x
        

        elif self.attn_type == 'extra_attn36':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            residual_q = rearrange(q, '(b n) d t s -> b (t s) (n d)', b=B, n=self.num_heads)
            x = context_x + self.drop_path(motion_x) + residual_q

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x

        
        elif self.attn_type == 'extra_attn38':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_attn = self.layer_norm(motion_attn)
            motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x
      

        elif self.attn_type == 'extra_attn39':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            a = context_x + motion_x
            a = a.mean(dim=1)
            a = self.reweight(a).reshape(B, 1, C, 2).softmax(dim=-1)

            x = a[:, :, :, 0] * context_x + a[:, :, :, 1] * motion_x

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x

        
        elif self.attn_type == 'extra_attn40':
            x = rearrange(x, '(b t) s d -> b (t s) d', t=8)
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))

            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            print('q shape:', q.shape, flush=True)
            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) d h w -> (b t) (h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> (b t) (h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x

        
        elif self.attn_type == 'extra_attn41':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            print('q shape:', q.shape, flush=True)
            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + 0.1 * self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x
        

        elif self.attn_type == 'extra_attn42':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            print('q shape:', q.shape, flush=True)
            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv_1(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) d h w -> (b t) (n d) h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv_2(motion_x)
            motion_x = rearrange(motion_x, '(b t) d h w -> b (t h w) d', b=B, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x 

            x = self.proj(x)
            x = self.proj_drop(x)

            x = x + 0.1 * self.drop_path(motion_x)
            
            return x

        elif self.attn_type == 'extra_attn33' or self.attn_type == 'extra_attn31':
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            attn = (q @ k.transpose(-2, -1)) * self.scale


            pad_attn = rearrange(attn, 'b n (t h w) (i j k) -> (b n t h w) i j k', b=B, n=self.num_heads, t=T, h=H, w=W, i=T, j=H, k=W)
            pad_attn = self.padding(pad_attn)
            pad_attn = rearrange(pad_attn, '(b n t h w) i j k -> b n t h w i (j k)', b=B, n=self.num_heads, t=T,h=H, w=W, i=T)


            new_attn = pad_attn.new_ones(B, self.num_heads, T, H, W, (H + self.max_disp - 1) * (W + self.max_disp - 1))
            for i in range(T-1):
                new_attn[:, :, i, ...] = pad_attn[:, :, i, :, :, i+1, ...]
            new_attn[:, :, -1, ...] = new_attn[:, :, -2, ...]
            new_attn = new_attn.reshape(B, self.num_heads, T, -1)
   
   
            hw_indexs = []
            for h in range(H):
                h_index = torch.range(h, h + self.max_disp - 1) * (H + self.max_disp - 1)
                h_index = h_index[:, None]
                for w in range(W):
                    w_index = torch.range(w, w + self.max_disp - 1)
                    w_index = w_index[None, :]

                    hw_index = h_index + w_index
                    hw_index = hw_index.flatten().long()

                    hw_indexs.append(hw_index)
            hw_indexs = torch.cat(hw_indexs).to(new_attn.device)

            motion = torch.index_select(new_attn, dim=-1, index=hw_indexs)
            motion = motion.reshape(B, self.num_heads, T, H, W, self.max_disp, self.max_disp)



            motion = rearrange(motion, 'b n t h w p q -> (b n t) (p q) h w')
            motion_x = self.motion_conv(motion)
            motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)

            return x
        

        elif self.attn_type == 'extra_attn34':
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            attn = (q @ k.transpose(-2, -1)) * self.scale


            pad_attn = rearrange(attn, 'b n (t h w) (i j k) -> (b n t h w) i j k', b=B, n=self.num_heads, t=T, h=H, w=W, i=T, j=H, k=W)
            pad_attn = self.padding(pad_attn)
            pad_attn = rearrange(pad_attn, '(b n t h w) i j k -> b n t h w i (j k)', b=B, n=self.num_heads, t=T,h=H, w=W, i=T)


            new_attn = pad_attn.new_ones(B, self.num_heads, T, H, W, (H + self.max_disp - 1) * (W + self.max_disp - 1))
            for i in range(T-1):
                new_attn[:, :, i, ...] = pad_attn[:, :, i, :, :, i+1, ...]
            new_attn[:, :, -1, ...] = new_attn[:, :, -2, ...]
            new_attn = new_attn.reshape(B, self.num_heads, T, -1)
   
   
            hw_indexs = []
            for h in range(H):
                h_index = torch.range(h, h + self.max_disp - 1) * (H + self.max_disp - 1)
                h_index = h_index[:, None]
                for w in range(W):
                    w_index = torch.range(w, w + self.max_disp - 1)
                    w_index = w_index[None, :]

                    hw_index = h_index + w_index
                    hw_index = hw_index.flatten().long()

                    hw_indexs.append(hw_index)
            hw_indexs = torch.cat(hw_indexs).to(new_attn.device)

            motion = torch.index_select(new_attn, dim=-1, index=hw_indexs)
            motion = motion.reshape(B, self.num_heads, T, H, W, self.max_disp, self.max_disp)



            motion = rearrange(motion, 'b n t h w p q -> (b n t) (p q) h w')
            motion_x = self.motion_conv_1(motion)
            motion_x = rearrange(motion_x, '(b n t) d h w -> (b t) (n d) h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv_2(motion_x)
            motion_x = rearrange(motion_x, '(b t) d h w -> b (t h w) d', b=B, t=T, h=H, w=W)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

            x = self.proj(x)
            x = self.proj_drop(x)

            x = x + motion_x

            return x

        
        elif self.attn_type == 'extra_attn32':
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            attn = (q @ k.transpose(-2, -1)) * self.scale


            pad_attn = rearrange(attn.detach(), 'b n (t h w) (i j k) -> (b n t h w) i j k', b=B, n=self.num_heads, t=T, h=H, w=W, i=T, j=H, k=W)
            pad_attn = self.padding(pad_attn)
            pad_attn = rearrange(pad_attn, '(b n t h w) i j k -> b n t h w i (j k)', b=B, n=self.num_heads, t=T,h=H, w=W, i=T)


            new_attn = pad_attn.new_ones(B, self.num_heads, T, H, W, (H + self.max_disp - 1) * (W + self.max_disp - 1))
            for i in range(T-1):
                new_attn[:, :, i, ...] = pad_attn[:, :, i, :, :, i+1, ...]
            new_attn[:, :, -1, ...] = new_attn[:, :, -2, ...]
            new_attn = new_attn.reshape(B, self.num_heads, T, -1)
   
   
            hw_indexs = []
            for h in range(H):
                h_index = torch.range(h, h + self.max_disp - 1) * (H + self.max_disp - 1)
                h_index = h_index[:, None]
                for w in range(W):
                    w_index = torch.range(w, w + self.max_disp - 1)
                    w_index = w_index[None, :]

                    hw_index = h_index + w_index
                    hw_index = hw_index.flatten().long()

                    hw_indexs.append(hw_index)
            hw_indexs = torch.cat(hw_indexs).to(new_attn.device)

            motion = torch.index_select(new_attn, dim=-1, index=hw_indexs)
            motion = motion.reshape(B, self.num_heads, T, H, W, self.max_disp, self.max_disp)



            motion = rearrange(motion, 'b n t h w p q -> (b n t) (p q) h w')
            motion_x = self.motion_conv(motion)
            motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)

            return x

        
        elif self.attn_type == 'extra_attn30':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            print('q shape:', q.shape, flush=True)
            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn.detach(), '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x

        
        elif self.attn_type == 'extra_attn43':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv_1(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_norm(motion_x)
            motion_x = self.motion_conv_2(motion_x)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x
        
        elif self.attn_type == 'extra_attn44' or self.attn_type == 'extra_attn45' or self.attn_type == 'extra_attn46':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            print('q shape:', q.shape, flush=True)
            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = attn


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x 

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x, motion_attn
        
        elif self.attn_type == 'extra_attn48':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            print('q shape:', q.shape, flush=True)
            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv_1(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) d h w -> (b t) (n d) h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv_2(motion_x)
            motion_x = rearrange(motion_x, '(b t) d h w -> b (t h w) d', b=B, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x 

            x = self.proj(x)
            x = self.proj_drop(x)

            x = x + self.weight * self.drop_path(motion_x)
            
            return x
        
        elif self.attn_type == 'extra_attn50':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            print('q shape:', q.shape, flush=True)
            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            # motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) s d -> b (t s) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x
        
        elif self.attn_type == 'extra_attn53':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            print('q shape:', q.shape, flush=True)
            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_attn = 10.0 * motion_attn
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x
        
        elif self.attn_type == 'extra_attn54' or self.attn_type == 'extra_attn55' or self.attn_type == 'extra_attn56' or self.attn_type == 'extra_attn57':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            print('q shape:', q.shape, flush=True)
            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_attn = self.weight * motion_attn
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x
        
        elif self.attn_type == 'extra_attn58':
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            attn = (q @ k.transpose(-2, -1)) * self.scale


            pad_attn = rearrange(attn, 'b n (t h w) (i j k) -> (b n t h w) i j k', b=B, n=self.num_heads, t=T, h=H, w=W, i=T, j=H, k=W)
            pad_attn = self.padding(pad_attn)
            pad_attn = rearrange(pad_attn, '(b n t h w) i j k -> b n t h w i (j k)', b=B, n=self.num_heads, t=T,h=H, w=W, i=T)


            new_attn = pad_attn.new_ones(B, self.num_heads, T, H, W, (H + self.max_disp - 1) * (W + self.max_disp - 1))
            for i in range(T-1):
                new_attn[:, :, i, ...] = pad_attn[:, :, i, :, :, i+1, ...]
            new_attn[:, :, -1, ...] = new_attn[:, :, -2, ...]
            new_attn = new_attn.reshape(B, self.num_heads, T, -1)
   
   
            hw_indexs = []
            for h in range(H):
                h_index = torch.range(h, h + self.max_disp - 1) * (H + self.max_disp - 1)
                h_index = h_index[:, None]
                for w in range(W):
                    w_index = torch.range(w, w + self.max_disp - 1)
                    w_index = w_index[None, :]

                    hw_index = h_index + w_index
                    hw_index = hw_index.flatten().long()

                    hw_indexs.append(hw_index)
            hw_indexs = torch.cat(hw_indexs).to(new_attn.device)

            motion = torch.index_select(new_attn, dim=-1, index=hw_indexs)
            motion = motion.reshape(B, self.num_heads, T, H, W, self.max_disp, self.max_disp)

            rpb = self.rpb.reshape(1, self.num_heads, 1, 1, 1, self.max_disp, self.max_disp)
            motion = motion + rpb



            motion = rearrange(motion, 'b n t h w p q -> (b n t) (p q) h w')
            motion_x = self.motion_conv(motion)
            motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)

            return x
        
        elif self.attn_type == 'extra_attn59':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_attn = (F.sigmoid(self.weight) - 0.5) * motion_attn
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x

        
        elif self.attn_type == 'extra_attn60':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_attn = F.sigmoid(self.weight) * motion_attn
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x
        
        elif self.attn_type == 'extra_attn61':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_attn = motion_attn
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + F.sigmoid(self.weight) * self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x

        elif self.attn_type == 'extra_attn62':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_attn =  motion_attn
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + (F.sigmoid(self.weight) - 0.5) * self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x
        
        elif self.attn_type == 'extra_attn63':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            print('q shape:', q.shape, flush=True)
            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        
        elif self.attn_type == 'extra_attn86':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)
            max_values = attn.max(dim=-1, keepdim=True).values
            attn = attn - max_values

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            attn_values = context_attn.max(dim=-1, keepdim=True).values
            context_attn = context_attn - attn_values
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + motion_x + rearrange(v, '(b n) t d h w -> b (t h w) (n d)', b=B, n=self.num_heads)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x
        
        elif self.attn_type == 'extra_attn85':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale



            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            attn_values = context_attn.max(dim=-1, keepdim=True).values
            context_attn = context_attn - attn_values
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x
        
        elif self.attn_type == 'extra_attn64' or self.attn_type == 'extra_attn64_double' or self.attn_type == 'extra_attn64_relu' or self.attn_type == 'extra_attn64_relu6' or self.attn_type == 'extra_attn64_relu_iter' or self.attn_type == 'extra_attn64_iter' or self.attn_type == 'extra_attn65' or self.attn_type == 'extra_attn73' or self.attn_type == 'extra_attn75' or self.attn_type == 'extra_attn76' or self.attn_type == 'extra_attn77':
            B, N, C = x.shape
            T= 16
            H = int(math.sqrt(N//16))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            with torch.cuda.amp.autocast(enabled=False):
                context_attn = F.softmax(context_attn.float(), dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            # x = x + self.adapter(x)
            
            return x
        
        elif self.attn_type == 'extra_attn64_softmax':
            B, N, C = x.shape
            T= 16
            H = int(math.sqrt(N//16))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_attn = F.softmax(motion_attn, dim=-1)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            with torch.cuda.amp.autocast(enabled=False):
                context_attn = F.softmax(context_attn.float(), dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            # x = x + self.adapter(x)
            
            return x
        
        elif self.attn_type == 'extra_attn64_non_sliding':
            B, N, C = x.shape
            T= 16
            H = int(math.sqrt(N//16))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t h w) d -> (b n) d t (h w)', t=T, h=H, w=W)
            q = q.reshape(-1, self.head_dim, T, H // self.max_disp, self.max_disp, W // self.max_disp, self.max_disp)
            q_prefix = q[:, :, :-1, ...]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, ...]], dim=2)
            q_prefix = q_prefix * self.scale
            q_prefix = rearrange(q_prefix, 'b d t i j p q -> (b i p t) (j q) d')
            q_prefix = rearrange(q_prefix, '(b n t) s d -> (b t) n s d', b=B, n=self.num_heads)

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k = k.reshape(-1, T, self.head_dim, H // self.max_disp, self.max_disp, W // self.max_disp, self.max_disp)
            k_postfix = k[:, 1:, ...]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, ...]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d i j p q -> (b i p t) (j q) d')
            k_postfix = rearrange(k_postfix, '(b n t) s d -> (b t) n s d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v = v.reshape(-1, T, self.head_dim, H // self.max_disp, self.max_disp, W // self.max_disp, self.max_disp)
            v_postfix = v[:, 1:, ...]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, ...]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d i j p q -> (b i p t) (j q) d')
            v_postfix = rearrange(v_postfix, '(b n t) s d -> (b t) n s d', b=B, n=self.num_heads)


            attn = (q_prefix @ k_postfix.transpose(-2, -1))
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.max_disp**2, self.max_disp**2, -1) 
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)


            # relative_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            #     self.max_disp**2, self.max_disp**2, -1
            # )

            # attn = attn + relative_bias.unsqueeze(0)



            # motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            # motion_x = self.motion_conv(motion_attn)
            # motion_x = rearrange(motion_x, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_attn = rearrange(attn, 'b n p q -> (b n) p q', n=self.num_heads)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b i p t n) (j q) d -> b (t i j p q) (n d)', b=B, i=H // self.max_disp, p=W // self.max_disp, n=self.num_heads, j=self.max_disp, q=self.max_disp)


            context_attn = attn
            with torch.cuda.amp.autocast(enabled=False):
                context_attn = F.softmax(context_attn.float(), dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            # context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            # context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)
            context_x = context_attn @ v_postfix
            context_x = rearrange(context_x, '(b i p t) n (j q) d -> b (t i j p q) (n d)', b=B, i=H//self.max_disp, p=W//self.max_disp, j=self.max_disp, q=self.max_disp)

            x = context_x + motion_x

            x = self.proj(x)
            x = self.proj_drop(x)
            
            # x = x + self.adapter(x)
            
            return x
        
        elif self.attn_type == 'extra_attn64_non_sliding_bias':
            B, N, C = x.shape
            T= 16
            H = int(math.sqrt(N//16))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t h w) d -> (b n) d t (h w)', t=T, h=H, w=W)
            q = q.reshape(-1, self.head_dim, T, H // self.max_disp, self.max_disp, W // self.max_disp, self.max_disp)
            q_prefix = q[:, :, :-1, ...]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, ...]], dim=2)
            q_prefix = q_prefix * self.scale
            q_prefix = rearrange(q_prefix, 'b d t i j p q -> (b i p t) (j q) d')
            q_prefix = rearrange(q_prefix, '(b n t) s d -> (b t) n s d', b=B, n=self.num_heads)

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k = k.reshape(-1, T, self.head_dim, H // self.max_disp, self.max_disp, W // self.max_disp, self.max_disp)
            k_postfix = k[:, 1:, ...]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, ...]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d i j p q -> (b i p t) (j q) d')
            k_postfix = rearrange(k_postfix, '(b n t) s d -> (b t) n s d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v = v.reshape(-1, T, self.head_dim, H // self.max_disp, self.max_disp, W // self.max_disp, self.max_disp)
            v_postfix = v[:, 1:, ...]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, ...]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d i j p q -> (b i p t) (j q) d')
            v_postfix = rearrange(v_postfix, '(b n t) s d -> (b t) n s d', b=B, n=self.num_heads)


            attn = (q_prefix @ k_postfix.transpose(-2, -1))
            # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            #     self.max_disp**2, self.max_disp**2, -1) 
            # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            # attn = attn + relative_position_bias.unsqueeze(0)


            # relative_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            #     self.max_disp**2, self.max_disp**2, -1
            # )

            # attn = attn + relative_bias.unsqueeze(0)



            # motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            # motion_x = self.motion_conv(motion_attn)
            # motion_x = rearrange(motion_x, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_attn = rearrange(attn, 'b n p q -> (b n) p q', n=self.num_heads)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b i p t n) (j q) d -> b (t i j p q) (n d)', b=B, i=H // self.max_disp, p=W // self.max_disp, n=self.num_heads, j=self.max_disp, q=self.max_disp)


            context_attn = attn
            with torch.cuda.amp.autocast(enabled=False):
                context_attn = F.softmax(context_attn.float(), dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            # context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            # context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)
            context_x = context_attn @ v_postfix
            context_x = rearrange(context_x, '(b i p t) n (j q) d -> b (t i j p q) (n d)', b=B, i=H//self.max_disp, p=W//self.max_disp, j=self.max_disp, q=self.max_disp)

            x = context_x + motion_x

            x = self.proj(x)
            x = self.proj_drop(x)
            
            # x = x + self.adapter(x)
            
            return x
        
        
        elif self.attn_type == 'extra_attn64_without_bias':
            B, N, C = x.shape
            T= 16
            H = int(math.sqrt(N//16))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_Without_RPB.apply(q_prefix, k_postfix)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            with torch.cuda.amp.autocast(enabled=False):
                context_attn = F.softmax(context_attn.float(), dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_Without_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            # x = x + self.adapter(x)
            
            return x
        
        

        elif self.attn_type == 'extra_attn64_without_motion':
            B, N, C = x.shape
            T= 16
            H = int(math.sqrt(N//16))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)



            context_attn = attn
            with torch.cuda.amp.autocast(enabled=False):
                context_attn = F.softmax(context_attn.float(), dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x 

            x = self.proj(x)
            x = self.proj_drop(x)
            
            # x = x + self.adapter(x)
            
            return x
        
        elif self.attn_type == 'extra_attn68':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            # x = x + self.adapter(x)
            
            return x
        
        elif self.attn_type == 'extra_attn64_skip':
            B, N, C = x.shape
            T= 16
            H = int(math.sqrt(N//16))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix_odd = q[:, :, ::2, :]
            q_prefix_even = q[:, :, 1::2, :]
            q_prefix_odd = q_prefix_odd * self.scale
            q_prefix_even = q_prefix_even * self.scale
            q_prefix = torch.cat([q_prefix_odd, q_prefix_even], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix_odd = k[:, 2::2, :, :, :]
            k_postfix_odd = torch.cat([k_postfix_odd, k_postfix_odd[:, -1:, :, :, :]], dim=1)

            k_postfix_even = k[:, 3::2, :, :, :]
            k_postfix_even = torch.cat([k_postfix_even, k_postfix_even[:, -1:, :, :, :]], dim=1)

            k_postfix = torch.cat([k_postfix_odd, k_postfix_even], dim=1)
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            

            v = rearrange(v, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            v_postfix_odd = v[:, 2::2, :, :, :]
            v_postfix_odd = torch.cat([v_postfix_odd, v_postfix_odd[:, -1:, :, :, :]], dim=1)

            v_postfix_even = v[:, 3::2, :, :, :]
            v_postfix_even = torch.cat([v_postfix_even, v_postfix_even[:, -1:, :, :, :]], dim=1)

            v_postfix = torch.cat([v_postfix_odd, v_postfix_even], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


         
           



            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)

            x = rearrange(x, 'b (t h w) d -> b t h w d', t=T, h=H, w=W)
            new_x = torch.zeros_like(x).to(x.device)

            half_len = T // 2
            new_x[:, ::2, ...] = x[:, :half_len, ...]
            new_x[:, 1::2, ...] = x[:, half_len:, ...]

            new_x = rearrange(new_x, 'b t h w d -> b (t h w) d')

            
            return new_x

        elif self.attn_type  == 'extra_attn64_up_down':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) p h w', b=B, t=T, n=self.num_heads, h=H, w=W)

            motion_down = self.motion_down(motion_attn)
            h_down = motion_down.shape[-2]
            w_down = motion_down.shape[-1]
            motion_down = motion_down.flatten(2).transpose(1, 2)

            motion_down = self.motion_norm(motion_down)
            motion_down = motion_down.reshape(-1, h_down, w_down, self.max_disp**2 * 4)
            motion_down = rearrange(motion_down, 'b h w d -> b d h w')

            motion_up = F.interpolate(motion_down, size=H)
            motion_up = self.motion_up(motion_up)

            motion_x = rearrange(motion_up, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            # x = x + self.adapter(x)
            
            return x
        
        elif self.attn_type == 'extra_attn64_fb':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q = rearrange(q, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q = q * self.scale
         

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            k_prefix = k[:, :-1, :, :, :]
            k_prefix = torch.cat([k_prefix[:, 0:1, :, :, :], k_prefix], dim=1)  # [b t d h w]
            k_prefix = rearrange(k_prefix, 'b t d h w -> (b t) d h w')
            k_prefix = self.padding(k_prefix)
            k_prefix = rearrange(k_prefix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            v_prefix = v[:, :-1, :, :, :]
            v_prefix = torch.cat([v_prefix[:, 0:1, :, :, :], v_prefix], dim=1)  # [b t d h w]
            v_prefix = rearrange(v_prefix, 'b t d h w -> (b t) d h w')
            v_prefix = self.padding(v_prefix)
            v_prefix = rearrange(v_prefix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

    


            postfix_attn = NATTENQKRPBFunction_With_RPB.apply(q, k_postfix, self.rpb)

            postfix_motion = rearrange(postfix_attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            postfix_motion = self.motion_conv(postfix_motion)
            postfix_motion = rearrange(postfix_motion, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            postfix_attn = F.softmax(postfix_attn, dim=-1)
            postfix_attn = self.attn_drop(postfix_attn)
            
            postfix_x = NATTENAVFunction_With_RPB.apply(postfix_attn, v_postfix)
            postfix_x = rearrange(postfix_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)


            prefix_attn = NATTENQKRPBFunction_With_RPB.apply(q, k_prefix, self.rpb)

            prefix_motion = rearrange(prefix_attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            prefix_motion = self.motion_conv(prefix_motion)
            prefix_motion = rearrange(prefix_motion, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            prefix_attn = F.softmax(prefix_attn, dim=-1)
            prefix_attn = self.attn_drop(prefix_attn)
            
            prefix_x = NATTENAVFunction_With_RPB.apply(prefix_attn, v_prefix)
            prefix_x = rearrange(prefix_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)
    
            x = postfix_x + postfix_motion + prefix_x + prefix_motion

            x = self.proj(x)
            x = self.proj_drop(x)
            
            
            return x
        
        elif self.attn_type == 'extra_attn64_fb2':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q = rearrange(q, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q = q * self.scale
         

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            k_prefix = k[:, :-1, :, :, :]
            k_prefix = torch.cat([k_prefix[:, 0:1, :, :, :], k_prefix], dim=1)  # [b t d h w]
            k_prefix = rearrange(k_prefix, 'b t d h w -> (b t) d h w')
            k_prefix = self.padding(k_prefix)
            k_prefix = rearrange(k_prefix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            v_prefix = v[:, :-1, :, :, :]
            v_prefix = torch.cat([v_prefix[:, 0:1, :, :, :], v_prefix], dim=1)  # [b t d h w]
            v_prefix = rearrange(v_prefix, 'b t d h w -> (b t) d h w')
            v_prefix = self.padding(v_prefix)
            v_prefix = rearrange(v_prefix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

    


            postfix_attn = NATTENQKRPBFunction_With_RPB.apply(q, k_postfix, self.rpb_postfix)

            postfix_motion = rearrange(postfix_attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            postfix_motion = self.motion_conv(postfix_motion)
            postfix_motion = rearrange(postfix_motion, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            postfix_attn = F.softmax(postfix_attn, dim=-1)
            postfix_attn = self.attn_drop(postfix_attn)
            
            postfix_x = NATTENAVFunction_With_RPB.apply(postfix_attn, v_postfix)
            postfix_x = rearrange(postfix_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)


            prefix_attn = NATTENQKRPBFunction_With_RPB.apply(q, k_prefix, self.rpb_prefix)

            prefix_motion = rearrange(prefix_attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            prefix_motion = self.motion_conv(prefix_motion)
            prefix_motion = rearrange(prefix_motion, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            prefix_attn = F.softmax(prefix_attn, dim=-1)
            prefix_attn = self.attn_drop(prefix_attn)
            
            prefix_x = NATTENAVFunction_With_RPB.apply(prefix_attn, v_prefix)
            prefix_x = rearrange(prefix_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)
    
            x = postfix_x + postfix_motion + prefix_x + prefix_motion

            x = self.proj(x)
            x = self.proj_drop(x)
            
            
            return x
        
        elif self.attn_type == 'extra_attn64_fb3':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q = rearrange(q, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q = q * self.scale
         

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            k_prefix = k[:, :-1, :, :, :]
            k_prefix = torch.cat([k_prefix[:, 0:1, :, :, :], k_prefix], dim=1)  # [b t d h w]
            k_prefix = rearrange(k_prefix, 'b t d h w -> (b t) d h w')
            k_prefix = self.padding(k_prefix)
            k_prefix = rearrange(k_prefix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            v_prefix = v[:, :-1, :, :, :]
            v_prefix = torch.cat([v_prefix[:, 0:1, :, :, :], v_prefix], dim=1)  # [b t d h w]
            v_prefix = rearrange(v_prefix, 'b t d h w -> (b t) d h w')
            v_prefix = self.padding(v_prefix)
            v_prefix = rearrange(v_prefix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

    


            postfix_attn = NATTENQKRPBFunction_With_RPB.apply(q, k_postfix, self.rpb_postfix)

            postfix_motion = rearrange(postfix_attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            postfix_motion = self.motion_conv_postfix(postfix_motion)
            postfix_motion = rearrange(postfix_motion, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            postfix_attn = F.softmax(postfix_attn, dim=-1)
            postfix_attn = self.attn_drop(postfix_attn)
            
            postfix_x = NATTENAVFunction_With_RPB.apply(postfix_attn, v_postfix)
            postfix_x = rearrange(postfix_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)


            prefix_attn = NATTENQKRPBFunction_With_RPB.apply(q, k_prefix, self.rpb_prefix)

            prefix_motion = rearrange(prefix_attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            prefix_motion = self.motion_conv_prefix(prefix_motion)
            prefix_motion = rearrange(prefix_motion, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            prefix_attn = F.softmax(prefix_attn, dim=-1)
            prefix_attn = self.attn_drop(prefix_attn)
            
            prefix_x = NATTENAVFunction_With_RPB.apply(prefix_attn, v_prefix)
            prefix_x = rearrange(prefix_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)
    
            x = postfix_x + postfix_motion + prefix_x + prefix_motion

            x = self.proj(x)
            x = self.proj_drop(x)
            
            
            return x
        
        elif self.attn_type == 'extra_attn64_fb4':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q = rearrange(q, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q = q * self.scale
         

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            k_prefix = k[:, :-1, :, :, :]
            k_prefix = torch.cat([k_prefix[:, 0:1, :, :, :], k_prefix], dim=1)  # [b t d h w]
            k_prefix = rearrange(k_prefix, 'b t d h w -> (b t) d h w')
            k_prefix = self.padding(k_prefix)
            k_prefix = rearrange(k_prefix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            v_prefix = v[:, :-1, :, :, :]
            v_prefix = torch.cat([v_prefix[:, 0:1, :, :, :], v_prefix], dim=1)  # [b t d h w]
            v_prefix = rearrange(v_prefix, 'b t d h w -> (b t) d h w')
            v_prefix = self.padding(v_prefix)
            v_prefix = rearrange(v_prefix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

    


            postfix_attn = NATTENQKRPBFunction_With_RPB.apply(q, k_postfix, self.rpb_postfix)

            postfix_motion = rearrange(postfix_attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            postfix_motion = self.motion_conv_postfix(postfix_motion)
            postfix_motion = rearrange(postfix_motion, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            postfix_attn = F.softmax(postfix_attn, dim=-1)
            postfix_attn = self.attn_drop(postfix_attn)
            
            postfix_x = NATTENAVFunction_With_RPB.apply(postfix_attn, v_postfix)
            postfix_x = rearrange(postfix_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)


            prefix_attn = NATTENQKRPBFunction_With_RPB.apply(q, k_prefix, self.rpb_prefix)

            prefix_motion = rearrange(prefix_attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            prefix_motion = self.motion_conv_prefix(prefix_motion)
            prefix_motion = rearrange(prefix_motion, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            prefix_attn = F.softmax(prefix_attn, dim=-1)
            prefix_attn = self.attn_drop(prefix_attn)
            
            prefix_x = NATTENAVFunction_With_RPB.apply(prefix_attn, v_prefix)
            prefix_x = rearrange(prefix_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)
    
            x = 0.5 * postfix_x + 0.5 * postfix_motion + 0.5 * prefix_x + 0.5 * prefix_motion

            x = self.proj(x)
            x = self.proj_drop(x)
            
            
            return x
        
        elif self.attn_type == 'extra_attn64_fb5':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q = rearrange(q, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q = q * self.scale
         

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            k_prefix = k[:, :-1, :, :, :]
            k_prefix = torch.cat([k_prefix[:, 0:1, :, :, :], k_prefix], dim=1)  # [b t d h w]
            k_prefix = rearrange(k_prefix, 'b t d h w -> (b t) d h w')
            k_prefix = self.padding(k_prefix)
            k_prefix = rearrange(k_prefix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            v_prefix = v[:, :-1, :, :, :]
            v_prefix = torch.cat([v_prefix[:, 0:1, :, :, :], v_prefix], dim=1)  # [b t d h w]
            v_prefix = rearrange(v_prefix, 'b t d h w -> (b t) d h w')
            v_prefix = self.padding(v_prefix)
            v_prefix = rearrange(v_prefix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

    


            postfix_attn = NATTENQKRPBFunction_With_RPB.apply(q, k_postfix, self.rpb_postfix)

            postfix_motion = rearrange(postfix_attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            postfix_motion = self.motion_conv_postfix(postfix_motion)
            postfix_motion = rearrange(postfix_motion, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            postfix_attn = F.softmax(postfix_attn, dim=-1)
            postfix_attn = self.attn_drop(postfix_attn)
            
            postfix_x = NATTENAVFunction_With_RPB.apply(postfix_attn, v_postfix)
            postfix_x = rearrange(postfix_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)


            prefix_attn = NATTENQKRPBFunction_With_RPB.apply(q, k_prefix, self.rpb_prefix)

            prefix_motion = rearrange(prefix_attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            prefix_motion = self.motion_conv_prefix(prefix_motion)
            prefix_motion = rearrange(prefix_motion, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            prefix_attn = F.softmax(prefix_attn, dim=-1)
            prefix_attn = self.attn_drop(prefix_attn)
            
            prefix_x = NATTENAVFunction_With_RPB.apply(prefix_attn, v_prefix)
            prefix_x = rearrange(prefix_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)
    
            weight = F.sigmoid(self.weight)
            x = weight * postfix_x + (1 - weight) * postfix_motion + weight * prefix_x + (1 - weight) * prefix_motion

            x = self.proj(x)
            x = self.proj_drop(x)
            
            
            return x
        
        elif self.attn_type == 'extra_attn64_identical':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]
            avg_qkv =  (qkv[0] + qkv[1] + qkv[2]) / 3.0
            q = avg_qkv
            k = avg_qkv
            v = avg_qkv

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            # x = x + self.adapter(x)
            
            return x
        
        elif self.attn_type == 'extra_attn64_sandwich':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_x = self.motion_conv1(motion_attn)
            motion_x = F.relu(motion_x)
            motion_x = rearrange(motion_x, 'b (h w) p -> b p h w', h=H, w=W)
            motion_x = self.motion_conv2(motion_x)
            motion_x = F.relu(motion_x)
            motion_x = rearrange(motion_x, 'b p h w -> b (h w) p')
            motion_x = self.motion_conv3(motion_x)
            motion_x = F.relu(motion_x)
            motion_x = rearrange(motion_x, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            # x = x + self.adapter(x)
            
            return x

        elif self.attn_type == 'extra_attn64_conv':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b t) (n p) h w', b=B, t=T, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b t) d h w -> b (t h w) d', b=B, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            # x = x + self.adapter(x)
            
            return x


        elif self.attn_type == 'extra_attn84':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            # q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            # k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            # v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T-1, n=self.num_heads, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) (h w) d -> b t (h w) (n d)', b=B, n=self.num_heads, t=T-1, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b t (h w) (n d)', b=B, t=T-1)

            x = context_x + motion_x
            
            new_x = torch.zeros(B, T, H*W, C).cuda()
            new_x[:, :-1, :, :] = x
            new_x = rearrange(new_x, 'b t (h w) d -> b (t h w) d', h=H, w=W)

            x = self.proj(new_x)
            x = self.proj_drop(x)
            
            return x
        
        elif self.attn_type == 'extra_attn66' or self.attn_type == 'extra_attn67':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            print('q shape:', q.shape, flush=True)
            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = (1 - self.sigmoid(self.weight)) * context_x + self.sigmoid(self.weight) * self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x
        
        
           
        
        elif self.attn_type == 'extra_attn69':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            print('q shape:', q.shape, flush=True)
            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> b (t h w) (n p)', b=B, t=T, n=self.num_heads, h=H, w=W)
            # motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            # motion_x = rearrange(motion_x, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)
            # motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)


            x = self.proj(x)
            x = self.proj_drop(x)

            x = (1 - self.sigmoid(self.weight)) * x + self.sigmoid(self.weight) * self.drop_path(motion_x)

            
            return x
        
        elif self.attn_type == 'extra_attn70':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            print('q shape:', q.shape, flush=True)
            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> b (t h w) (n p)', b=B, t=T, n=self.num_heads, h=H, w=W)
            # motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            # motion_x = rearrange(motion_x, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)
            # motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)


            x = self.proj(x)
            x = self.proj_drop(x)

            x =  x + self.sigmoid(self.weight) * self.drop_path(motion_x)

            
            return x
        
        elif self.attn_type == 'extra_attn71' or self.attn_type == 'extra_attn72':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            print('q shape:', q.shape, flush=True)
            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> b (t h w) (n p)', b=B, t=T, n=self.num_heads, h=H, w=W)
            # motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
           # motion_x = self.motion_conv(motion_attn)
            # motion_x = rearrange(motion_x, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)
            # motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)


            x = self.proj(context_x)
            x = self.proj_drop(x)

          #  x =  x + self.sigmoid(self.weight) * self.drop_path(motion_x)

            
            return x, motion_attn
        
        elif self.attn_type == 'extra_attn74':
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            attn = (q @ k.transpose(-2, -1)) * self.scale


            pad_attn = rearrange(attn, 'b n (t h w) (i j k) -> (b n t h w) i j k', b=B, n=self.num_heads, t=T, h=H, w=W, i=T, j=H, k=W)
            pad_attn = self.padding(pad_attn)
            pad_attn = rearrange(pad_attn, '(b n t h w) i j k -> b n t h w i (j k)', b=B, n=self.num_heads, t=T,h=H, w=W, i=T)


            new_attn = pad_attn.new_ones(B, self.num_heads, T, H, W, (H + self.max_disp - 1) * (W + self.max_disp - 1))
            for i in range(T-1):
                new_attn[:, :, i, ...] = pad_attn[:, :, i, :, :, i+1, ...]
            new_attn[:, :, -1, ...] = new_attn[:, :, -2, ...]
            new_attn = new_attn.reshape(B, self.num_heads, T, -1)
   
   
            hw_indexs = []
            for h in range(H):
                h_index = torch.range(h, h + self.max_disp - 1) * (H + self.max_disp - 1)
                h_index = h_index[:, None]
                for w in range(W):
                    w_index = torch.range(w, w + self.max_disp - 1)
                    w_index = w_index[None, :]

                    hw_index = h_index + w_index
                    hw_index = hw_index.flatten().long()

                    hw_indexs.append(hw_index)
            hw_indexs = torch.cat(hw_indexs).to(new_attn.device)

            motion = torch.index_select(new_attn, dim=-1, index=hw_indexs)
            motion = motion.reshape(B, self.num_heads, T, H, W, self.max_disp, self.max_disp)



            motion = rearrange(motion, 'b n t h w p q -> b (t h w) (n p q)')
            motion_x = self.motion_conv(motion)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)

            return x
        

        elif self.attn_type == 'extra_attn34':
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            attn = (q @ k.transpose(-2, -1)) * self.scale


            pad_attn = rearrange(attn, 'b n (t h w) (i j k) -> (b n t h w) i j k', b=B, n=self.num_heads, t=T, h=H, w=W, i=T, j=H, k=W)
            pad_attn = self.padding(pad_attn)
            pad_attn = rearrange(pad_attn, '(b n t h w) i j k -> b n t h w i (j k)', b=B, n=self.num_heads, t=T,h=H, w=W, i=T)


            new_attn = pad_attn.new_ones(B, self.num_heads, T, H, W, (H + self.max_disp - 1) * (W + self.max_disp - 1))
            for i in range(T-1):
                new_attn[:, :, i, ...] = pad_attn[:, :, i, :, :, i+1, ...]
            new_attn[:, :, -1, ...] = new_attn[:, :, -2, ...]
            new_attn = new_attn.reshape(B, self.num_heads, T, -1)
   
   
            hw_indexs = []
            for h in range(H):
                h_index = torch.range(h, h + self.max_disp - 1) * (H + self.max_disp - 1)
                h_index = h_index[:, None]
                for w in range(W):
                    w_index = torch.range(w, w + self.max_disp - 1)
                    w_index = w_index[None, :]

                    hw_index = h_index + w_index
                    hw_index = hw_index.flatten().long()

                    hw_indexs.append(hw_index)
            hw_indexs = torch.cat(hw_indexs).to(new_attn.device)

            motion = torch.index_select(new_attn, dim=-1, index=hw_indexs)
            motion = motion.reshape(B, self.num_heads, T, H, W, self.max_disp, self.max_disp)



            motion = rearrange(motion, 'b n t h w p q -> (b n t) (p q) h w')
            motion_x = self.motion_conv_1(motion)
            motion_x = rearrange(motion_x, '(b n t) d h w -> (b t) (n d) h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv_2(motion_x)
            motion_x = rearrange(motion_x, '(b t) d h w -> b (t h w) d', b=B, t=T, h=H, w=W)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

            x = self.proj(x)
            x = self.proj_drop(x)

            x = x + motion_x

            return x

        
        elif self.attn_type == 'extra_attn78':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            print('q shape:', q.shape, flush=True)
            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_attn = self.norm(motion_attn)
            # motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)
            # motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x
        

        elif self.attn_type == 'extra_attn128':
            B, N, C = x.shape
            T= 32 
            H = int(math.sqrt(N//T))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            # motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
            motion_x = rearrange(motion_x, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)
            # motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            with torch.cuda.amp.autocast(enabled=False):
                context_attn = F.softmax(context_attn.float(), dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x
        
        elif self.attn_type == 'extra_attn80' or self.attn_type == 'extra_attn81':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            # motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_attn = rearrange(attn, '(b t) n h w p -> b (n p) t h w', b=B, t=T, n=self.num_heads, h=H, w=W)
            # motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
          
            motion_x =rearrange(motion_x, 'b d t h w -> b (t h w) d', b=B, t=T, h=H, w=W)
            # motion_x = rearrange(motion_x, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)
            # motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
        
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

           
            x = context_x + self.drop_path(motion_x)

            x = self.proj(x)
            x = self.proj_drop(x)
            
            return x
        
        elif self.attn_type == 'extra_attn82':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            # motion_attn = rearrange(attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
            motion_attn = rearrange(attn, '(b t) n h w p -> b (n p) t h w', b=B, t=T, n=self.num_heads, h=H, w=W)
            # motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
            motion_x = self.motion_conv(motion_attn)
          
            motion_x =rearrange(motion_x, 'b d t h w -> b (t h w) d', b=B, t=T, h=H, w=W)
            # motion_x = rearrange(motion_x, '(b n t) (h w) d -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)
            # motion_x = rearrange(motion_x, '(b n t) d h w -> b (t h w) (n d)', b=B, n=self.num_heads, t=T, h=H, w=W)


            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
        
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x 

            x = self.proj(x)
            x = self.proj_drop(x)

            x = x +  motion_x
            
            return x
        
        elif self.attn_type == 'extra_attn129':
            B, N, C = x.shape
            T= 8
            H = int(math.sqrt(N//8))
            W = H

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B n s d]

            q = rearrange(q, 'b n (t s) d -> (b n) d t s', t=T)
            q_prefix = q[:, :, :-1, :]
            q_prefix = torch.cat([q_prefix, q_prefix[:, :, -1:, :]], dim=2)
            q_prefix = rearrange(q_prefix, '(b n) d t (h w) -> (b t) n h w d', b=B, n=self.num_heads, h=H, w=W)
            q_prefix = q_prefix * self.scale

            k = rearrange(k, 'b n (t h w)  d -> (b n)  t d h w', h=H, w=W)
            k_postfix = k[:, 1:, :, :, :]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, :, :, :]], dim=1)  # [b t d h w]
            k_postfix = rearrange(k_postfix, 'b t d h w -> (b t) d h w')
            k_postfix = self.padding(k_postfix)
            k_postfix = rearrange(k_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)

            
            v = rearrange(v, 'b n (t h w)  d -> (b n) t d h w', h=H, w=W)
            v_postfix = v[:, 1:, :, :, :]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, :, :, :]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t d h w -> (b t) d h w')
            v_postfix = self.padding(v_postfix)
            v_postfix = rearrange(v_postfix, '(b n t) d h w -> (b t) n h w d', b=B, n=self.num_heads)


            attn = NATTENQKRPBFunction_With_RPB.apply(q_prefix, k_postfix, self.rpb)

            context_attn = attn
            context_attn = F.softmax(context_attn, dim=-1)
            context_attn = self.attn_drop(context_attn)
            
            context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
            context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

            x = context_x 

            x = self.proj(x)
            x = self.proj_drop(x)

            return x
 



     
        

  









class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = conv_1x1x1(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = conv_1x1x1(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    
class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = bn_3d(dim)
        self.conv1 = conv_1x1x1(dim, dim, 1)
        self.conv2 = conv_1x1x1(dim, dim, 1)
        self.attn = conv_5x5x5(dim, dim, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = bn_3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x   


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, img_size=0, 
                 attn_type=None, temporal_attn='', attn_drop_path=False, max_disp=7, layer_scale=False):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = norm_layer(dim)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads

        self.attn_type = attn_type
        print('SABlock Attention Type:', self.attn_type, 'Layer_scale:', layer_scale, 'Attn drop_path:', attn_drop_path, flush=True)
   
        if not attn_drop_path:
            self.attn = Attention(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, spatial_resolution=img_size,
                attn_type=attn_type, max_disp=max_disp)
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, spatial_resolution=img_size,
                attn_type=attn_type, drop_path=drop_path)
   
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.ls = layer_scale

        if self.ls:
            init_value = 1.0
            self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)
        
        if attn_type == 'extra_attn44' or attn_type =='extra_attn45' or attn_type == 'extra_attn46':
            self.max_disp = 7
            self.motion_conv_1 = nn.Sequential(
                    nn.Conv2d(self.max_disp**2, self.max_disp**2 * 4, kernel_size=1, padding=0, stride=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.max_disp**2 * 4, self.head_dim, kernel_size=1, padding=0, stride=1, bias=True))
            self.motion_conv_2 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)

        elif self.attn_type == 'extra_attn71':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.num_heads * self.max_disp**2, self.num_heads * self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.num_heads * self.max_disp**2 * 4, dim))

            self.norm3 = nn.LayerNorm(dim)
        
        elif self.attn_type == 'extra_attn72':
            self.max_disp = max_disp
            self.motion_conv = nn.Sequential(
                    nn.Linear(self.num_heads * self.max_disp**2, self.num_heads * self.max_disp**2 * 4),
                    QuickGELU(),
                    nn.Linear(self.num_heads * self.max_disp**2 * 4, dim))




    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        if not self.ls:
            if self.attn_type == 'extra_attn44':
                attn_output = self.attn(self.norm1(x))
                attn_x, motion_attn = attn_output[0], attn_output[1]

                motion_attn = rearrange(motion_attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
                motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
                motion_x = self.motion_conv_1(motion_attn)
                motion_x = rearrange(motion_x, '(b n t) d h w -> (b t) (n d) h w', b=B, n=self.num_heads, t=T, h=H, w=W)
                motion_x = self.motion_conv_2(motion_x)
                motion_x = rearrange(motion_x, '(b t) d h w -> b (t h w) d', b=B, t=T, h=H, w=W)

                x = x + self.drop_path(attn_x) + self.drop_path(motion_x)
                x = x + self.drop_path(self.mlp(self.norm2(x)))

            elif self.attn_type == 'extra_attn45':
                attn_output = self.attn(self.norm1(x))
                attn_x, motion_attn = attn_output[0], attn_output[1]

                motion_attn = rearrange(motion_attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
                motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
                motion_x = self.motion_conv_1(motion_attn)
                motion_x = rearrange(motion_x, '(b n t) d h w -> (b t) (n d) h w', b=B, n=self.num_heads, t=T, h=H, w=W)
                motion_x = self.motion_conv_2(motion_x)
                motion_x = rearrange(motion_x, '(b t) d h w -> b (t h w) d', b=B, t=T, h=H, w=W)

                x = x + self.drop_path(attn_x)
                x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(motion_x)

            elif self.attn_type == 'extra_attn46':
                attn_output = self.attn(self.norm1(x))
                attn_x, motion_attn = attn_output[0], attn_output[1]

                motion_attn = rearrange(motion_attn, '(b t) n h w p -> (b n t) (h w) p', b=B, t=T, n=self.num_heads, h=H, w=W)
                motion_attn = rearrange(motion_attn, '(b n t) (h w) p -> (b n t) p h w', b=B, n=self.num_heads, t=T, h=H, w=W)
                motion_x = self.motion_conv_1(motion_attn)
                motion_x = rearrange(motion_x, '(b n t) d h w -> (b t) (n d) h w', b=B, n=self.num_heads, t=T, h=H, w=W)
                motion_x = self.motion_conv_2(motion_x)
                motion_x = rearrange(motion_x, '(b t) d h w -> b (t h w) d', b=B, t=T, h=H, w=W)

                x = x + self.drop_path(attn_x)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                x = x + self.drop_path(motion_x)
            elif self.attn_type == 'extra_attn71':
                attn_output = self.attn(self.norm1(x))

                attn_x, motion_x = attn_output[0], attn_output[1]
                x = x + self.drop_path(attn_x)
                motion_x = self.motion_conv(motion_x)
                x = x + self.drop_path(self.mlp(self.norm2(x) + self.norm3(motion_x)))
            elif self.attn_type == 'extra_attn72':
                attn_output  = self.attn(self.norm1(x))

                attn_x, motion_x = attn_output[0], attn_output[1]
                x = x + self.drop_path(attn_x)
                motion_x = self.motion_conv(motion_x)
                x = x + self.drop_path(self.mlp(self.norm2(x)) + motion_x)
                
            else:
                with torch.cuda.amp.autocast(dtype=torch.float32):
                    x = x + self.drop_path(self.attn(self.norm1(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        return x    


class SplitSABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, img_size=0, 
                 attn_type=None, temporal_attn='', attn_drop_path=False, max_disp=7, layer_scale=False):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.t_norm = norm_layer(dim)
        self.t_attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop,
            spatial_resolution=img_size,
            attn_type=attn_type, max_disp=max_disp)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        attn = x.view(B, C, T, H * W).permute(0, 3, 2, 1).contiguous()
        attn = attn.view(B * H * W, T, C)
        attn = attn + self.drop_path(self.t_attn(self.t_norm(attn)))
        attn = attn.view(B, H * W, T, C).permute(0, 2, 1, 3).contiguous()
        attn = attn.view(B * T, H * W, C)
        residual = x.view(B, C, T, H * W).permute(0, 2, 3, 1).contiguous()
        residual = residual.view(B * T, H * W, C)
        attn = residual + self.drop_path(self.attn(self.norm1(attn)))
        attn = attn.view(B, T * H * W, C)
        out = attn + self.drop_path(self.mlp(self.norm2(attn)))
        out = out.transpose(1, 2).reshape(B, C, T, H, W)
        return out


class SpeicalPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = conv_3xnxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])

    def forward(self, x):
        x = x[None, :, :, :, :]
        print('Input Image Shape:', x.shape, flush=True)
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x
    

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, std=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        if std:
            self.proj = conv_3xnxn_std(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])
        else:
            self.proj = conv_1xnxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])

    def forward(self, x):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x



class Uniformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self):
        super().__init__()

        depth = [3,4,8,3]
        num_classes = 174
        img_size = 224
        in_chans = 3
        embed_dim = [64, 128, 320, 512]
        head_dim = 64
        mlp_ratio = 4
        qkv_bias = True
        qk_scale = None
        representation_size = None
        drop_rate = 0
        attn_drop_rate = 0.0
        drop_path_rate = 0.2
        split = False
        std = False
        # uniformer_attn_type = ['extra_attn64', 'extra_attn64']\
        uniformer_attn_type = [None, None]
        self.use_checkpoint = False
        self.checkpoint_num = [0,0,0,0]
        attn_type = [None, None]
        # attn_type = ['extra_attn64', 'extra_attn64']

        attn_drop_path = False
        temporal_attn =''
        max_disp =[7, 7]
        consensus = False
        # continuous_attn ='lg'
        continuous_attn ='gg'
        layer_scale = False
        special_init = False
        print('Continuous Attention Type:', continuous_attn, flush=True)
        self.consensus = consensus

        # logger.info(f'Use checkpoint: {self.use_checkpoint}')
        # logger.info(f'Checkpoint number: {self.checkpoint_num}')

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = partial(nn.LayerNorm, eps=1e-6) 
        
        self.patch_embed1 = SpeicalPatchEmbed(
            img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1], std=std)
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2], std=std)
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3], std=std)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]], norm_layer=norm_layer)
            for i in range(depth[1])])
        if split:
            if 'gg' in continuous_attn or 'lg4' == continuous_attn:
                self.blocks3 = nn.ModuleList([
                    SplitSABlock(
                        dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer)
                    for i in range(depth[2])])
            
            if 'gg' in continuous_attn or 'lg3' == continuous_attn:
                self.blocks4 = nn.ModuleList([
                    SplitSABlock(
                        dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer)
                for i in range(depth[3])])
            
            else:
                blocks3 = []

                for i in range(depth[2]):
                    if 'lg' == continuous_attn or 'lg2' == continuous_attn or 'lg3' == continuous_attn:
                        if i % 2 == 0:
                            blocks3.append(SplitSABlock(
                                dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                                attn_type=attn_type[0], temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                        else:
                            blocks3.append(SplitSABlock(
                                dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                                attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
              
                self.blocks3 = nn.ModuleList(blocks3)

                blocks4 = []

                for i in range(depth[3]):
                    if 'lg' == continuous_attn or 'lg4' == context_attn:
                        if i % 2 == 0:
                            blocks4.append(SplitSABlock(
                                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                                attn_type=attn_type[1], temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))
                        else:
                            blocks4.append(SplitSABlock(
                                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                                attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))
                    elif 'lg2' == continuous_attn:
                        blocks4.append(
                            SplitSABlock(
                                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                                attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale),
                            SplitSABlock(
                                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                                attn_type=attn_type[1], temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale),
                            SplitSABlock(
                                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                                attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale)
                        )

                self.blocks4 = nn.ModuleList(blocks4)


        else:
            blocks3 = []
            blocks4 = []
            if 'gg' in continuous_attn or 'lg4' == continuous_attn:
                blocks3 = [
                    SABlock(
                        dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                        attn_type=attn_type[0], temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale)
                    for i in range(depth[2])]
            elif 'lg' == continuous_attn or 'lg3' == continuous_attn:
                for i in range(depth[2]):
                    if i % 2 == 0:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=attn_type[0], temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                    else:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
            elif 'gl' == continuous_attn:
                for i in range(depth[2]):
                    if i % 2 == 1:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=attn_type[0], temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                    else:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
            elif 'lll' == continuous_attn:
                for i in range(depth[2]):
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=attn_type[0], temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
            elif 'lgg' == continuous_attn:
                for i in range(depth[2]):
                    if i % 3 == 0:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=attn_type[0], temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                    elif i % 3 == 1:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                    elif i % 3 == 2:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
            elif 'glg' == continuous_attn:
                for i in range(depth[2]):
                    if i % 3 == 1:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=attn_type[0], temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                    elif i % 3 == 0:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                    elif i % 3 == 2:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
            elif 'lg_new' == continuous_attn:
                for i in range(depth[2]-2):
                    if i % 2 == 0:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=attn_type[0], temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                    else:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))

                for i in range(depth[2]-2, depth[2]):
                    blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
            
            elif 'lg_new2' == continuous_attn:
                for i in range(depth[2]-4):
                    if i % 2 == 0:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=attn_type[0], temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                    else:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))

                for i in range(depth[2]-4, depth[2]):
                    blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))

            elif 'sandwich'== continuous_attn:
                for i in range(depth[2] - 2):
                    if i % 3 == 0:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                    elif i % 3  == 1:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=attn_type[0], temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                    elif i % 3 == 2:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))

                blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
            elif 'sandwich1' == continuous_attn:
                for i in range(depth[2]):
                    if i % 4 == 0:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=attn_type[0], temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                    elif i % 4 == 1:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                    elif i % 4 == 2:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=attn_type[0], temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                    elif i % 4 == 3:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
            
            elif 'sandwich2' == continuous_attn:
                for i in range(depth[2]):
                    if i % 4 == 0:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                    elif i % 4 == 1:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=attn_type[0], temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                    elif i % 4 == 2:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                    elif i % 4 == 3:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=attn_type[0], temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
            
            elif 'local_last' == continuous_attn:
                for i in range(depth[2]):
                    if i % 2 == 0:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                    else:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=attn_type[0], temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))

            elif 'lg5' == continuous_attn:
                for i in range(depth[2]):
                    if i % 2 == 0:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]] * 0.5, norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=attn_type[0], temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                    else:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale)) 

            elif 'lg6' == continuous_attn:
                for i in range(depth[2]):
                    if i % 2 == 0:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]] * 2.0, norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=attn_type[0], temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                    else:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
            
            elif 'lg7' == continuous_attn:
                for i in range(depth[2]):
                    if i % 2 == 0:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]] * 0.2, norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=attn_type[0], temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                    else:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))

            elif 'lg8' == continuous_attn:
                for i in range(depth[2]):
                    if i == 0 or i == (depth[2] - 1):
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                    else:
                        if i % 2 == 0:
                            blocks3.append(SABlock(
                                dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]] * 0.2, norm_layer=norm_layer, img_size=img_size // 16,
                                attn_type=attn_type[0], temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                        else:
                            blocks3.append(SABlock(
                                dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                                attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
            elif 'lg9' == continuous_attn:
                for i in range(depth[2]):
                    if i == 0 or i == (depth[2] - 1):
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))
                    else:
                        blocks3.append(SABlock(
                            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]] * 0.2, norm_layer=norm_layer, img_size=img_size // 16,
                            attn_type=attn_type[0], temporal_attn=temporal_attn, attn_drop_path=attn_drop_path, max_disp=max_disp[0], layer_scale=layer_scale))




            if 'gg' in continuous_attn or 'lg3' == continuous_attn:
                blocks4 = nn.ModuleList([
                    SABlock(
                        dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                        attn_type=attn_type[1], temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale)
                for i in range(depth[3])])
            elif 'lg' == continuous_attn or 'lg4' == continuous_attn:
                for i in range(depth[3]):
                    if i % 2 == 0:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=attn_type[1], temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))
                    else:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))   
            elif 'gl' == continuous_attn:
                for i in range(depth[3]):
                    if i % 2 == 1:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=attn_type[1], temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))
                    else:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale)) 
            elif 'lll' == continuous_attn:
                for i in range(depth[3]):
                    blocks4.append(SABlock(
                        dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                        attn_type=attn_type[1], temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))
            elif 'lgg' == continuous_attn:
                for i in range(depth[3]):
                    if i % 3 == 0:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=attn_type[1], temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))
                    elif i % 3 == 1:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))  
                    elif i % 3 == 2:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))  
            elif 'glg' == continuous_attn:
                for i in range(depth[3]):
                    if i % 3 == 1:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=attn_type[1], temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))
                    elif i % 3 == 0:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))  
                    elif i % 3 == 2:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))  
                    
            elif 'lg_new' == continuous_attn:
                for i in range(depth[3]-2):
                    if i % 2 == 0:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=attn_type[1], temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))
                    else:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))   

                for i in range(depth[3]-2, depth[3]):
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))   
            
            elif 'lg_new2' == continuous_attn:
                for i in range(depth[3]-4):
                    if i % 2 == 0:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=attn_type[1], temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))
                    else:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))   

                for i in range(depth[3]-4, depth[3]):
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))   


            elif 'sandwich' ==  continuous_attn:
                for i in range(depth[3]):
                    if i % 3 == 0:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))    
                    elif i % 3 == 1:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=attn_type[1], temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))
                    elif i % 3 == 2:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))    
            
            elif 'sandwich1' == continuous_attn:
                for i in range(depth[3]):
                    if i % 3 == 0:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=attn_type[1], temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))
                    elif i % 3 ==1:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))
                    elif i % 3 == 2:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=attn_type[1], temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))
            
            elif 'sandwich2' == continuous_attn:
                for i in range(depth[3]):
                    if i % 3 == 0:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))
                    elif i % 3 ==1:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=attn_type[1], temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))
                    elif i % 3 == 2:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))

            elif 'local_last' == continuous_attn:
                for i in range(depth[3]):
                    if i % 2 == 0:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=attn_type[1], temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))
                    else:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))

            elif 'lg5' == continuous_attn:
                for i in range(depth[3]):
                    if i % 2 == 0:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]] * 0.5, norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=attn_type[1], temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))
                    else:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))  


            elif 'lg6' == continuous_attn:
                for i in range(depth[3]):
                    if i % 2 == 0:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]] *  2.0, norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=attn_type[1], temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))
                    else:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))  

            elif 'lg7' == continuous_attn:
                for i in range(depth[3]):
                    if i % 2 == 0:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]] *  0.2, norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=attn_type[1], temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))
                    else:
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale)) 

            elif 'lg8' == continuous_attn or 'lg9' == continuous_attn:
                for i in range(depth[3]):
                    if i == 0 or i == (depth[3] - 1):
                        blocks4.append(SABlock(
                            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                            attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))
                    else: 
                        if i % 2 == 0:
                            blocks4.append(SABlock(
                                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]] *  0.2, norm_layer=norm_layer, img_size=img_size // 32,
                                attn_type=attn_type[1], temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))
                        else:
                            blocks4.append(SABlock(
                                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer, img_size=img_size // 32,
                                attn_type=None, temporal_attn=temporal_attn, attn_drop_path=False, max_disp=max_disp[1], layer_scale=layer_scale))   


            self.blocks3 = nn.ModuleList(blocks3)
            self.blocks4 = nn.ModuleList(blocks4)



        self.norm = bn_3d(embed_dim[-1])

        self.extra_block = ''
        self.extra_block_idx = [-1,-1,-1]
        self.extra_block_depth = 1
        self.extra_attn_type = 'extra_attn1'
        self.extra_temporal_attn = None
        self.extra_before = True
        self.extra_max_disp =[7,7,7]

        if self.extra_block == '':
            self.extra_module_2 = nn.Identity()
            self.extra_module_3 = nn.Identity()
            self.extra_module_4 = nn.Identity()
        else:
            for index, extra_block_idx in enumerate(self.extra_block_idx):
                extra_module = [
                        SABlock(
                        dim=embed_dim[extra_block_idx-1],
                        num_heads=num_heads[extra_block_idx-1],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depth[:extra_block_idx-1])+1],
                        norm_layer=norm_layer,
                        img_size = 224 // (2**extra_block_idx),
                        attn_type=self.extra_attn_type,
                        temporal_attn=self.extra_temporal_attn,
                        max_disp=self.extra_max_disp[index]
                        )
                        for i in range(self.extra_block_depth)]

                if index == 0:
                    if extra_block_idx == -1:
                        self.extra_module_2 = nn.Identity()
                    else:
                        self.extra_module_2 = nn.Sequential(*extra_module)
                elif index == 1:
                    if extra_block_idx == -1:
                        self.extra_module_3 = nn.Identity()
                    else:
                        self.extra_module_3 = nn.Sequential(*extra_module)
                elif index == 2:
                    if extra_block_idx == -1:
                        self.extra_module_4 = nn.Identity()
                    else:
                        self.extra_module_4 = nn.Sequential(*extra_module)

        print('extra_module_2:', self.extra_module_2, flush=True)
        print('extra_module_3:', self.extra_module_3, flush=True)
        print('extra_module_4:', self.extra_module_4, flush=True)
        
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
        
        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            if 't_attn.qkv.weight' in name:
                nn.init.constant_(p, 0)
            if 't_attn.qkv.bias' in name:
                nn.init.constant_(p, 0)
            if 't_attn.proj.weight' in name:
                nn.init.constant_(p, 1)
            if 't_attn.proj.bias' in name:
                nn.init.constant_(p, 0)

        for name, p in self.named_parameters():
            if 'attn.motion_conv.0.weight' in name:
                nn.init.xavier_uniform_(p)
                print('Re-initialize {}'.format(name))
            if 'attn.motion_conv.2.weight' in name or 'attn.motion_conv.2.bias' in name:
                nn.init.constant_(p, 0.0)
                print('Re-initialize {}'.format(name))
        
        for name, p in self.named_parameters():
            if 'motion_conv.0.weight' in name:
                nn.init.xavier_uniform_(p)
                print('Re-initialize {}'.format(name))
            if 'motion_conv.2.weight' in name or 'motion_conv.2.bias' in name:
                nn.init.constant_(p, 0.0)
                print('Re-initialize {}'.format(name))

        for name, p in self.named_parameters():
            if 'adapter.0.weight' in name:
                nn.init.xavier_uniform_(p)
                print('Re-initialize {}'.format(name))
            if 'adapter.2.weight' in name or 'adapter.2.bias' in name:
                nn.init.constant_(p, 0.0)
                print('Re-initialize {}'.format(name))

        if attn_type[0] == 'extra_attn80':
            for name, p in self.named_parameters():
                if 'motion_conv.0.weight' in name or 'motion_conv.0.bias' in name:
                    nn.init.constant_(p, 0.0)
                    print('Re-initialize {} in extra_attn80'.format(name))
        
        if attn_type[0] == 'extra_attn64_fb3' or attn_type[0] == 'extra_attn64_fb4' or attn_type[0] == 'extra_attn64_fb5':
            for name, p in self.named_parameters():
                if 'attn.motion_conv_postfix.0.weight' in name:
                    nn.init.xavier_uniform_(p)
                    print('Re-initialize {}'.format(name))
                if 'attn.motion_conv_postfix.2.weight' in name or 'attn.motion_conv_postfix.2.bias' in name:
                    nn.init.constant_(p, 0.0)
                    print('Re-initialize {}'.format(name))

                if 'attn.motion_conv_prefix.0.weight' in name:
                    nn.init.xavier_uniform_(p)
                    print('Re-initialize {}'.format(name))
                if 'attn.motion_conv_prefix.2.weight' in name or 'attn.motion_conv_prefix.2.bias' in name:
                    nn.init.constant_(p, 0.0)
                    print('Re-initialize {}'.format(name))

        if special_init:
            for name, p in self.named_parameters():
                if 'motion_conv.0.wieght' in name or 'motion_conv.2.weight' in name:
                    nn.init.constant_(p, 1)
                
                if 'motion_conv.0.bias' in name or 'motion_conv.0.bias' in name:
                    nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):

        return {'pos_embed', 'cls_token', 'rpb', 'geometric_rpb', 'relative_rpb', 'rpb_postfix', 'rpb_prefix'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def inflate_weight(self, weight_2d, time_dim, center=False):
        if center:
            weight_3d = torch.zeros(*weight_2d.shape)
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            middle_idx = time_dim // 2
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        else:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim
        return weight_3d

    def get_pretrained_model(self, cfg):
        if cfg.UNIFORMER.PRETRAIN_NAME:
            checkpoint = torch.load(model_path[cfg.UNIFORMER.PRETRAIN_NAME], map_location='cpu')

            state_dict_3d = self.state_dict()
            for k in checkpoint.keys():
                if checkpoint[k].shape != state_dict_3d[k].shape:
                    if len(state_dict_3d[k].shape) <= 2:
                        # logger.info(f'Ignore: {k}')
                        continue
                    # logger.info(f'Inflate: {k}, {checkpoint[k].shape} => {state_dict_3d[k].shape}')
                    time_dim = state_dict_3d[k].shape[2]
                    checkpoint[k] = self.inflate_weight(checkpoint[k], time_dim)

            if self.num_classes != checkpoint['head.weight'].shape[0]:
                del checkpoint['head.weight'] 
                del checkpoint['head.bias'] 
            return checkpoint
        else:
            return None
            
    def forward_features(self, x):
        import ipdb; ipdb.set_trace()
        x = self.patch_embed1(x)
        x = self.pos_drop(x)

        y = []
        for i, blk in enumerate(self.blocks1):
            if self.use_checkpoint and i < self.checkpoint_num[0]:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            y.append(x)
        x = self.patch_embed2(x)


        if 2 in self.extra_block_idx and self.extra_before:
            if self.extra_block == 'parallel':
                x =  x + self.extra_module_2(x)
            elif self.extra_block == 'sequential':
                x = self.extra_module_2(x)
        else:
            x = self.extra_module_2(x)

        if self.extra_block != '' and self.extra_block_idx == 2 and self.extra_before_patch:
            if self.extra_block == 'parallel':
                x = x + self.extra_module(x)
            elif self.extra_block == 'sequential':
                x = self.extra_module(x)
        else:
            pass

        for i, blk in enumerate(self.blocks2):
            if self.use_checkpoint and i < self.checkpoint_num[1]:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            y.append(x)

        
        if  2 in self.extra_block_idx  and (not self.extra_before):
            if self.extra_block == 'parallel':
                x =  x + self.extra_module_2(x)
            elif self.extra_block == 'sequential':
                x = self.extra_module_2(x)
        else:
            x = self.extra_module_2(x)

        x = self.patch_embed3(x)


        if 3 in self.extra_block_idx and self.extra_before:
            if self.extra_block == 'parallel':
                x = x + self.extra_module_3(x)
            else:
                x = self.extra_module_3(x)
        else:
            x = self.extra_module_3(x)

        if self.extra_block != '' and self.extra_block_idx == 3 and self.extra_before_patch:
            if self.extra_block == 'parallel':
                x = x + self.extra_module(x)
            elif self.extra_block == 'sequential':
                x = self.extra_module(x)
        else:
            pass

        for i, blk in enumerate(self.blocks3):
            if self.use_checkpoint and i < self.checkpoint_num[2]:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            y.append(x)

        if 3 in self.extra_block_idx  and (not self.extra_before):
            if self.extra_block == 'parallel':
                x = x + self.extra_module_3(x)
            else:
                x = self.extra_module_3(x)
        else:
            x = self.extra_module_3(x)

        x = self.patch_embed4(x)

        
        if  4 in self.extra_block_idx and self.extra_before:
            if self.extra_block == 'parallel':
                x = x + self.extra_module_4(x)
            else:
                x = self.extra_module_4(x)
        else:
            x = self.extra_module_4(x)

        for i, blk in enumerate(self.blocks4):
            if self.use_checkpoint and i < self.checkpoint_num[3]:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            y.append(x)

        if  4 in self.extra_block_idx and (not self.extra_before):
            if self.extra_block == 'parallel':
                x = x + self.extra_module_4(x)
            else:
                x = self.extra_module_4(x)
        else:
            x = self.extra_module_4(x)

   

        x = self.norm(x)
        x = self.pre_logits(x)
        return x, y

    def forward(self, x):
        x = x[0]
        x, y = self.forward_features(x)
        x = x.flatten(2).mean(-1)
        x = self.head(x)
        return x, y
