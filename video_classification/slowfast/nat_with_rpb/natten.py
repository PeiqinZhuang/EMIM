import torch
from torch import nn
from timm.models.layers import trunc_normal_
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
from einops import rearrange

try:
    from torch.utils.cpp_extension import load
    nattenav_common_cuda = load(
        'nattenav_cuda', ['common_cuda/nattenav_cuda.cpp', 'common_cuda/nattenav_cuda_kernel.cu'], verbose=False)
    nattenqkrpb_common_cuda = load(
        'nattenqkrpb_cuda', ['common_cuda/nattenqkrpb_cuda.cpp', 'common_cuda/nattenqkrpb_cuda_kernel.cu'], verbose=False)
except:
    try:
        import nattenav_with_rpb
        import nattenqkrpb_with_rpb
    except:
        raise RuntimeError("Could not load NATTEN CUDA extension. " +
                           "Please make sure your device has CUDA, the CUDA toolkit for PyTorch is installed, and that you've compiled NATTEN correctly.")


class NATTENAVFunction_With_RPB(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, attn, value):
        attn = attn.contiguous()
        value = value.contiguous()
        out = nattenav_with_rpb.forward(
                attn, 
                value)[0]
        ctx.save_for_backward(attn, value)
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = nattenav_with_rpb.backward(
            grad_out.contiguous(), *ctx.saved_variables)
        d_attn, d_value = outputs
        return d_attn, d_value


class NATTENQKRPBFunction_With_RPB(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, query, key, rpb):
        query = query.contiguous()
        key = key.contiguous()
        attn = nattenqkrpb_with_rpb.forward(
                query,
                key,
                rpb.contiguous())[0]
        ctx.save_for_backward(query, key)
        return attn

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = nattenqkrpb_with_rpb.backward(
            grad_out.contiguous(), *ctx.saved_variables)
        d_query, d_key, d_rpb = outputs
        return d_query, d_key, d_rpb


class NeighborhoodAttention(nn.Module):
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., attn_type=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        assert self.head_dim == 32 , \
            f"CUDA kernel only supports 32 dim per head, got {self.head_dim}."
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, and 11; got {kernel_size}."

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_type = attn_type

        if self.attn_type is None:
            self.kernel_size = kernel_size
            self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
            trunc_normal_(self.rpb, std=.02)
        elif self.attn_type == 'attn1':
            self.kernel_size = kernel_size
            self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
            trunc_normal_(self.rpb, std=.02)

    def forward(self, x):
        if self.attn_type is None:
            B, H, W, C = x.shape
            T = 16
            qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
            q, k, v = qkv[0], qkv[1], qkv[2] #[B, N, H, W, D]
            q = q * self.scale

            q = rearrange(q, '(b t) n h w d -> b t n h w d', t=T)
            q_prefix = q[:, :-1, ...]
            q_prefix = torch.cat([q_prefix, q_prefix[:, -1:, ...]], dim=1) #[b t n h w d]
            q_prefix = rearrange(q_prefix, 'b t n h w d -> (b t) n h w d')

            v = rearrange(v, '(b t) n h w d -> b t n h w d', t=T)
            v_postfix = v[:, 1:, ...]
            v_postfix = torch.cat([v_postfix, v_postfix[:, -1:, ...]], dim=1)
            v_postfix = rearrange(v_postfix, 'b t n h w d -> (b t) n h w d')

            k = rearrange(k, '(b t) n h w d -> b t n h w d', t=T)
            k_postfix = k[:, 1:, ...]
            k_postfix = torch.cat([k_postfix, k_postfix[:, -1:, ...]], dim=1)
            k_postfix = rearrange(k_postfix, 'b t n h w d -> (b t) n h w d')

            attn = NATTENQKRPBFunction.apply(q_prefix, k_postfix, self.rpb)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = NATTENAVFunction.apply(attn, v_postfix)
            x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)

            return self.proj_drop(self.proj(x))
        elif self.attn_type == 'attn1':
            B, H, W, C = x.shape
            T = 16
            qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
            q, k, v = qkv[0], qkv[1], qkv[2]  # [B, N, H, W, D]
            q = q * self.scale

            q_prefix = q
            v_postfix = v
            k_postfix = k
            attn = NATTENQKRPBFunction.apply(q_prefix, k_postfix, self.rpb)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = NATTENAVFunction.apply(attn, v_postfix)
            x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)

            return self.proj_drop(self.proj(x))

