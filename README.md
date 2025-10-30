#  A Renaissance of Explicit Motion Information Mining from Transformers for Action Recognition
Peiqin Zhuang, Lei Bai, Yichao Wu, Ding Liang, Luping Zhou, Yali Wang, Wanli Ouyang

# Introduction
In this work, we present the Explicit Motion Information Mining (EMIM) module, which seamlessly integrates effective motion modeling into the transformer framework in a unified and elegant manner. Specifically, EMIM constructs a desirable affinity matrix in a cost-volume style, where the set of key candidate tokens is dynamically sampled from query-centered neighboring regions in the subsequent frame using a sliding-window strategy. The resulting affinity matrix is then employed not only to aggregate contextual information for appearance modeling but also to derive explicit motion features for motion representation. By doing so, EMIM preserves the original strength of the transformer in contextual aggregation while endowing it with a new capability for efficient motion modeling. Extensive experiments on four widely used benchmarks demonstrate the superior motion modeling ability of our method, achieving state-of-the-art performance, particularly on motion-sensitive datasets such as Something-Something V1 and V2. For more details, please refer to [our paper](https://www.arxiv.org/abs/2510.18705).


## 1. Property Comparison
<div align="center">
    <img src="https://github.com/PeiqinZhuang/EMIM/blob/main/figures/property.png" width="80%">
</div>


## 2. General Framework
<div align="center">
    <img src="https://github.com/PeiqinZhuang/EMIM/blob/main/figures/framework.png" width="80%">
</div>

# How to Use
- Please follow the installation instructions provided in [Uniformer](https://github.com/Sense-X/UniFormer/tree/main/video_classification) for environment preparation.
- Training: Suppose we want to train a small model on Something-Something V1 using the 16-frame setting. Please run the following script:
  
  `bash exp/uniformer_s16_sthv1_pre1k_uniformer_extra_attn64_ATTN_ATTN_7_7_LG/run.sh`
- Test: Suppose we want to evaluate a small model on Something-Something V1 using the 16-frame setting. Please run the following script:
  
  `bash exp/uniformer_s16_sthv1_pre1k_uniformer_extra_attn64_ATTN_ATTN_7_7_LG/test.sh`

# Take Home Message
```python
Input Shape: [B, TxHxW, C]

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    
    def forward(self, x):
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

            return x

```

```python
from nat_with_rpb.natten import NATTENAVFunction_With_RPB as NATTENAVFunction_With_RPB
from timm.models.layers import trunc_normal_


Input Shape: [B, TxHxW, C]
class EMIM(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, max_disp=7):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)


        self.max_disp = max_disp
        self.motion_conv = nn.Sequential(
                nn.Linear(self.max_disp**2, self.max_disp**2 * 4),
                QuickGELU(),
                nn.Linear(self.max_disp**2 * 4, self.head_dim))
        self.padding = nn.ConstantPad2d(self.max_disp // 2, 1e-6)

        self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.max_disp, self.max_disp))
        trunc_normal_(self.rpb, std=0.02)

    def forward(self, x):
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
        k_postfix = k[:, 1:, :, :, :] # We shifted features along the time axis, which serves the purpose of a temporal shift.
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
        
        context_x = NATTENAVFunction_With_RPB.apply(context_attn, v_postfix)
        context_x = rearrange(context_x, '(b t) n h w d -> b (t h w) (n d)', b=B, t=T)

        x = context_x + motion_x

        x = self.proj(x)
        
        
        return x


# 1. As illustrated, our method can directly reuse the parameters from the original Attention module,
#and simply introduces a motion_conv module to equip it with motion modeling capability.
# 2. However, if you would like to replace your original Attention module with our EMIM module, it is
#recommended to perform zero initialization at the beginning to maintain parameter stability — a useful
#tip for parameter-efficient fine-tuning. When training from scratch, this initialization can be omitted.

for name, p in self.named_parameters():
    if 'motion_conv.0.weight' in name:
        nn.init.xavier_uniform_(p)
        print('Re-initialize {}'.format(name))
    if 'motion_conv.2.weight' in name or 'motion_conv.2.bias' in name:
        nn.init.constant_(p, 0.0)
        print('Re-initialize {}'.format(name))

```


# Citing:
Please kindly cite the following paper, if you use EMIM in your work.

```
@article{zhuang2025renaissance,
  title={A Renaissance of Explicit Motion Information Mining from Transformers for Action Recognition},
  author={Zhuang, Peiqin and Bai, Lei and Wu, Yichao and Liang, Ding and Zhou, Luping and Wang, Yali and Ouyang, Wanli},
  journal={arXiv preprint arXiv:2510.18705},
  year={2025}
}
```

# Contact:
Please feel free to contact zpq0316@163.com, if you have any questions about WildFish.

# Acknowledgement:
Some of the codes are borrowed from [Uniformer](https://github.com/Sense-X/UniFormer), [Neighborhood Attention](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer) and [SlowFast](https://github.com/facebookresearch/SlowFast). Many thanks to them.




