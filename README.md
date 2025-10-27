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




