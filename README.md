#  A Renaissance of Explicit Motion Information Mining from Transformers for Action Recognition
Peiqin Zhuang, Lei Bai, Yichao Wu, Ding Liang, Luping Zhou, Yali Wang, Wanli Ouyang

# Introduction
In this work, we present the Explicit Motion Information Mining (EMIM) module, which seamlessly integrates effective motion modeling into the transformer framework in a unified and elegant manner. Specifically, EMIM constructs a desirable affinity matrix in a cost-volume style, where the set of key candidate tokens is dynamically sampled from query-centered neighboring regions in the subsequent frame using a sliding-window strategy. The resulting affinity matrix is then employed not only to aggregate contextual information for appearance modeling but also to derive explicit motion features for motion representation. By doing so, EMIM preserves the original strength of the transformer in contextual aggregation while endowing it with a new capability for efficient motion modeling. Extensive experiments on four widely used benchmarks demonstrate the superior motion modeling ability of our method, achieving state-of-the-art performance, particularly on motion-sensitive datasets such as Something-Something V1 and V2.



<div align="center">
    <img src="https://github.com/PeiqinZhuang/EMIM/blob/main/figures/property.png" width="80%">
</div>
<div align=center>
    <h3> Property Comparison </h3>
</div>


<div align="center">
    <img src="https://github.com/PeiqinZhuang/EMIM/blob/main/figures/framework.png" width="80%">
</div>
<div align=center>
    <h3> General Framework </h3>
</div>
