# Purpose

This file notes down the important insights/highlights of the papers that are being implemented in modern LLMs.

# Papers
## [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) (Zhang 2019)

### LayerNorm
$$
y_i = \frac{x_i - \mu}{\sigma} \cdot g_i + b_i \\[1em]

\mu = \frac{1}{n} \sum^n_{i=1}x_i \hspace{1em} 
\sigma = \sqrt{\frac{1}{n} \sum^n_{i=1}(x_i - \mu)^2}
$$

- LayerNorm has invariance on re-centering and re-scaling.

### RMSNorm
$$
y_i = \frac{x_i}{\sqrt{\frac{1}{n} \sum^n_{i=1}x_i^2}} \cdot g_i
$$

- RMSNorm simplifies LayerNorm by removing the mean $\mu$ and bias $b$. This resulted in less computation without sacrificing performance.
- Paper argued that only the re-scaling invariance matters.
- The activations are scaled by Root Mean Square (RMS), which ensures the same scale that benefits stability. 
    - Robust across vectors of different size, e.g. L2 norm does not work.

## [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) (Xiong 2020)

![alt text](images/prenorm.png)


## [RoPE](https://arxiv.org/abs/2104.09864) (Su 2021)


