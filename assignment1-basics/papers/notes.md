# Purpose

This file notes down the important insights/highlights of the papers that are being implemented in modern LLMs.

# Papers
## [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) (Zhang 2019)

### LayerNorm
$$
y_i = \frac{x_i - \mu}{\sigma} \cdot g_i + b_i
$$

$$
\mu = \frac{1}{n} \sum^n_{i=1}x_i \hspace{15pt} 
\sigma = \sqrt{\frac{1}{n} \sum^n_{i=1}(x_i - \mu)^2}
$$

- LayerNorm has invariance on re-centering and re-scaling.

### RMSNorm
$$
y_i = \frac{x_i}{\sqrt{\frac{1}{n} \sum^n_{i=1}x_i^2}} \cdot g_i
$$

- RMSNorm simplifies LayerNorm by removing the mean $\mu$ and bias $b$. This resulted in less computation without sacrificing performance.
- Paper argued that only the re-scaling invariance matters.
- The activations are scaled by Root Mean Square (RMS), which ensures unit scale that benefits stability. 
    - Imagine mapping the activations into $\sqrt{n}$-scaled unit sphere.
    - Robust across vectors of different size, e.g. L2 norm does not work.

## [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) (Xiong 2020)

![alt text](images/prenorm.png)

- PostNorm requires a learning rate warm-up stage.
    - Requires longer training time and hyperparameter tuning.
    - During initialization, the gradient norm is large near the output layer, hence need small learning rate at start. (Refer to the graph below)

![alt text](images/gradient.png)

- On the other hand, PreNorm and DoubleNorm are the main trend nowadays by having well-behaved gradient.
    - One explanation is that they have a clean residual highway for backpropagation without any norm layers in between.

## [RoPE](https://arxiv.org/abs/2104.09864) (Su 2021)


