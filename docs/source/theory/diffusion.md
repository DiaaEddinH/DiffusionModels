# Diffusion processes

A diffusion model defines a continuous-time perturbation process that incrementally injects noise onto data. The forward process can be written in the form of a stochastic differential equation.[^SongSDE]

```{math} 
---
label: forward_sde
---
    dx_t = f(x_t, t) dt + g(t) dW_t 
```

where:

- {math}`f(x_t, t)` is the drift coefficient,
- {math}`g(t)` is the diffusion coefficient,
- {math}`W_t` is the Wiener process.


[^SongSDE]: Score-Based Generative Modeling through Stochastic Differential Equations, [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)