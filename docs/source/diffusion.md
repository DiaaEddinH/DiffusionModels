# Diffusion processes

A diffusion model defines a continuous-time perturbation process that incrementally injects noise onto data. The forward process can be written in the form of a stochastic differential equation

$$
	dx_t = f(x_t, t) dt + g(t) dW_t,
$$

where:

- $f(x_t, t)$ is the drift coefficient,
- $g(t)$ is the diffusion coefficient,
- $W_t$ is the Wiener process.

