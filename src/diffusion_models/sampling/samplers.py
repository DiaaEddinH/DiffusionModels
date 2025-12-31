import torch
from tqdm import tqdm


def em_sampler(
    model: torch.nn.Module,
    shape: tuple,
    num_steps: int,
    *labels,
    history: bool = False,
    eps=1e-3,
):
    """EULER-MARUYAMA stochastic sampler ~ √dt"""
    output = []
    batch_size = shape[0]
    device = model.device

    timesteps = torch.linspace(1, eps, num_steps, device=device)

    g_t = model.schedule.diffusion_coeff(timesteps)
    step_size = 1 / num_steps
    step_size_sqrt = step_size**0.5

    t0 = torch.ones(1, device=device)
    std = model.schedule.stddev(t0)
    x = torch.randn(*shape, device=device) * std

    model.eval()

    for i, t_i in enumerate(tqdm(timesteps)):
        batch_t = t_i.expand(batch_size)

        drift = g_t[i] ** 2 * model(x, batch_t, *labels)
        noise = g_t[i] * torch.randn_like(x) if t_i > eps else 0.0

        x = x + drift * step_size + step_size_sqrt * noise

        if history:
            output.append(x)
    if history:
        return torch.stack(output)
    return x


@torch.no_grad()
def ot_sampler(
    model: torch.nn.Module,
    shape: tuple,
    num_steps: int = 50,
    *labels,
    history: bool = False,
):
    """
    Euler sampler for flow matching (deterministic).

    Args:
        model: trained FlowMatchingModel (time-dependent velocity field)
        shape: tuple, shape of samples (batch_size, dims...)
        num_steps: number of integration steps
        *labels: optional conditioning inputs
        history: if True, returns all intermediate states
    """
    output = []
    batch_size = shape[0]
    device = model.device

    # Start from source distribution ~ N(0, I)
    x = torch.randn(*shape, device=device)

    # Integration parameters
    dt = 1.0 / num_steps
    timesteps = torch.linspace(0, 1, num_steps, device=device)

    model.eval()

    for i, t_i in enumerate(tqdm(timesteps)):
        # Expand scalar timestep for batch
        batch_t = t_i.expand(batch_size)

        # Predict velocity
        v = model(x, batch_t, *labels)

        # Euler integration step
        x = x + dt * v

        if history:
            output.append(x.clone())

    if history:
        return torch.stack(output)  # [num_steps, batch_size, ...]
    return x


# def em_sampler(
#     model: torch.nn.Module,
#     shape: tuple,
#     num_steps: int,
#     *labels,
#     history: bool = False,
#     eps=1e-3
# ):
#     """EULER-MARUYAMA stochastic sampler ~ √dt"""
#     output = []
#     batch_size = shape[0]
#     device = model.device

#     # step_size = 1 / num_steps
#     # step_size_sqrt = step_size**0.5

#     timesteps = torch.linspace(1, eps, num_steps, device=device)

#     g_t = model.diffusion_coeff(timesteps)
#     step_size = 1 / num_steps
#     step_size_sqrt = step_size**0.5

#     t0 = torch.ones(1, device=device)
#     std = model.stddev(t0)
#     x = torch.randn(*shape, device=device) * std

#     model.eval()

#     for i, t_i in enumerate(tqdm(timesteps)):
#         batch_t = t_i.expand(batch_size)

#         drift = g_t[i] ** 2 * model(x, batch_t, *labels)
#         noise = g_t[i] * torch.randn_like(x) if t_i > eps else 0.0

#         x = x + drift * step_size + step_size_sqrt * noise

#         if history:
#             output.append(x)
#     if history:
#         return torch.stack(output)
#     return x
