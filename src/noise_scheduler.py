import torch
from math import log


def get_noise_schedule(obj, schedule_type="geometric", **kwargs):
    """
    Returns a tuple of two functions:
            - get_mean_std(x, t)
            - diffusion_coeff(t)
    schedule_type: str, one of ["geometric", "linear", "cosine"]
    kwargs: parameters for each schedule
    """
    if schedule_type == "geometric":
        sigma_min = kwargs.get("sigma_min", 0.02)
        sigma_max = kwargs.get("sigma_max", 10.0)
        logsigma = log(sigma_max / sigma_min)

        obj.stddev = lambda t: sigma_min * torch.sqrt(
            (torch.exp(2 * t * logsigma) - 1) / (2 * logsigma)
        )
        obj.diffusion_coeff = lambda t: sigma_min * torch.exp(t * logsigma)

    elif schedule_type == "linear":
        sigma_min = kwargs.get("sigma_min", 0.02)
        sigma_max = kwargs.get("sigma_max", 10.0)

        obj.stddev = lambda t: sigma_min + (sigma_max - sigma_min) * t
        obj.diffusion_coeff = lambda t: (sigma_max - sigma_min) * torch.ones_like(t)

    # elif schedule_type == "cosine":
    #     sigma_min = kwargs.get("sigma_min", 0.02)
    #     sigma_max = kwargs.get("sigma_max", 10.0)
    #     PI = acos(-1.0)

    #     def get_mean_std(x, t):
    #         cos_term = torch.cos(t * PI / 2)
    #         std = sigma_min + (sigma_max - sigma_min) * (1 - cos_term)
    #         mean = x
    #         return mean, std

    #     def diffusion_coeff(t):
    #         cos_term = torch.cos(t * PI / 2)
    #         return sigma_min + (sigma_max - sigma_min) * (1 - cos_term)

    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")

    obj.get_mean_stddev = lambda x, t: (x, obj.stddev(t))
