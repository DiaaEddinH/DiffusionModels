import os
import torch
import numpy as np

import torch.distributed as dist
from torch.nn.modules import activation

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def set_device(device: str = "cpu") -> torch.device:
    """
    Set the device to GPU if available, otherwise CPU.
    :param device: "cpu" or "gpu"
    :return: torch.device object
    """
    device = device.lower()
    assert device in ["gpu", "cpu"], f"{device} is not a supported device"

    if device == "gpu":
        if torch.cuda.is_available():
            rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(rank)
            return torch.device(f"cuda:{rank}")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        print("Supported GPUs are not available. Setting CPU as device.")
    return torch.device("cpu")


def ddp_setup(use_ddp: bool = True):
    """
    Initialize the distributed data parallel (DDP) environment.
    :param use_ddp: Whether to use DDP or not.
    """
    backend = "gloo"
    if torch.cuda.device_count() > 1:
        backend = "nccl"
    if use_ddp:
        dist.init_process_group(backend=backend)


def destroy_ddp(use_ddp: bool = True):
    """
    Destroy the distributed data parallel (DDP) environment.
    :param use_ddp: Whether to use DDP or not.
    """
    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def count_trainable_parameters(model):
    """
    Count the number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def detach_to_numpy(x: torch.Tensor) -> np.ndarray:
    """
    Move a tensor to CPU and convert to numpy array.
    """
    return x.detach().cpu().numpy()


def get_activation_func(act: str):
    """
    Get activation function from string.
    """
    # get list from activatoin submodule as lower-case
    activations_list = [str(a).lower() for a in activation.__all__]
    if (act := str(act).lower()) in activations_list:
        # match actual name from lower-case list, return function/factory
        index = activations_list.index(act)
        act_name = activation.__all__[index]
        act_func = getattr(activation, act_name)
        return act_func
    else:
        raise ValueError(f"Cannot find activation function for string <{act}>")


def set_default_plot_parameters():
    """
    Set default plot parameters for matplotlib to ensure consistency and readability in plots.
    """
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],  # or any other serif font you prefer
            "font.size": 20,  # Set the default font size
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "xtick.minor.width": 1,
            "ytick.minor.width": 1,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            # Colorblind-friendly colors
            "axes.prop_cycle": plt.cycler(
                color=[
                    "#4477AA",  # blue
                    "#EE6677",  # red
                    "#228833",  # green
                    "#CCBB44",  # yellow
                    "#66CCEE",  # cyan
                    "#AA3377",  # purple
                    "#BBBBBB",  # gray
                ]
            ),
            "image.cmap": "viridis",  # Colorblind-friendly colormap
        }
    )

    plt.minorticks_on()
    # Set the minor tick frequency globally
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))
    plt.close()
