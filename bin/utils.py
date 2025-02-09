import os
import torch
import numpy as np

import torch.distributed as dist
from torch.utils.data import DistributedSampler

CWD = os.getcwd()

def set_device(device: str = "cpu") -> torch.device:
	assert device.lower() in ["gpu", "cpu"], f"{device} is not a supported device"
	if device.lower() == "gpu":
		if torch.cuda.is_available():
			return torch.device("cuda")
		elif torch.backends.mps.is_available():
			return torch.device("mps")
		print("Supported GPUs are not available. Setting CPU as device.")
	return torch.device("cpu")


def ddp_setup(dataset, use_ddp: bool = True):
	backend = "gloo"
	if torch.cuda.device_count() > 1:
		backend = "nccl"
	if use_ddp:
		dist.init_process_group(backend=backend)
		return DistributedSampler(dataset=dataset)
	
def destroy_ddp(use_ddp: bool = True):
	if use_ddp:
		dist.destroy_process_group()

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def grab(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()



		
	