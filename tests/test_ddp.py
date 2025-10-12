import os
import torch
import numpy as np

from diffusion_models.config.config_loader import parse_configs
from diffusion_models.datasets.datasets import DoublePeak
from diffusion_models.networks.networks import LinearNet
from diffusion_models.models.models import ScoreModel
from diffusion_models.training.trainer import Trainer

from torch.utils.data import DataLoader, DistributedSampler

import torch.distributed as dist


def ddp_setup(use_ddp: bool = True):
    backend = "gloo"
    if torch.cuda.device_count() > 1:
        backend = "nccl"
    if use_ddp:
        dist.init_process_group(backend=backend)


def destroy_ddp(use_ddp: bool = True):
    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def set_device(device: str = "cpu") -> torch.device:
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


if __name__ == "__main__":
    args = parse_configs()
    ddp_setup(use_ddp=args.ddp)
    device = set_device(args.device)

    dataset = DoublePeak(mu=np.array([1, -1]), sigma=0.25)
    sampler = DistributedSampler(dataset) if args.ddp else None

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    network = LinearNet(
        in_channels=args.in_channels,
        channels=args.hidden_channels,
        time_channels=args.time_channels,
        activation=args.activation(),
        device=device,
    )

    model = ScoreModel(
        network=network,
        schedule="geometric",
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        device=device,
    )

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    trainer = Trainer(
        model=model, optimizer=optimiser, file_path=args.file, device=device
    )

    trainer.train(loader=loader, N_epochs=args.max_epochs, early_stopping=2)

    destroy_ddp(args.ddp)
