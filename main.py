import os
import torch
import numpy as np

from src.utils import set_device, ddp_setup, destroy_ddp, grab
from torch.utils.data import DataLoader, DistributedSampler
from src.datasets.datasets import DoublePeak
from src.config_loader import parse_configs
from src.samplers import em_sampler
from src.networks import LinearNet
from src.models import ScoreModel
from src.trainer import Trainer
from pathlib import Path


samples_dir = Path("./data/samples")
samples_dir.mkdir(parents=True, exist_ok=True)

weight_dir = Path("./data/weights")

if __name__ == "__main__":
    args = parse_configs()
    ddp_setup(use_ddp=args.ddp)
    device = set_device(args.device)

    weight_file = weight_dir / f"{args.file}_weights.pt"

    dataset = DoublePeak(mu=np.array([1, -1]), sigma=0.25, size=1_000_000)

    sampler = DistributedSampler(dataset=dataset) if args.ddp else None

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
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

    trainer.train(loader=loader, N_epochs=args.max_epochs, early_stopping=args.patience)

    destroy_ddp(args.ddp)

    print("Generating samples...")
    # Turn off gradient calculations for inference
    em_sampler = torch.no_grad(em_sampler)

    model._load_weights(weight_file)

    model.ema.apply_shadow()
    samples = grab(em_sampler(model, (args.sample_size, 2), args.time_steps))
    model.ema.restore()

    print("Saving samples...")
    samples_file = samples_dir / f"{args.file}_samples.npy"
    np.save(samples_file, samples)
    print("All done!")
