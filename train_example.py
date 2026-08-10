import torch
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler

from diffusion_models.trainer import Trainer
from diffusion_models.models.models import ScoreModel
from diffusion_models.networks.networks import LinearNet
from diffusion_models.datasets.datasets import DoublePeak
from diffusion_models.config.config import ExperimentConfig
from diffusion_models.sampling.samplers import EulerMaruyamaSampler
from diffusion_models.noise.noise_scheduler import GeometricSchedule
from diffusion_models.utils import set_device, ddp_setup, destroy_ddp, detach_to_numpy
from diffusion_models.config.config import NETWORK_REGISTRY, SCHEDULE_REGISTRY

NETWORK_REGISTRY.register("linear")(LinearNet)
SCHEDULE_REGISTRY.register("geometric")(GeometricSchedule)

samples_dir = Path("./data/samples")
samples_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    yaml_path = "configs/example_config.yaml"
    config = ExperimentConfig.from_yaml(yaml_path)

    ddp_setup(use_ddp=config.trainer.use_ddp)

    dataset = DoublePeak(sigma=0.25, mu=np.array([1, -1]), size=10_000)
    data_sampler = DistributedSampler(dataset=dataset) if config.trainer.use_ddp else None

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.run.batch_size,
        sampler=data_sampler,
        shuffle=(data_sampler is None),
        drop_last=True,
    )

    model = ScoreModel.from_config(config)

    trainer = Trainer.from_config(model, config)

    trainer.train(loader=data_loader)

    destroy_ddp(use_ddp=config.trainer.use_ddp)

    # print("Generating samples...")
    # # Turn off gradient calculations for inference
    # em_sampler = torch.no_grad(em_sampler)

    # model._load_weights(weight_file)

    # model.ema.apply_shadow()
    # samples = detach_to_numpy(em_sampler(model, (args.sample_size, 2), args.time_steps))
    # model.ema.restore()

    # print("Saving samples...")
    # samples_file = samples_dir / f"{args.file}_samples.npy"
    # np.save(samples_file, samples)
    # print("All done!")
