import torch

from pathlib import Path

from diffusion_models.models.models import ScoreModel
from diffusion_models.networks.networks import LinearNet
from diffusion_models.config.config import ExperimentConfig
from diffusion_models.sampling.samplers import EulerMaruyamaSampler
from diffusion_models.noise.noise_scheduler import GeometricSchedule
from diffusion_models.config.config import (
    NETWORK_REGISTRY,
    SCHEDULE_REGISTRY,
    SAMPLER_REGISTRY,
    build_sampler
)

NETWORK_REGISTRY.register("linear")(LinearNet)
SCHEDULE_REGISTRY.register("geometric")(GeometricSchedule)
SAMPLER_REGISTRY.register("euler_maruyama")(EulerMaruyamaSampler)

samples_dir = Path("./data/samples")
samples_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    yaml_path = "configs/example_config.yaml"
    config = ExperimentConfig.from_yaml(yaml_path)

    model = ScoreModel.from_config(config)
    sampler = build_sampler(model, config.sampler)

    with torch.no_grad(), model.exponential_moving_average.average_parameters():
        samples = sampler.sample(
            shape = (10_000, 2),
            **config.sampler.params
        )

    # At this point you detach to numpy and save it as .npy/.npz or save it as a torch tensor to .pt

    