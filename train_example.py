import numpy as np

from torch.utils.data import DataLoader, DistributedSampler

from diffusion_models.trainer import Trainer
from diffusion_models.models.models import ScoreModel
from diffusion_models.networks.networks import LinearNet
from diffusion_models.utils import ddp_setup, destroy_ddp
from diffusion_models.datasets.datasets import DoublePeak
from diffusion_models.config.config import ExperimentConfig
from diffusion_models.noise.noise_scheduler import GeometricSchedule
from diffusion_models.config.config import NETWORK_REGISTRY, SCHEDULE_REGISTRY


NETWORK_REGISTRY.register("linear")(LinearNet)
SCHEDULE_REGISTRY.register("geometric")(GeometricSchedule)

if __name__ == "__main__":
    yaml_path = "configs/example_config.yaml"
    config = ExperimentConfig.from_yaml(yaml_path)

    ddp_setup(use_ddp=config.trainer.use_ddp)

    dataset = DoublePeak(sigma=0.25, mu=np.array([1, -1]), size=10_000)
    data_sampler = (
        DistributedSampler(dataset=dataset) if config.trainer.use_ddp else None
    )

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
