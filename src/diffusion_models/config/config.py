import yaml
import torch

from pathlib import Path
from torch.nn import Module
from typing import Any, TypeVar
from torch.optim import Optimizer
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from torch.optim.lr_scheduler import LRScheduler


T = TypeVar("T")

# ------------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------------


class Registry:
    """
    Name -> class lookup for a category of component eg network, schedule etc
    """
    def __init__(self, kind: str):
        self._kind = kind
        self._entries: dict[str, type] = {}

    def register(self, *names: str) -> Callable[[type], type]:
        """
        Use as a decorator, with one or more aliases for the same class::

            @NETWORK_REGISTRY.register("unet")
            class UNet(torch.nn.Module): ...

            @SCHEDULE_REGISTRY.register("geometric", "ve")
            class GeometricSchedule(Schedule): ...
        
        Also, callable directly for out-of-package classes::
            SCHEDULE_REGISTRY.register("geometric", "ve")(GeometricSchedule)
        """
        if not names:
            raise ValueError("register() requires at least one name")

        def _decorator(cls: type) -> type:
            for name in names:
                if name in self._entries:
                    raise ValueError(f"{self._kind} '{name}' is already registered")
                self._entries[name] = cls
            return cls

        return _decorator

    def get(self, name: str) -> type:
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(
                f"Unknown {self._kind} '{name}'. Available: {sorted(self._entries)}"
            ) from None

    def unregister(self, name: str):
        """
        Remove a registration. Mainly for testing purposes

        :param name: Name of registered object to be removed
        :type name: str
        """
        self._entries.pop(name, None)

    def build(self, name: str, params: dict[str, Any] | None = None) -> Any:
        return self.get(name)(**(params or {}))


NETWORK_REGISTRY = Registry("network")
SCHEDULE_REGISTRY = Registry("schedule")
OPTIMIZER_REGISTRY = Registry("optimizer")
LR_SCHEDULER_REGISTRY = Registry("lr_scheduler")

for _name in ("Adam", "AdamW", "SGD", "RMSprop"):
    OPTIMIZER_REGISTRY.register(_name.lower())(getattr(torch.optim, _name))

for _name, _alias in (
    ("StepLR", "step"),
    ("CosineAnnealingLR", "cosine"),
    ("ExponentialLR", "exponential"),
):
    LR_SCHEDULER_REGISTRY.register(_alias)(getattr(torch.optim.lr_scheduler, _name))


# ------------------------------------------------------------------------
# Config section
# ------------------------------------------------------------------------

def _dataclass_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """
    Build a dataclass from a dict, raising on unknown keys.

    :param cls: Dataclass to construct
    :type cls: type[T]
    :param data: dict read from a YAML config file
    :type data: dict[str, Any]
    :return: Constructed dataclass.
    :rtype: T
    """
    valid = {f.name for f in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown key(s) for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


@dataclass
class ComponentConfig:
    """
    Generic 'name + params' block for anything resolved via a Registry
    (network, schedule, optimizer, lr_scheduler).
    """
    name: str
    params: dict[str, Any] = field(default_factory=dict)

@dataclass
class ScoreModelConfig:
    """
    Maps directly onto ScoreModel.__init__'s keyword arguments.
    """
    ema_decay: float = 0.999
    device: str | torch.device | None = None

@dataclass
class TrainerConfig:
    """
    Maps directly onto Trainer.__init__'s keyword arguments.
    """
    file_path: str
    use_ddp: bool = False
    checkpoint_dir: str = "./data/checkpoints"
    weight_dir: str = "./data/weights"

@dataclass
class RunConfig:
    """
    Maps directly onto Trainer.train()'s keyword arguments.
    """
    N_epochs: int
    early_stopping: int = 10
    min_delta: float = 1e-4


@dataclass
class ExperimentConfig:
    network: ComponentConfig
    schedule: ComponentConfig
    trainer: TrainerConfig
    run: RunConfig
    model: ScoreModelConfig = field(default_factory=ScoreModelConfig)
    optimizer: ComponentConfig = field(
        default_factory=lambda: ComponentConfig(name="adam", params={"lr": 2e-4})
    )
    lr_scheduler: ComponentConfig | None = None

    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


# ------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------

def load_experiment_config(yaml_path: str | Path) -> ExperimentConfig:
    """Parse a Yaml file into an ExperimentConfig.

    :param yaml_path: File path of Yaml config file
    :type yaml_path: str | Path
    :return: An `ExperimentConfig` dataclass
    :rtype: ExperimentConfig
    """
    yaml_path = Path(yaml_path)
    with yaml_path.open("r") as f:
        raw = yaml.safe_load(f)

    return ExperimentConfig(
        network=_dataclass_from_dict(ComponentConfig, raw["network"]),
        schedule=_dataclass_from_dict(ComponentConfig, raw["schedule"]),
        trainer=_dataclass_from_dict(TrainerConfig, raw["trainer"]),
        run=_dataclass_from_dict(RunConfig, raw["run"]),
        model=_dataclass_from_dict(ScoreModelConfig, raw.get("model", {})),
        optimizer=(
            _dataclass_from_dict(ComponentConfig, raw["optimizer"])
            if "optimizer" in raw
            else ComponentConfig(name="adam", params={"lr": 2e-4})
        ),
        lr_scheduler=(
            _dataclass_from_dict(ComponentConfig, raw["lr_scheduler"])
            if "lr_scheduler" in raw
            else None
        ),
        raw=raw,
    )


# ------------------------------------------------------------------------
# Builders
# ------------------------------------------------------------------------

def build_optimizer(model: Module, config: ComponentConfig) -> Optimizer:
    cls = OPTIMIZER_REGISTRY.get(config.name)
    return cls(model.parameters(), **config.name)

def build_lr_scheduler(optimizer: Optimizer, config: ComponentConfig | None) -> LRScheduler:
    if config is None:
        return None
    cls = LR_SCHEDULER_REGISTRY.get(config.name)
    return cls(optimizer, **config.params)
