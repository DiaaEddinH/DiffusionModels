from __future__ import annotations

import yaml
import torch

from abc import ABC
from pathlib import Path
from types import UnionType
from torch.nn import Module
from torch.optim import Optimizer
from collections.abc import Callable
from torch.optim.lr_scheduler import LRScheduler
from dataclasses import dataclass, field, fields, asdict, is_dataclass
from typing import Any, Union, TypeVar, get_args, get_origin, get_type_hints


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


@dataclass
class YAMLConfig(ABC):
    """
    Base class for decoding/enconding dataclasses from/onto YAML config files.
    """

    yaml_dumper = getattr(yaml, "CDumper", yaml.Dumper)
    yaml_loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

    def _encoder(self, data: Any) -> str | bytes:
        return yaml.dump(data, Dumper=self.yaml_dumper)

    def _decoder(self, data: str | bytes) -> dict[str, Any]:
        return yaml.load(data, Loader=self.yaml_loader)

    def to_dict(self) -> dict[str | Any]:
        exclude = {f.name for f in fields(self) if f.metadata.get("yaml_exclude")}
        return {k: v for k, v in asdict(self).items() if k not in exclude}

    def to_yaml(self, path: str | Path):
        with Path(path).open("w") as fp:
            yaml.dump(
                self.to_dict(), stream=fp, Dumper=self.yaml_dumper, sort_keys=False
            )

    @classmethod
    def from_dict(cls: type[T], data: dict[str | Any]) -> T:
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError(f"Unknown key(s) for {cls.__name__}: {sorted(unknown)}")

        hints = get_type_hints(cls)
        kwargs: dict[str, Any] = {}
        for name, value in data.items():
            field_type = cls._unwrap_optional(hints[name])
            if (
                isinstance(value, dict)
                and is_dataclass(field_type)
                and issubclass(field_type, YAMLConfig)
            ):
                value = field_type.from_dict(value)
            kwargs[name] = value
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls: type[T], path: str | Path) -> T:
        with Path(path).open("r") as fp:
            raw = yaml.load(stream=fp, Loader=cls.yaml_loader)
        return cls.from_dict(raw)

    @staticmethod
    def _unwrap_optional(_type: Any) -> Any:
        """
        Optional[X] / X | None -> X, else X
        """
        origin = get_origin(_type)
        if origin is Union or origin is UnionType:
            args = [a for a in get_args(_type) if a is not type(None)]
            if len(args) == 1:
                return args[0]
        return _type


@dataclass
class ComponentConfig(YAMLConfig):
    """
    Generic 'name + params' block for anything resolved via a Registry
    (network, schedule, optimizer, lr_scheduler).
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreModelConfig(YAMLConfig):
    """
    Maps directly onto ScoreModel.__init__'s keyword arguments.
    """

    ema_decay: float = 0.999
    device: str | torch.device | None = None


@dataclass
class TrainerConfig(YAMLConfig):
    """
    Maps directly onto Trainer.__init__'s keyword arguments.
    """

    file_path: str
    use_ddp: bool = False
    checkpoint_dir: str = "./data/checkpoints"
    weight_dir: str = "./data/weights"


@dataclass
class RunConfig(YAMLConfig):
    """
    Maps directly onto Trainer.train()'s keyword arguments.
    """

    N_epochs: int
    early_stopping: int = 10
    min_delta: float = 1e-4


@dataclass
class ExperimentConfig(YAMLConfig):
    network: ComponentConfig
    schedule: ComponentConfig
    trainer: TrainerConfig
    run: RunConfig
    model: ScoreModelConfig = field(default_factory=ScoreModelConfig)
    optimizer: ComponentConfig = field(
        default_factory=lambda: ComponentConfig(name="adam", params={"lr": 2e-4})
    )
    lr_scheduler: ComponentConfig | None = None

    raw: dict[str, Any] = field(
        default_factory=dict, repr=False, compare=False, metadata={"yaml_exclude": True}
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        obj = super().from_dict(data)
        obj.raw = data
        return obj


# ------------------------------------------------------------------------
# Builders
# ------------------------------------------------------------------------


def build_optimizer(model: Module, config: ComponentConfig) -> Optimizer:
    cls = OPTIMIZER_REGISTRY.get(config.name)
    return cls(model.parameters(), **config.name)


def build_lr_scheduler(
    optimizer: Optimizer, config: ComponentConfig | None
) -> LRScheduler:
    if config is None:
        return None
    cls = LR_SCHEDULER_REGISTRY.get(config.name)
    return cls(optimizer, **config.params)


if __name__ == "__main__":

    yaml_path = "configs/newexample_config.yaml"
    config = ExperimentConfig.from_yaml(yaml_path)
    # config.to_yaml("configs/test_exampler.yaml")
    print(config)
