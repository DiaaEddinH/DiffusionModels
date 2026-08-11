from __future__ import annotations

import yaml
import torch
import difflib

from abc import ABC
from pathlib import Path
from torch.nn import Module
from types import UnionType
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
            for name in names:
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
        Remove a registration. Mainly for testing isolation - register
        a throwaway name in fxture, then unregister it in teardown.

        :param name: Name of registered object to be removed
        :type name: str
        """
        self._entries.pop(name, None)

    def build(self, name: str, params: dict[str, Any] | None = None) -> Any:
        return self.get(name)(**(params or {}))


NETWORK_REGISTRY = Registry("network")
SAMPLER_REGISTRY = Registry("sampler")
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

    def to_dict(self) -> dict[str | Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path):
        with Path(path).open("w") as fp:
            yaml.dump(
                self.to_dict(), stream=fp, Dumper=self.yaml_dumper, sort_keys=False
            )

    @classmethod
    def from_dict(cls: type[T], data: dict[str | Any]) -> T:
        valid_field_names = {f.name: f for f in fields(cls)}
        unknown_field_names = set(data.keys()) - valid_field_names.keys()
        if unknown_field_names:
            cls.suggest_correct_field_names(unknown_field_names, valid_field_names)
            # raise ValueError(f"Unknown key(s) for {cls.__name__}: {sorted(unknown_field_names)}")

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

    @classmethod
    def suggest_correct_field_names(
        cls, unknown_field_names: list[str], valid_field_names: list[str]
    ):
        suggestions = {}
        for unknown_field_name in unknown_field_names:
            close_field_names = difflib.get_close_matches(
                unknown_field_name,
                valid_field_names.keys(),
            )
            suggestions[unknown_field_name] = close_field_names

        details = []
        for field_name, close_field_names in suggestions.items():
            if close_field_names:
                details.append(
                    f"  - {field_name!r} (did you mean: "
                    f"{', '.join(repr(name) for name in close_field_names)})"
                )
            else:
                details.append(f"  - {field_name!r}")

        message = "Unknown field name(s):\n" + "\n".join(details)
        raise ValueError(message)


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

    decay_rate: float = 0.999
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
    log_dir: str = "logs/runs"
    metadata_csv_path: str = "./data/run_metadata.csv"
    save_weight_history: bool = False
    weight_history_frequency: int = 10


@dataclass
class RunConfig(YAMLConfig):
    """
    Maps directly onto Trainer.train()'s keyword arguments.
    `batch_size` is used to construct the DataLoader used in Trainer.
    """

    N_epochs: int
    batch_size: int = 32
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
    sampler: ComponentConfig = field(
        default_factory=lambda: ComponentConfig(name="euler_maruyama", params={})
    )
    extra: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------------
# Builders
# ------------------------------------------------------------------------


def build_optimizer(model: Module, config: ComponentConfig) -> Optimizer:
    cls = OPTIMIZER_REGISTRY.get(config.name)
    return cls(model.parameters(), **config.params)

def build_sampler(model: Module, config: ComponentConfig) -> Any:
    cls = SAMPLER_REGISTRY.get(config.name)
    return cls(model)

def build_lr_scheduler(
    optimizer: Optimizer, config: ComponentConfig | None
) -> LRScheduler:
    if config is None:
        return None
    cls = LR_SCHEDULER_REGISTRY.get(config.name)
    return cls(optimizer, **config.params)
