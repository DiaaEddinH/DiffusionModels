import csv
import json
import logging

from enum import StrEnum, auto
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, fields, asdict, is_dataclass
from diffusion_models.config.config import ExperimentConfig, YAMLConfig
from typing import Any, Union, TypeVar, get_args, get_origin, get_type_hints


class LogStatus(StrEnum):
    STARTED = auto()
    COMPLETED = auto()
    INTERRUPTED = auto()
    FAILED = auto()


def build_run_logger(name: str, log_path: str | Path, rank: int = 0) -> logging.Logger:
    """
    Builds or rebuilds a logger that write to `log_path` and stdout.
    In distributed instances, only rank 0 gets handlers attached - other ranks get a logger without handles
    so they won't duplicate output across processes.


    :param name: Logger name. If a log with this name already exists, the same logger will be rebuilt/reused.
    :type name: str
    :param log_path: File to write log lines into. Parent directories are created if missing.
    :type log_path: str | Path
    :param rank: Distributed data parallel rank of the current process. defaults to 0 (single-process).
    :type rank: int, optional
    :return: A configured logger.
    :rtype: logging.Logger
    """
    logger = logging.getLogger(f"trainer.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear existing handlers so re-calling for the same run doesn't stack duplicate handlers, eg when resuming a run in the same process.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    if rank == 0:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

    return logger


def stringify_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def csv_fieldnames_for_experiment_config() -> list[str]:
    hints = get_type_hints(ExperimentConfig)
    names: list[str] = []
    for f in fields(ExperimentConfig):
        field_type = YAMLConfig._unwrap_optional(hints[f.name])
        if is_dataclass(field_type) and issubclass(field_type, YAMLConfig):
            names.extend(f"{f.name}.{sub_f.name}" for sub_f in fields(field_type))
        else:
            names.append(f.name)
    return names


def flatten_experiment_config(config: ExperimentConfig) -> dict[str, str]:
    hints = get_type_hints(ExperimentConfig)
    row: dict[str, str] = {}

    for f in fields(ExperimentConfig):
        field_type = YAMLConfig._unwrap_optional(hints[f.name])
        value = getattr(config, f.name)

        if is_dataclass(field_type) and issubclass(field_type, YAMLConfig):
            for sub_f in fields(field_type):
                col = f"{f.name}.{sub_f.name}"
                sub_value = getattr(value, sub_f.name) if value is not None else None
                row[col] = stringify_csv_value(sub_value)
        else:
            row[f.name] = stringify_csv_value(value)
    return row


class RunMetadataLogger:
    """
    Appends one CSV row per training run with the exact ExperimentConfig that produced it.
    """

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.field_names = [
            "run_id",
            "timestamp",
            "log_file",
            "status",
            "final_epoch",
            "final_loss",
        ] + csv_fieldnames_for_experiment_config()

    def log_run(
        self,
        config: ExperimentConfig,
        run_id: str,
        log_file: str | Path | None = None,
        status: LogStatus = LogStatus.STARTED,
        final_epoch: int | None = None,
        final_loss: float | None = None,
    ):
        row = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "log_file": str(log_file) if log_file is not None else "",
            "status": status,
            "final_epoch": "" if final_epoch is None else str(final_epoch),
            "final_loss": "" if final_loss is None else str(final_loss),
        }
        row.update(flatten_experiment_config(config))

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=self.field_names)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
