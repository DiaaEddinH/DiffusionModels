import csv
import json
import pytest
from pathlib import Path
from types import SimpleNamespace

import torch

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from diffusion_models.trainer import Trainer
from diffusion_models.models.models import ScoreModel
from tests.conftest import DummyNetwork, DummySchedule
from diffusion_models.config.config import (
    ExperimentConfig,
    ComponentConfig,
    ScoreModelConfig,
    TrainerConfig,
    RunConfig,
    NETWORK_REGISTRY,
    SCHEDULE_REGISTRY,
)
from diffusion_models.loggingtools.utils import LogStatus


@pytest.fixture
def score_model(dummy_network, dummy_schedule):
    return ScoreModel(dummy_network, dummy_schedule, decay_rate=0.9)


@pytest.fixture
def loader():
    data = torch.randn(16, 3)
    return DataLoader(TensorDataset(data), batch_size=4)


@pytest.fixture
def trainer_dirs(tmp_path):
    return {
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "weight_dir": str(tmp_path / "weights"),
        "log_dir": str(tmp_path / "logs"),
        "metadata_csv_path": str(tmp_path / "metadata.csv"),
    }


@pytest.fixture
def trainer(score_model, trainer_dirs):
    optimizer = torch.optim.SGD(score_model.parameters(), lr=0.1)
    return Trainer(
        model=score_model,
        optimizer=optimizer,
        file_path="test_run",
        device=torch.device("cpu"),
        **trainer_dirs,
    )


def _read_rows(path):
    with open(path, newline="") as fp:
        return list(csv.DictReader(fp))


class TestBasicTrainingLoop:
    def test_runs_and_updates_history(self, trainer, loader):
        trainer.train(loader, N_epochs=3)
        assert len(trainer.history) == 3

    def test_checkpoint_created(self, trainer, loader):
        trainer.train(loader, N_epochs=1)
        assert trainer.checkpoint.exists()

    def test_best_weight_file_created(self, trainer, loader):
        trainer.train(loader, N_epochs=3)
        assert trainer.weight_file.exists()

    def test_log_file_created_and_contains_epoch_lines(self, trainer, loader):
        trainer.train(loader, N_epochs=2)
        assert trainer.log_file.exists()
        content = trainer.log_file.read_text()
        assert "Epoch 0" in content
        assert "Epoch 1" in content

    def test_delegates_to_model_train_step(
        self, trainer, score_model, loader, monkeypatch
    ):
        calls = []
        orig = score_model.train_step

        def spy(batch, optimizer, *labels, lr_scheduler=None):
            calls.append((optimizer is trainer.optimizer, lr_scheduler))
            return orig(batch, optimizer, *labels, lr_scheduler=lr_scheduler)

        monkeypatch.setattr(score_model, "train_step", spy)
        trainer.train(loader, N_epochs=1)
        assert len(calls) > 0
        assert all(uses_optimizer for uses_optimizer, _ in calls)


class TestExponentialMovingAverageCheckpointRoundtrip:
    def test_exponential_moving_average_state_survives_checkpoint_reload(
        self, score_model, trainer_dirs, loader
    ):
        optimizer = torch.optim.SGD(score_model.parameters(), lr=0.1)
        trainer1 = Trainer(
            model=score_model,
            optimizer=optimizer,
            file_path="ema_run",
            device=torch.device("cpu"),
            **trainer_dirs,
        )
        trainer1.train(loader, N_epochs=1)
        shadow_before = score_model.exponential_moving_average.shadow[
            "network.scale"
        ].clone()

        fresh_network = DummyNetwork(scale=999.0)
        fresh_schedule = DummySchedule(std_scale=2.0)
        fresh_model = ScoreModel(
            network=fresh_network, schedule=fresh_schedule, decay_rate=0.9
        )
        fresh_optimizer = torch.optim.SGD(fresh_model.parameters(), lr=0.1)
        Trainer(
            model=fresh_model,
            optimizer=fresh_optimizer,
            file_path="ema_run",
            device=torch.device("cpu"),
            **trainer_dirs,
        )

        assert torch.allclose(
            fresh_model.exponential_moving_average.shadow["network.scale"],
            shadow_before,
        )


class TestResume:
    def test_epochs_continue_from_checkpoint(self, score_model, trainer_dirs, loader):
        optimizer = torch.optim.SGD(score_model.parameters(), lr=0.1)
        trainer1 = Trainer(
            model=score_model,
            optimizer=optimizer,
            file_path="resume_run",
            device=torch.device("cpu"),
            **trainer_dirs,
        )
        trainer1.checkpoint_frequency = (
            1  # checkpoint every epoch, for a deterministic resume point
        )
        trainer1.train(loader, N_epochs=3)

        fresh_network = DummyNetwork(scale=999.0)
        fresh_schedule = DummySchedule(std_scale=2.0)
        fresh_model = ScoreModel(network=fresh_network, schedule=fresh_schedule)
        fresh_optimizer = torch.optim.SGD(fresh_model.parameters(), lr=0.1)
        trainer2 = Trainer(
            model=fresh_model,
            optimizer=fresh_optimizer,
            file_path="resume_run",
            device=torch.device("cpu"),
            **trainer_dirs,
        )

        assert trainer2.epochs == 2
        assert len(trainer2.history) == 3


class TestWeightHistory:
    def test_disabled_by_default_no_history_dir_created(self, trainer, loader):
        trainer.train(loader, N_epochs=3)
        assert not trainer.weight_history_dir.exists()

    def test_enabled_saves_snapshots_at_configured_frequency(
        self, score_model, trainer_dirs, loader
    ):
        optimizer = torch.optim.SGD(score_model.parameters(), lr=0.1)
        trainer = Trainer(
            model=score_model,
            optimizer=optimizer,
            file_path="with_history",
            device=torch.device("cpu"),
            save_weight_history=True,
            weight_history_frequency=2,
            **trainer_dirs,
        )
        trainer.train(loader, N_epochs=5)

        for epoch in (0, 2, 4):
            path = (
                trainer.weight_history_dir / f"with_history_weights_epoch{epoch:05d}.pt"
            )
            assert path.exists()
        for epoch in (1, 3):
            path = (
                trainer.weight_history_dir / f"with_history_weights_epoch{epoch:05d}.pt"
            )
            assert not path.exists()

    def test_snapshot_is_independent_of_best_weight_file(
        self, score_model, trainer_dirs, loader
    ):
        optimizer = torch.optim.SGD(score_model.parameters(), lr=0.1)
        trainer = Trainer(
            model=score_model,
            optimizer=optimizer,
            file_path="independent",
            device=torch.device("cpu"),
            save_weight_history=True,
            weight_history_frequency=1,
            **trainer_dirs,
        )
        trainer.train(loader, N_epochs=2)
        snapshot_path = trainer.weight_history_dir / "independent_weights_epoch00000.pt"
        assert snapshot_path.exists()
        assert trainer.weight_file.exists()
        assert snapshot_path != trainer.weight_file


# ---------------------------------------------------------------------------
# Metadata logging
# ---------------------------------------------------------------------------


class TestMetadataLoggingWithConfig:
    @pytest.fixture(autouse=True)
    def _register(self):
        NETWORK_REGISTRY.register("__trainer_test_network__")(DummyNetwork)
        SCHEDULE_REGISTRY.register("__trainer_test_schedule__")(DummySchedule)
        yield
        NETWORK_REGISTRY.unregister("__trainer_test_network__")
        SCHEDULE_REGISTRY.unregister("__trainer_test_schedule__")

    @pytest.fixture
    def configured_model(self):
        config = ExperimentConfig(
            network=ComponentConfig(
                name="__trainer_test_network__", params={"scale": 0.3}
            ),
            schedule=ComponentConfig(
                name="__trainer_test_schedule__", params={"std_scale": 2.0}
            ),
            trainer=TrainerConfig(file_path="configured_run"),
            run=RunConfig(N_epochs=2),
        )
        return ScoreModel.from_config(config)

    def test_start_row_written_on_construction(
        self, configured_model, trainer_dirs, loader
    ):
        optimizer = torch.optim.SGD(configured_model.parameters(), lr=0.1)
        Trainer(
            model=configured_model,
            optimizer=optimizer,
            file_path="configured_run",
            device=torch.device("cpu"),
            **trainer_dirs,
        )
        rows = _read_rows(trainer_dirs["metadata_csv_path"])
        assert len(rows) == 1
        assert rows[0]["status"] == LogStatus.STARTED
        assert rows[0]["run_id"] == "configured_run"
        assert rows[0]["network.name"] == "__trainer_test_network__"

    def test_end_row_written_on_normal_completion(
        self, configured_model, trainer_dirs, loader
    ):
        optimizer = torch.optim.SGD(configured_model.parameters(), lr=0.1)
        trainer = Trainer(
            model=configured_model,
            optimizer=optimizer,
            file_path="configured_run",
            device=torch.device("cpu"),
            **trainer_dirs,
        )
        trainer.train(loader, N_epochs=2)
        rows = _read_rows(trainer_dirs["metadata_csv_path"])
        assert len(rows) == 2
        assert rows[0]["status"] == LogStatus.STARTED
        assert rows[1]["status"] == LogStatus.COMPLETED
        assert rows[1]["final_epoch"] == "1"
        assert rows[1]["final_loss"] != ""

    def test_no_metadata_rows_when_model_has_no_config(
        self, score_model, trainer_dirs, loader
    ):
        optimizer = torch.optim.SGD(score_model.parameters(), lr=0.1)
        trainer = Trainer(
            model=score_model,
            optimizer=optimizer,
            file_path="no_config_run",
            device=torch.device("cpu"),
            **trainer_dirs,
        )
        trainer.train(loader, N_epochs=2)
        assert not Path(trainer_dirs["metadata_csv_path"]).exists()

    def test_interrupted_row_written_on_keyboard_interrupt(
        self, configured_model, trainer_dirs, loader, monkeypatch
    ):
        optimizer = torch.optim.SGD(configured_model.parameters(), lr=0.1)
        trainer = Trainer(
            model=configured_model,
            optimizer=optimizer,
            file_path="configured_run",
            device=torch.device("cpu"),
            **trainer_dirs,
        )

        call_count = {"n": 0}
        orig_train_step = configured_model.train_step

        def flaky_train_step(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 3:
                raise KeyboardInterrupt()
            return orig_train_step(*args, **kwargs)

        monkeypatch.setattr(configured_model, "train_step", flaky_train_step)

        with pytest.raises(KeyboardInterrupt):
            trainer.train(loader, N_epochs=5)

        rows = _read_rows(trainer_dirs["metadata_csv_path"])
        assert rows[-1]["status"] == LogStatus.INTERRUPTED
        assert rows[-1]["final_epoch"] != ""

    def test_failed_row_written_on_exception(
        self, configured_model, trainer_dirs, loader, monkeypatch
    ):
        optimizer = torch.optim.SGD(configured_model.parameters(), lr=0.1)
        trainer = Trainer(
            model=configured_model,
            optimizer=optimizer,
            file_path="configured_run",
            device=torch.device("cpu"),
            **trainer_dirs,
        )

        def broken_train_step(*args, **kwargs):
            raise RuntimeError("simulated crash")

        monkeypatch.setattr(configured_model, "train_step", broken_train_step)

        with pytest.raises(RuntimeError):
            trainer.train(loader, N_epochs=5)

        rows = _read_rows(trainer_dirs["metadata_csv_path"])
        assert rows[-1]["status"] == LogStatus.FAILED

    def test_interrupted_run_can_be_filtered_from_completed_runs(
        self, configured_model, trainer_dirs, loader, monkeypatch
    ):
        optimizer = torch.optim.SGD(configured_model.parameters(), lr=0.1)
        trainer = Trainer(
            model=configured_model,
            optimizer=optimizer,
            file_path="configured_run",
            device=torch.device("cpu"),
            **trainer_dirs,
        )

        def raise_immediately(*args, **kwargs):
            raise KeyboardInterrupt()

        monkeypatch.setattr(configured_model, "train_step", raise_immediately)
        with pytest.raises(KeyboardInterrupt):
            trainer.train(loader, N_epochs=5)

        rows = _read_rows(trainer_dirs["metadata_csv_path"])
        statuses = {r["status"] for r in rows}
        assert LogStatus.INTERRUPTED in statuses
        assert LogStatus.COMPLETED not in statuses


class TestEarlyStopping:
    def test_stops_when_no_improvement(self, trainer, score_model, loader):
        def constant_loss_train_step(batch, optimizer, *labels, lr_scheduler=None):
            return torch.tensor(1.0)

        score_model.train_step = constant_loss_train_step
        trainer.train(loader, N_epochs=100, early_stopping=3, min_delta=1e-4)
        assert (
            len(trainer.history) == 4
        )  # epochs 0 (best), 1, 2, 3 (counter hits 3) -> stop


class TestFromConfig:
    @pytest.fixture(autouse=True)
    def _register(self):
        NETWORK_REGISTRY.register("__trainer_from_config_network__")(DummyNetwork)
        SCHEDULE_REGISTRY.register("__trainer_from_config_schedule__")(DummySchedule)
        yield
        NETWORK_REGISTRY.unregister("__trainer_from_config_network__")
        SCHEDULE_REGISTRY.unregister("__trainer_from_config_schedule__")

    @pytest.fixture
    def config(self, trainer_dirs):
        return ExperimentConfig(
            network=ComponentConfig(
                name="__trainer_from_config_network__", params={"scale": 0.3}
            ),
            schedule=ComponentConfig(
                name="__trainer_from_config_schedule__", params={"std_scale": 2.0}
            ),
            trainer=TrainerConfig(file_path="from_config_run", **trainer_dirs),
            run=RunConfig(N_epochs=2),
            optimizer=ComponentConfig(name="sgd", params={"lr": 0.05}),
        )

    def test_builds_optimizer_from_config(self, config):
        model = ScoreModel.from_config(config)
        trainer = Trainer.from_config(model, config)
        assert isinstance(trainer.optimizer, torch.optim.SGD)
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.05)

    def test_train_uses_run_config_defaults(self, config, loader):
        model = ScoreModel.from_config(config)
        trainer = Trainer.from_config(model, config)
        trainer.train(loader)  # no N_epochs given
        assert len(trainer.history) == 2

    def test_wires_directories_from_config(self, config):
        model = ScoreModel.from_config(config)
        trainer = Trainer.from_config(model, config)
        assert str(trainer.checkpoint.parent) == config.trainer.checkpoint_dir
        assert str(trainer.weight_file.parent) == config.trainer.weight_dir

    def test_stamps_model_config_even_if_model_had_none(
        self, config, dummy_network, dummy_schedule
    ):
        model = ScoreModel(network=dummy_network, schedule=dummy_schedule)
        assert model.config is None
        Trainer.from_config(model, config)
        assert model.config is config

    def test_wires_weight_history_settings_from_config(self, trainer_dirs):
        config = ExperimentConfig(
            network=ComponentConfig(
                name="__trainer_from_config_network__", params={"scale": 0.3}
            ),
            schedule=ComponentConfig(
                name="__trainer_from_config_schedule__", params={"std_scale": 2.0}
            ),
            trainer=TrainerConfig(
                file_path="hist_run",
                save_weight_history=True,
                weight_history_frequency=3,
                **trainer_dirs,
            ),
            run=RunConfig(N_epochs=2),
        )
        model = ScoreModel.from_config(config)
        trainer = Trainer.from_config(model, config)
        assert trainer.save_weight_history is True
        assert trainer.weight_history_frequency == 3


class TestLabelsInBatch:
    def test_trains_successfully_with_multi_tensor_batches(self, trainer):
        x = torch.randn(8, 3)
        y1 = torch.randn(8, 1)
        y2 = torch.randn(8, 1)
        loader = DataLoader(TensorDataset(x, y1, y2), batch_size=4)

        trainer.train(loader, N_epochs=1)
        assert len(trainer.history) == 1

    def test_labels_forwarded_to_network(self, trainer_dirs):
        seen = {}

        class LabelCapturingNetwork(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, x, t, *labels):
                seen["labels"] = labels
                return self.scale * x

        schedule = DummySchedule(std_scale=2.0)
        model = ScoreModel(network=LabelCapturingNetwork(), schedule=schedule)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            file_path="labels_test",
            device=torch.device("cpu"),
            **trainer_dirs,
        )

        x = torch.randn(4, 3)
        y1 = torch.randn(4, 1)
        loader = DataLoader(TensorDataset(x, y1), batch_size=4)
        trainer.train(loader, N_epochs=1)

        assert len(seen["labels"]) == 1


class TestCheckpointPathsAndManualLoad:
    def test_paths_set_correctly_on_construction(self, score_model, trainer_dirs):
        optimizer = torch.optim.SGD(score_model.parameters(), lr=0.1)
        trainer = Trainer(
            model=score_model,
            optimizer=optimizer,
            file_path="path_check",
            device=torch.device("cpu"),
            **trainer_dirs,
        )
        assert (
            trainer.checkpoint
            == Path(trainer_dirs["checkpoint_dir"]) / "path_check_ckpt.pt"
        )
        assert (
            trainer.weight_file
            == Path(trainer_dirs["weight_dir"]) / "path_check_weights.pt"
        )
        assert (
            trainer.history_file
            == Path(trainer_dirs["checkpoint_dir"]) / "path_check_history.json"
        )

    def test_loads_manually_constructed_checkpoint_on_init(
        self, score_model, trainer_dirs
    ):
        # Isolated test of _load_checkpoint's parsing, independent of
        # actually having trained first (unlike TestResume).
        ckpt_dir = Path(trainer_dirs["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        file_path = "manual_ckpt"
        checkpoint_path = ckpt_dir / f"{file_path}_ckpt.pt"

        ema_shadow = {"network.scale": torch.tensor(0.42)}
        ckpt = {
            "MODEL_STATE": score_model.state_dict(),
            "EPOCHS": 5,
            "HISTORY": [1.0, 0.9],
            "EMA": ema_shadow,
        }
        torch.save(ckpt, checkpoint_path)

        optimizer = torch.optim.SGD(score_model.parameters(), lr=0.1)
        trainer = Trainer(
            model=score_model,
            optimizer=optimizer,
            file_path=file_path,
            device=torch.device("cpu"),
            **trainer_dirs,
        )

        assert trainer.epochs == 5
        assert trainer.history == [1.0, 0.9]
        assert torch.allclose(
            trainer.model.exponential_moving_average.shadow["network.scale"],
            ema_shadow["network.scale"],
        )
        assert trainer.checkpoint == checkpoint_path

    def test_loads_checkpoint_missing_ema_and_history_gracefully(self, trainer_dirs):
        # A bare nn.Module (no exponential_moving_average) and a checkpoint
        # with no "EMA"/"HISTORY" keys at all - _load_checkpoint must not
        # crash, and HISTORY should default to [].
        class NoEmaModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(2, 1)

            def forward(self, x):
                return self.lin(x)

        model = NoEmaModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        ckpt_dir = Path(trainer_dirs["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        file_path = "no_ema_ckpt"
        checkpoint_path = ckpt_dir / f"{file_path}_ckpt.pt"
        torch.save({"MODEL_STATE": model.state_dict(), "EPOCHS": 3}, checkpoint_path)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            file_path=file_path,
            device=torch.device("cpu"),
            **trainer_dirs,
        )

        assert trainer.epochs == 3
        assert trainer.history == []


# ---------------------------------------------------------------------------
# _save_checkpoint - direct unit tests
# ---------------------------------------------------------------------------


class TestSaveCheckpointDirect:
    def test_writes_expected_keys_and_history_file(self, trainer):
        trainer.history = [0.5, 0.4]
        trainer._save_checkpoint(epoch=7)

        assert trainer.checkpoint.exists()
        assert trainer.history_file.exists()

        loaded = torch.load(trainer.checkpoint, map_location="cpu", weights_only=True)
        assert "MODEL_STATE" in loaded
        assert loaded["EPOCHS"] == 7
        assert loaded["HISTORY"] == [0.5, 0.4]
        assert "EMA" in loaded

        with open(trainer.history_file) as f:
            hist = json.load(f)
        assert hist == trainer.history

    def test_omits_ema_key_when_model_has_no_ema(self, trainer_dirs):
        class NoEmaModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(2, 1)

            def forward(self, x):
                return self.lin(x)

        model = NoEmaModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            file_path="save_no_ema",
            device=torch.device("cpu"),
            **trainer_dirs,
        )

        trainer.history = [0.123]
        trainer._save_checkpoint(epoch=2)

        loaded = torch.load(trainer.checkpoint, map_location="cpu", weights_only=True)
        assert "MODEL_STATE" in loaded and "EPOCHS" in loaded and "HISTORY" in loaded
        assert "EMA" not in loaded


# ---------------------------------------------------------------------------
# LR scheduler call count
# ---------------------------------------------------------------------------


class CountingLRScheduler:
    """Minimal fake scheduler that just counts .step() calls, so we can
    check the exact per-batch call count rather than only "LR changed"."""

    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class TestSchedulerStepCount:
    def test_scheduler_stepped_once_per_batch(self, trainer, loader):
        lr_scheduler = CountingLRScheduler()
        trainer.train(loader, N_epochs=1, lr_scheduler=lr_scheduler)
        assert lr_scheduler.steps == len(loader)

    def test_scheduler_stepped_across_multiple_epochs(self, trainer, loader):
        lr_scheduler = CountingLRScheduler()
        trainer.train(loader, N_epochs=3, lr_scheduler=lr_scheduler)
        assert lr_scheduler.steps == len(loader) * 3


# ---------------------------------------------------------------------------
# DistributedSampler.set_epoch
# ---------------------------------------------------------------------------


class TestDistributedSamplerSetEpoch:
    def test_set_epoch_called_when_sampler_supports_it(self, trainer, monkeypatch):
        x = torch.randn(8, 3)
        ds = TensorDataset(x)
        # num_replicas/rank passed explicitly so this doesn't require an
        # initialized torch.distributed process group.
        sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True)

        called = {"flag": False}

        def fake_set_epoch(epoch):
            called["flag"] = True

        monkeypatch.setattr(sampler, "set_epoch", fake_set_epoch, raising=True)

        loader = DataLoader(ds, batch_size=4, sampler=sampler)
        trainer.train(loader, N_epochs=1)

        assert called["flag"] is True


# ---------------------------------------------------------------------------
# mps device forces use_ddp off
# ---------------------------------------------------------------------------


class TestMpsDisablesDdp:
    def test_mps_device_string_disables_ddp(
        self, score_model, trainer_dirs, monkeypatch
    ):
        # Avoid touching a real MPS device (not available in this test
        # environment) - patch _set_model to skip the actual .to(device)
        # call, since this test only cares about the use_ddp override in
        # __init__, not real device placement.
        def noop_set_model(self, model, optimizer):
            self.model = model
            self.optimizer = optimizer

        monkeypatch.setattr(Trainer, "_set_model", noop_set_model, raising=True)

        optimizer = torch.optim.SGD(score_model.parameters(), lr=0.1)
        trainer = Trainer(
            model=score_model,
            optimizer=optimizer,
            file_path="mps_test",
            device="mps",
            use_ddp=True,
            **trainer_dirs,
        )
        assert trainer.use_ddp is False


# ---------------------------------------------------------------------------
# DDP - logic-level tests via monkeypatching (no real process group needed)
# ---------------------------------------------------------------------------
#
# These exercise the DDP-specific *code paths* (wrapping, loss all_reduce,
# should_stop broadcast) in a single process by faking torch.distributed
# and the DDP wrapper itself. They do NOT exercise real multi-process
# semantics (actual NCCL/gloo communication, DDP's real gradient-sync
# hooks) - that still needs a genuine multi-GPU/multi-process job. But the
# logic that was actually the source of the deadlock bug (the broadcast
# fix) IS meaningfully covered this way, since the bug was about which
# ranks execute which branches, not about the transport layer.


class TestDDPWrapping:
    def test_set_model_wraps_with_ddp_on_init(
        self, score_model, trainer_dirs, monkeypatch
    ):
        calls = {}

        class DummyDDP:
            def __init__(self, module, device_ids=None):
                calls["wrapped"] = True
                self.module = module
                self.device_ids = device_ids

        monkeypatch.setattr("diffusion_models.trainer.DDP", DummyDDP)

        optimizer = torch.optim.SGD(score_model.parameters(), lr=0.1)
        trainer = Trainer(
            model=score_model,
            optimizer=optimizer,
            file_path="ddp_wrap",
            device=torch.device("cpu"),
            use_ddp=True,
            **trainer_dirs,
        )

        assert calls.get("wrapped") is True
        assert hasattr(trainer.model, "module")
        assert trainer.model.module is score_model

    def test_no_ddp_wrapping_when_disabled(
        self, score_model, trainer_dirs, monkeypatch
    ):
        calls = {}

        class DummyDDP:
            def __init__(self, module, device_ids=None):
                calls["wrapped"] = True

        monkeypatch.setattr("diffusion_models.trainer.DDP", DummyDDP)

        optimizer = torch.optim.SGD(score_model.parameters(), lr=0.1)
        Trainer(
            model=score_model,
            optimizer=optimizer,
            file_path="no_ddp_wrap",
            device=torch.device("cpu"),
            use_ddp=False,
            **trainer_dirs,
        )

        assert "wrapped" not in calls


class TestDDPTrainingBranches:
    def test_all_reduce_and_broadcast_called_when_ddp_active(
        self, score_model, trainer_dirs, loader, monkeypatch
    ):
        optimizer = torch.optim.SGD(score_model.parameters(), lr=0.1)
        trainer = Trainer(
            model=score_model,
            optimizer=optimizer,
            file_path="ddp_reduce",
            device=torch.device("cpu"),
            use_ddp=False,  # build normally first
            **trainer_dirs,
        )
        # Force the DDP branches in train() without a real process group -
        # fake "already DDP-wrapped" by giving self.model a .module.
        trainer.use_ddp = True
        trainer.world_size = 2
        trainer.model = SimpleNamespace(module=score_model)

        class DummyReduceOp:
            SUM = object()

        all_reduce_calls = []
        broadcast_calls = []

        def fake_all_reduce(tensor, op=None):
            all_reduce_calls.append(1)
            tensor.mul_(trainer.world_size)

        def fake_broadcast(tensor, src=0):
            broadcast_calls.append(1)

        monkeypatch.setattr(torch.distributed, "ReduceOp", DummyReduceOp)
        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
        monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)

        trainer.train(loader, N_epochs=1, early_stopping=1)

        assert len(all_reduce_calls) == len(loader)  # once per batch
        assert len(broadcast_calls) == 1  # once per epoch, for should_stop sync
        assert len(trainer.history) == 1

    def test_broadcast_called_every_epoch_not_just_on_stop(
        self, score_model, trainer_dirs, loader, monkeypatch
    ):
        # Regression-shaped test for the deadlock fix: the broadcast must
        # happen every epoch (so all ranks stay in sync), not only on the
        # epoch where rank 0 actually decides to stop.
        optimizer = torch.optim.SGD(score_model.parameters(), lr=0.1)
        trainer = Trainer(
            model=score_model,
            optimizer=optimizer,
            file_path="ddp_broadcast_every_epoch",
            device=torch.device("cpu"),
            use_ddp=False,
            **trainer_dirs,
        )
        trainer.use_ddp = True
        trainer.world_size = 2
        trainer.model = SimpleNamespace(module=score_model)

        class DummyReduceOp:
            SUM = object()

        broadcast_calls = []

        monkeypatch.setattr(torch.distributed, "ReduceOp", DummyReduceOp)
        monkeypatch.setattr(
            torch.distributed,
            "all_reduce",
            lambda tensor, op=None: tensor.mul_(trainer.world_size),
        )
        monkeypatch.setattr(
            torch.distributed,
            "broadcast",
            lambda tensor, src=0: broadcast_calls.append(1),
        )

        # early_stopping large enough that no epoch actually triggers a stop
        trainer.train(loader, N_epochs=3, early_stopping=100)

        assert len(broadcast_calls) == 3  # one per epoch actually run
        assert len(trainer.history) == 3
