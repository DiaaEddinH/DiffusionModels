import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from diffusion_models.trainer import Trainer


class DummyEMA:
    def __init__(self, param):
        # Keep a tensor as shadow to mimic real EMA buffers
        self.shadow = param.detach().clone()
        self.updates = 0

    def update(self):
        # Just count updates and slightly change the shadow to simulate movement
        self.updates += 1
        self.shadow = self.shadow + 0.0


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 1)
        self.ema = DummyEMA(next(self.lin.parameters()))
        self.saved_weights_paths = []

    def forward(self, x):
        return self.lin(x)

    def loss_fn(self, x, *labels):
        # Simple loss using model output so it has gradients
        out = self.forward(x)
        # include labels in graph if provided to exercise label device move path
        add = 0
        for lb in labels:
            add = add + lb.float().mean()
        return (out.mean() + 0.0 * add) ** 2

    # Trainer calls this when a new best model is found
    def _save_weights(self, path: Path):
        self.saved_weights_paths.append(Path(path))


class DummyScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def make_loader_with_labels(batch_size=2, n_batches=3):
    # Build a dataset that returns (x, y1, y2) so labels path is taken
    x = torch.randn(n_batches * batch_size, 4)
    y1 = torch.randn(n_batches * batch_size, 1)
    y2 = torch.randn(n_batches * batch_size, 1)
    ds = TensorDataset(x, y1, y2)
    loader = DataLoader(ds, batch_size=batch_size)
    return loader


def test_init_loads_existing_checkpoint_and_sets_paths(tmp_path):
    device = torch.device("cpu")

    # Prepare dirs and pre-create a checkpoint
    ckpt_dir = tmp_path / "ckpts"
    w_dir = tmp_path / "weights"
    ckpt_dir.mkdir()
    w_dir.mkdir()

    file_path = "unit"

    # Build a model and checkpoint matching its state
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    checkpoint_path = ckpt_dir / f"{file_path}_ckpt.pt"
    history_file = ckpt_dir / f"{file_path}_history.json"

    # Create a realistic checkpoint including EMA and history
    ckpt = {
        "MODEL_STATE": model.state_dict(),
        "EPOCHS": 5,
        "HISTORY": [1.0, 0.9],
        "EMA": model.ema.shadow,
    }
    torch.save(ckpt, checkpoint_path)

    # Now create trainer, it should load the checkpoint automatically
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        file_path=file_path,
        device=device,
        use_ddp=False,
        checkpoint_dir=str(ckpt_dir),
        weight_dir=str(w_dir),
    )

    # epochs & history should be restored
    assert trainer.epochs == 5
    assert trainer.history == [1.0, 0.9]

    # EMA shadow should also be restored
    assert torch.allclose((trainer.model.ema.shadow).float(), ckpt["EMA"].float())

    # Ensure paths are set correctly
    assert trainer.checkpoint == checkpoint_path
    assert trainer.weight_file == w_dir / f"{file_path}_weights.pt"
    assert trainer.history_file == history_file


def test_save_checkpoint_and_history_written(tmp_path):
    device = torch.device("cpu")

    ckpt_dir = tmp_path / "ckpts"
    w_dir = tmp_path / "weights"

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        file_path="unit",
        device=device,
        use_ddp=False,
        checkpoint_dir=str(ckpt_dir),
        weight_dir=str(w_dir),
    )

    # Populate some history first
    trainer.history = [0.5, 0.4]
    trainer._save_checkpoint(epoch=7)

    # Files exist
    assert trainer.checkpoint.exists()
    assert trainer.history_file.exists()

    # Check contents of checkpoint
    loaded = torch.load(trainer.checkpoint, map_location=device)
    assert "MODEL_STATE" in loaded and "EPOCHS" in loaded and "HISTORY" in loaded
    # EMA presence
    assert "EMA" in loaded

    # History file mirrors trainer.history
    with open(trainer.history_file) as f:
        hist = json.load(f)
    assert hist == trainer.history


def test_train_basic_with_labels_scheduler_and_checkpoint(tmp_path):
    device = torch.device("cpu")

    ckpt_dir = tmp_path / "ckpts"
    w_dir = tmp_path / "weights"

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = DummyScheduler()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        file_path="unit",
        device=device,
        use_ddp=False,
        checkpoint_dir=str(ckpt_dir),
        weight_dir=str(w_dir),
    )

    # Small loader that yields (x, y1, y2), labels path exercised
    loader = make_loader_with_labels(batch_size=2, n_batches=2)

    # Train for one epoch; early stopping window tiny so we still log once
    trainer.train(loader=loader, N_epochs=1, scheduler=scheduler, early_stopping=1)

    # History should have one entry
    assert len(trainer.history) == 1

    # Best model should have been saved at least once
    assert len(model.saved_weights_paths) >= 1
    assert model.saved_weights_paths[-1] == trainer.weight_file

    # Checkpoint should be created at epoch 0 (0 % ckpt_freq == 0)
    assert trainer.checkpoint.exists()

    # Scheduler stepped once per batch
    assert scheduler.steps == len(loader)


def test_train_calls_set_epoch_on_distributed_sampler(tmp_path, monkeypatch):
    device = torch.device("cpu")

    ckpt_dir = tmp_path / "ckpts"
    w_dir = tmp_path / "weights"

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        file_path="unit",
        device=device,
        use_ddp=False,
        checkpoint_dir=str(ckpt_dir),
        weight_dir=str(w_dir),
    )

    x = torch.randn(8, 4)
    ds = TensorDataset(x)
    sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True)

    called = {"flag": False}

    def fake_set_epoch(epoch):
        called["flag"] = True

    # Patch the sampler's set_epoch method to record that it was called
    monkeypatch.setattr(sampler, "set_epoch", fake_set_epoch, raising=True)

    loader = DataLoader(ds, batch_size=4, sampler=sampler)

    trainer.train(loader=loader, N_epochs=1, early_stopping=1)

    assert called["flag"] is True


def test_train_ddp_reduction_branch_monkeypatched(tmp_path, monkeypatch):
    device = torch.device("cpu")

    ckpt_dir = tmp_path / "ckpts"
    w_dir = tmp_path / "weights"

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        file_path="unit",
        device=device,
        use_ddp=False,  # avoid real DDP wrapping
        checkpoint_dir=str(ckpt_dir),
        weight_dir=str(w_dir),
    )

    # Force DDP branch without actual DDP initialization/wrapping
    trainer.use_ddp = True
    trainer.world_size = 2
    # Emulate DDP-wrapped model shape expected by Trainer by providing .module
    trainer.model = SimpleNamespace(module=model)

    # Monkeypatch torch.distributed symbols used in the code
    class DummyReduceOp:
        SUM = object()

    def fake_all_reduce(tensor, op=None):
        # Simulate sum across world_size by scaling the tensor in-place
        tensor.mul_(trainer.world_size)

    monkeypatch.setattr(torch.distributed, "ReduceOp", DummyReduceOp)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    # Simple loader
    x = torch.randn(6, 4)
    loader = DataLoader(TensorDataset(x), batch_size=3)

    # If reduction path runs, there should be no error; run one epoch
    trainer.train(loader=loader, N_epochs=1, early_stopping=1)

    # History still recorded on rank 0
    assert len(trainer.history) == 1


def test_set_model_wraps_with_ddp_on_init(tmp_path, monkeypatch):
    """_set_model should call DDP(...) when use_ddp=True (without real process group)."""
    calls = {}

    class DummyDDP:
        def __init__(self, module, device_ids=None):
            calls["wrapped"] = True
            self.module = module
            self.device_ids = device_ids

    # Patch DDP before constructing Trainer so _set_model uses our DummyDDP
    monkeypatch.setattr("diffusion_models.trainer.DDP", DummyDDP)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(1, 1)

        def forward(self, x):
            return self.lin(x)

    model = Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)

    t = Trainer(
        model=model,
        optimizer=opt,
        file_path="ddp",
        device=torch.device("cpu"),
        use_ddp=True,  # should trigger DummyDDP
        checkpoint_dir=str(tmp_path / "ck"),
        weight_dir=str(tmp_path / "w"),
    )

    assert calls.get("wrapped", False) is True
    # trainer.model should now be a DummyDDP wrapper exposing .module
    assert hasattr(t.model, "module")


def test_load_checkpoint_without_ema_and_history(tmp_path):
    """_load_checkpoint should tolerate checkpoints missing EMA/HISTORY and model lacking .ema."""

    class NoEMAModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 1)

        def forward(self, x):  # pragma: no cover - trivial
            return self.lin(x)

    model = NoEMAModel()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    ckdir = tmp_path / "ck"
    wdir = tmp_path / "w"
    ckdir.mkdir()
    wdir.mkdir()
    file_path = "noema"

    # Create checkpoint with NO EMA and NO HISTORY
    ckpt_path = ckdir / f"{file_path}_ckpt.pt"
    torch.save({"MODEL_STATE": model.state_dict(), "EPOCHS": 3}, ckpt_path)

    t = Trainer(
        model=model,
        optimizer=opt,
        file_path=file_path,
        device=torch.device("cpu"),
        use_ddp=False,
        checkpoint_dir=str(ckdir),
        weight_dir=str(wdir),
    )

    # epochs restored, history defaults to []
    assert t.epochs == 3
    assert t.history == []


def test_save_checkpoint_without_ema_field(tmp_path):
    """_save_checkpoint should not write EMA key if model has no .ema."""

    class NoEMAModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 1)

        def forward(self, x):  # pragma: no cover - trivial
            return self.lin(x)

    model = NoEMAModel()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    ckdir = tmp_path / "ck"
    wdir = tmp_path / "w"
    ckdir.mkdir()
    wdir.mkdir()

    t = Trainer(
        model=model,
        optimizer=opt,
        file_path="save_noema",
        device=torch.device("cpu"),
        use_ddp=False,
        checkpoint_dir=str(ckdir),
        weight_dir=str(wdir),
    )

    t.history = [0.123]
    t._save_checkpoint(epoch=2)

    loaded = torch.load(t.checkpoint, map_location="cpu")
    assert "MODEL_STATE" in loaded and "EPOCHS" in loaded and "HISTORY" in loaded
    assert "EMA" not in loaded  # no ema saved


def test_train_early_stopping_breaks(tmp_path):
    """Training should break when counter >= early_stopping (without accuracy bands)."""

    class ConstLossModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))  # ensures grads exist

        def forward(self, x):
            return x.sum(dim=1, keepdim=True) * 0.0 + self.p

        def loss_fn(self, x, *labels):
            out = self.forward(x)
            # constant (positive) loss; no improvement across epochs
            return out.mean() * 0.0 + 1.0

        def _save_weights(self, path):  # pragma: no cover - trivial side effect
            pass

    model = ConstLossModel()
    opt = torch.optim.SGD(model.parameters(), lr=1e-1)

    # tiny dataset
    x = torch.randn(6, 4)
    loader = DataLoader(TensorDataset(x), batch_size=3)

    t = Trainer(
        model=model,
        optimizer=opt,
        file_path="early",
        device=torch.device("cpu"),
        use_ddp=False,
        checkpoint_dir=str(tmp_path / "ck"),
        weight_dir=str(tmp_path / "w"),
    )

    # Expect: epoch 0 sets best; epoch 1 no improvement → counter==1 → break
    t.train(loader=loader, N_epochs=5, early_stopping=1, min_delta=1e-6)

    # Should have stopped after logging two epochs
    assert len(t.history) == 2


def test_mps_device_disables_ddp(tmp_path, monkeypatch):
    """Constructor should force-disable DDP when device string is 'mps' without requiring MPS runtime."""
    # Avoid touching the real device in _set_model (which would call model.to(device))
    from diffusion_models.trainer import Trainer

    def noop_set_model(self, model, optimizer):
        # minimal wiring so Trainer has attributes but never moves tensors/devices
        self.model = model
        self.optimizer = optimizer

    monkeypatch.setattr(Trainer, "_set_model", noop_set_model, raising=True)

    model = torch.nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Pass device as a string 'mps' so __init__ sees str(device) == 'mps'
    t = Trainer(
        model=model,
        optimizer=opt,
        file_path="mps",
        device="mps",
        use_ddp=True,  # requested, but should be turned off
        checkpoint_dir=str(tmp_path / "ck"),
        weight_dir=str(tmp_path / "w"),
    )
    assert t.use_ddp is False
