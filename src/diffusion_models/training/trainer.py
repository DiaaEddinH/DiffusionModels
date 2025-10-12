import os
import json
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import trange


class Trainer:
    """
    Trainer for score-based diffusion models with optional DDP support.

    Args:
            model (torch.nn.Module): Model to train. Must implement `train_step`.
            optimizer (torch.optim.Optimizer): Optimizer.
            file_path (str): Base path for saving weights and checkpoints.
            device (torch.device): Training device.
            use_ddp (bool): Use DistributedDataParallel.
            checkpoint_dir (str): Directory for checkpoints.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        file_path: str,
        device: torch.device,
        use_ddp: bool = False,
        checkpoint_dir: str = "./data/checkpoints",
        weight_dir: str = "./data/weights",
    ) -> None:
        self.rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.file_path = file_path
        self.device = device
        self.ckpt_freq = 10
        self.use_ddp = False if str(device) == "mps" else use_ddp

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        weight_dir = Path(weight_dir)
        weight_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint = checkpoint_dir / f"{file_path}_ckpt.pt"
        self.weight_file = weight_dir / f"{file_path}_weights.pt"
        self.history_file = checkpoint_dir / f"{file_path}_history.json"

        self.epochs = 0
        self.history = []

        self._set_model(model, optimizer)

    # ------------------------------
    # Model setup and checkpointing
    # ------------------------------
    def _set_model(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> None:
        self.model = model.to(self.device)
        self.optimizer = optimizer

        if self.checkpoint.exists():
            self._load_checkpoint()

        if self.use_ddp:
            self.model = DDP(self.model, device_ids=[self.rank])

    def _load_checkpoint(self):
        ckpt = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
        model = self.model.module if self.use_ddp else self.model
        model.load_state_dict(ckpt["MODEL_STATE"])
        self.epochs = ckpt["EPOCHS"]
        self.history = ckpt.get("HISTORY", [])

        # EMA safety
        if "EMA" in ckpt and hasattr(model, "ema"):
            model.ema.shadow = ckpt["EMA"]

        if self.rank == 0:
            print(f"Loaded checkpoint: continuing from epoch {self.epochs}")

    def _save_checkpoint(self, epoch):
        model = self.model.module if self.use_ddp else self.model
        checkpoint = {
            "MODEL_STATE": model.state_dict(),
            "EPOCHS": epoch,
            "HISTORY": self.history,
        }
        if hasattr(model, "ema"):
            checkpoint["EMA"] = model.ema.shadow

        torch.save(checkpoint, self.checkpoint)
        if self.rank == 0:
            print(f"Epoch {epoch} | Checkpoint saved at {self.checkpoint}")

        # Save history separately for safety
        if self.rank == 0:
            with open(self.history_file, "w") as f:
                json.dump(self.history, f)

    # ------------------------------
    # Training Loop
    # ------------------------------
    def train(
        self,
        loader: DataLoader,
        N_epochs: int,
        scheduler=None,
        early_stopping: int = 10,
        min_delta: float = 1e-4,
    ):
        model = self.model.module if self.use_ddp else self.model
        model.train()
        tqdm_epoch = trange(self.epochs, N_epochs, disable=(self.rank != 0))

        best_loss = float("inf")
        counter = 0

        for epoch in tqdm_epoch:
            epoch_loss = 0
            num_items = 0

            # Shuffle between epochs if using DistributedSampler
            if isinstance(loader.sampler, DistributedSampler):
                loader.sampler.set_epoch(epoch)

            for batch in loader:
                labels = []
                if isinstance(batch, (list, tuple)):
                    batch, labels = batch[0], batch[1:]

                batch = batch.to(self.device)
                labels = [l.to(self.device) for l in labels]

                self.optimizer.zero_grad()

                loss = model.loss_fn(batch, *labels)

                loss.backward()
                self.optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                # EMA update if available
                if hasattr(model, "ema"):
                    model.ema.update()

                # DDP loss reduction
                if self.use_ddp:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss /= self.world_size

                # Logging only on rank 0
                if self.rank == 0:
                    epoch_loss += loss.item() * batch.shape[0]
                    num_items += batch.shape[0]

            if self.rank == 0 and num_items > 0:
                current_loss = epoch_loss / num_items
                self.history.append(current_loss)

                log_string = f"Average Loss: {current_loss:.6f}"

                if epoch % self.ckpt_freq == 0:
                    self._save_checkpoint(epoch)

                # Early stopping check
                if best_loss - current_loss > min_delta:
                    counter = 0
                    best_loss = current_loss
                    model._save_weights(self.weight_file)
                    log_string += " ---> Best model so far (stored)"
                else:
                    counter += 1

                tqdm_epoch.set_description(log_string)

                if counter >= early_stopping:
                    print(
                        f"Stopping training at epoch {epoch}! Best loss: {best_loss:.6f}"
                    )
                    break
