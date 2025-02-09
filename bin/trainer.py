import torch, os
import torch.distributed as dist

from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import trange

CWD = os.getcwd()

class Trainer:
	"""
	Creates a trainer for a score-based diffusion model.

	Args:
		model (torch.nn.Module): the score model class to be trained.
		loader (DataLoader): loader of the training data.
		optimizer (torch.optim.Optimizer): optimisation scheme for training.
		params (dict): parameters of the model, e.g. batch_size, epochs, hidden layers, channels etc.
		file_path (str): The path to the file the model state is going to be saved.
		device (str): The device used to train the model e.g. CPU, GPU, MPS. Defaults to CPUs
	"""

	def __init__(
		self,
		model: torch.nn.Module,
		optimizer: torch.optim.Optimizer,
		file_path: str,
		device: torch.device,
		use_ddp: bool = False,
		checkpoint: str = CWD + "/data/weights/checkpoint.pt"
	) -> None:
		self.rank = int(os.environ.get("LOCAL_RANK", 0))
		self.world_size = int(os.environ.get("WORLD_SIZE", 1))
		self.file_path = file_path
		self.checkpoint = checkpoint
		self.epochs = 0
		self.device = device
		self.ckpt_freq = 10
		self.dims = None
		self.use_ddp = False if str(device) == 'mps' else use_ddp
		self._set_model(model, optimizer)

	def _set_model(
		self,
		model: torch.nn.Module,
		optimizer: torch.optim.Optimizer,
	) -> None:
		
		self.model = model
		self.optimizer = optimizer
		self.history = []

		if os.path.exists(self.checkpoint):
			self._load_checkpoint()

		if self.use_ddp:
			self.model = DDP(self.model, device_ids=[self.rank])

	def _load_checkpoint(self):
		ckpt = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
		self.model.load_state_dict(ckpt["MODEL_STATE"])
		self.epochs = ckpt["EPOCHS"]
		self.history = ckpt["HISTORY"]
		if self.rank == 0:
			print("Loading checkpoint")
			print(f"Training continues from checkpoint at epoch {self.epochs}...")

	def _save_checkpoint(self, epoch):
		model = self.model.module if self.use_ddp else self.model
		checkpoint = {"MODEL_STATE": model.state_dict(), "EPOCHS": epoch, "HISTORY": self.history}
		torch.save(checkpoint, self.checkpoint)
		print(f"Epoch {epoch} | Checkpoint saved at {self.checkpoint}")


	def train(self, loader: DataLoader, N_epochs: int, scheduler=None, early_stopping: int = 10):
		model = self.model.module if self.use_ddp else self.model
		model.train()
		tqdm_epoch = trange(self.epochs, N_epochs)
		best_loss = float("inf")
		counter = 0
		labels = []

		for epoch in tqdm_epoch:
			epoch_loss = 0
			num_items = 0

			# Sampler shuffles data between epochs
			if isinstance(loader.sampler, DistributedSampler):
				loader.sampler.set_epoch(epoch)

			for batch in loader:
				if isinstance(batch, list):
					batch, labels = batch[0], batch[1:]

				batch = batch.to(self.device)
				labels = [l.to(self.device) for l in labels]

				loss = model.train_step(batch, self.optimizer, *labels, scheduler=scheduler)

				if self.use_ddp:
					# Reduce loss across all processes to get avg loss
					reduced_loss = loss.clone()
					handle = dist.reduce(reduced_loss, dst=0, op=dist.ReduceOp.SUM, async_op=True)
					handle.wait()
					loss = reduced_loss / self.world_size

				if self.rank == 0:
					epoch_loss += loss.item() * batch.shape[0]
					num_items += batch.shape[0]
			if self.rank == 0:
				current_loss = epoch_loss / num_items
				# self.history.append(current_loss)
				log_string = f"Average Loss: {current_loss:5f}"
				self.history.append(current_loss)

				if epoch % self.ckpt_freq == 0:
					self._save_checkpoint(epoch)

				if best_loss > current_loss:
					counter = 0
					best_loss = current_loss
					torch.save(model.state_dict(), self.file_path)
					log_string += " ---> Best model so far (stored)"
				else:
					counter += 1

				tqdm_epoch.set_description(log_string)
				if counter == early_stopping:
					print(
						f"Stopping training at {epoch:g} epoch(s) ! Best loss: {best_loss: .5f}"
					)
					break