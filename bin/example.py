import os
import json
import torch
import argparse
import numpy as np

from trainer import Trainer
from samplers import sampler
from utils import set_device, ddp_setup, destroy_ddp, grab

from torch.utils.data import DataLoader
from DiffusionModels import ScoreModel
from Nets import CNet, CCNet

CWD = os.getcwd()

class Dataset:
	"""
	This is an example of a custom dataset. It contains dummy data.
	"""
	def __init__(self, use_labels = False):
		self.use_labels = use_labels
		self.images, self.labels = self.init_dataset()
	
	def init_dataset(self):
		data = np.random.randn(1000, 1, 8, 8).astype(np.float32)
		labels = np.random.randint(0, 2, data.shape[0]).astype(np.float32)
		if self.use_labels:
			data[labels == 1] = np.random.rand(sum(labels==1), 1, 8, 8).astype(np.float32)
		return data, labels
	
	def __len__(self):
		return len(self.images)
	
	def __getitem__(self, idx):
		if self.use_labels:
			return self.images[idx], self.labels[idx]
		return self.images[idx]

parser = argparse.ArgumentParser(
	description="An example of training. This uses a CNN-based diffusion model."
)
parser.add_argument("--lr", type=float, default=1e-4, help="Training learning rate")
parser.add_argument("--max_epochs", type=int, default=2, help="Maximum number of epochs in training")
parser.add_argument("--batch_size", type=int, default=32, help="Traiing batch size")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers in loading batches. Recommended to keep to default")
parser.add_argument(
	"--file",
	type=str,
	default="example",
	help="Name of file to save model's weight in (Default: 'example'). By defaults, save a .pt file.",
)
parser.add_argument("--ddp", action=argparse.BooleanOptionalAction, default=False, help="Flag determines whether to use PyTorch's distributed learning. Default is False")

if __name__=="__main__":
	args = parser.parse_args()

	parameters = {
		"in_channels": 1,
		"marginal_prob_sigma": 25,
		"channels": [16, 16],
		"time_channels": 32,
		"beta_channels": 32,
		"batch_size": args.batch_size,
		"base_lr": args.lr,
		"N_epochs": args.max_epochs,
		"activation": "silu",
		"dropout_rate": 0.2,
		"padding_mode": "zeros",
	}

	model_filename = CWD + "/data/weights/" + args.file + "_weights.pt"
	param_filename = CWD + "/data/weights/" + args.file + "_params.json"
	with open(param_filename, "w") as fp:
		json.dump(parameters, fp)

	device = set_device("gpu") #Set the device you want to work on

# 	DATASET

	dataset = Dataset(use_labels=True) 
	dist_sampler = ddp_setup(dataset, args.ddp) #Set up backend and sampler for distributed learning
	loader = DataLoader(
		dataset=dataset, 
		batch_size=args.batch_size, 
		sampler=dist_sampler,
		shuffle=True,
		num_workers=args.num_workers
	)

# 	MODEL
	net = CCNet(
		in_channels=parameters["in_channels"],
		channels=parameters["channels"],
		time_channels=parameters["time_channels"],
		beta_channels=parameters["beta_channels"],
		activation=torch.nn.SiLU(),
		dropout_rate=parameters["dropout_rate"],
		padding_mode=parameters["padding_mode"],
		device = device
	)

	model = ScoreModel(
		network=net, 
		marginal_prob_sigma=parameters["marginal_prob_sigma"], 
		device=device
	)


# 	TRAINER
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	trainer = Trainer(
		model=model,
		optimizer=optimizer,
		file_path=model_filename,
		device=device,
		use_ddp=args.ddp,
	)

	print("Training model...")

	trainer.train(
		loader=loader,
		N_epochs=args.max_epochs,
		scheduler=None,
		early_stopping=10
	)

	destroy_ddp(args.ddp)


#	SAMPLING

	print("Generating samples...")
	batch_size = 500 #
	samples = []
	for beta in torch.linspace(0., 1., 5, device=device):
		label = beta.expand(batch_size)
		samples_ = grab(sampler(model, (batch_size, 1, 32, 32), 250, label, eps=1e-3))
		samples.append(samples_)
	samples = np.stack(samples)
	print("Saving samples...")
	np.save(CWD + "/data/" + args.file + "_samples.npy", samples)
	print("All done!")