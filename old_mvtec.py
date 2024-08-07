import os
from glob import glob
from pathlib import Path

import yaml
import numpy as np
import torch.utils.data
from PIL import Image

import torch
# import pytorch_lightning as pl
from torchvision import transforms
from path_utils import get_path

# MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
# STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

ALL_CATEGORIES = [
	"carpet", "grid", "leather", "tile", "wood",
	"bottle", "cable", "capsule", "hazelnut", "metal_nut",
	"pill", "screw", "toothbrush", "transistor", "zipper"
]


def worker_init_fn(worker_id):
	np.random.seed(np.random.get_state()[1][0] + worker_id)


class MVTecDatamodule:
	def __init__(self, config):
		self.data_dir = str(get_path("data") / "mvtec")
		self.category = config.data.category
		self.input_size = config.data.image_size
		self.batch_size = config.training.batch_size

		self.train_dataset = MVTecDataset(self.data_dir, self.category, self.input_size, is_train=True)
		self.eval_dataset = MVTecDataset(self.data_dir, self.category, self.input_size, is_train=False)

	def train_dataloader(self):
		return torch.utils.data.DataLoader(
			self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=False,
			worker_init_fn=worker_init_fn
		)

	def eval_dataloader(self):
		return torch.utils.data.DataLoader(
			self.eval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=False,
			worker_init_fn=worker_init_fn
		)


class MVTecDataset(torch.utils.data.Dataset):
	def __init__(self, data_dir, category, input_size, is_train):
		self.is_train = is_train
		transform = [
			transforms.Resize(input_size),
			transforms.ToTensor(),
		]
		# if is_train:
		# 	transform.append(transforms.RandomHorizontalFlip())
		self.transform = transforms.Compose(transform)

		if is_train:
			self.image_files = list(glob(os.path.join(data_dir, category, "train", "good", "*.png")))
		else:
			self.image_files = sorted(glob(os.path.join(data_dir, category, "test", "*", "*.png")))

	def __getitem__(self, index):
		image_file = self.image_files[index]
		image = Image.open(image_file).convert('RGB')
		image = self.transform(image)

		if self.is_train:
			return {'image': image}
		else:
			if os.path.dirname(image_file).endswith("good"):
				target = torch.zeros([1, image.shape[-2], image.shape[-1]])
			else:
				target = Image.open(
					image_file.replace("test", "ground_truth").replace(
						".png", "_mask.png"
					)
				)
				target = self.transform(target)
			return {'image': image, 'target': target, 'image_path': image_file}

	def __len__(self):
		return len(self.image_files)
