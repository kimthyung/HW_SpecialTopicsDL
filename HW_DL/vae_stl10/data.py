import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_stl10_loaders(
	batch_size: int = 128,
	num_workers: int = 4,
	split: str = "train",
) -> Tuple[DataLoader, DataLoader]:
	"""Create STL-10 train and test loaders.

	Args:
		batch_size: batch size
		num_workers: dataloader workers
		split: "train" or "train+unlabeled"

	Returns:
		(train_loader, test_loader)
	"""
	assert split in {"train", "train+unlabeled"}

	transform = transforms.Compose([
		transforms.ToTensor(),  # produces [0,1]
	])

	root = os.path.expanduser("~/.torch")

	train_dataset = datasets.STL10(
		root=root,
		split=split,
		download=True,
		transform=transform,
	)

	test_dataset = datasets.STL10(
		root=root,
		split="test",
		download=True,
		transform=transform,
	)

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=True,
		drop_last=True,
	)

	test_loader = DataLoader(
		test_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
		drop_last=False,
	)

	return train_loader, test_loader
