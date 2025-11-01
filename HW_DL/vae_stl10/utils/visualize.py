from pathlib import Path
from typing import Optional

import torch
from torchvision.utils import save_image, make_grid


def save_image_grid(
	images: torch.Tensor,
	out_path: str,
	nrow: int = 5,
	normalize: bool = False,
	value_range: Optional[tuple] = None,
) -> None:
	"""Save a grid of images to disk.

	Args:
		images: tensor of shape (N, C, H, W), values in [0,1]
		out_path: file path to write (parent dirs created if needed)
		nrow: images per row
		normalize: whether to normalize via make_grid
		value_range: (min, max) if normalize=True
	"""
	Path(out_path).parent.mkdir(parents=True, exist_ok=True)
	grid = make_grid(images, nrow=nrow, normalize=normalize, value_range=value_range)
	save_image(grid, out_path)
