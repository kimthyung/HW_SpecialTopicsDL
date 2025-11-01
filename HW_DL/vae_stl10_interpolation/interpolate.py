import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torchvision import datasets, transforms

from vae_stl10.models import ConvVAE
from vae_stl10.utils import save_image_grid


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Latent space interpolation for STL-10 VAE")
	parser.add_argument("--ckpt", type=str, required=True, help="Path to model_final.pt")
	parser.add_argument("--out-dir", type=str, required=True, help="Output directory for images")
	parser.add_argument("--latent-dim", type=int, default=None, help="If not in checkpoint, specify latent dim")
	parser.add_argument("--split", type=str, default="test", choices=["test", "train"]) 
	parser.add_argument("--device", type=str, default=None)
	parser.add_argument("--alphas", type=float, nargs="*", default=[0.0, 0.25, 0.5, 0.75, 1.0])
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--label-top", type=str, default="A->B", help="Label for the top row")
	parser.add_argument("--label-bottom", type=str, default="C->D", help="Label for the bottom row")
	return parser.parse_args()


def set_seed(seed: int) -> None:
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def load_checkpoint(ckpt_path: str) -> Tuple[dict, int]:
	ckpt = torch.load(ckpt_path, map_location="cpu")
	latent_dim = ckpt.get("latent_dim")
	return ckpt, latent_dim


def pick_two_different_classes(dataset) -> Tuple[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int]]:
	# STL10 returns (image, label). We'll scan until we find two distinct labels.
	first = None
	seen = {}
	for img, label in dataset:
		if label not in seen:
			seen[label] = img
			if first is None:
				first = (img, label)
			elif label != first[1]:
				return first, (img, label)
	raise RuntimeError("Could not find two different classes in the split.")


def pick_two_different_classes_random(dataset, seed: int) -> Tuple[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int]]:
	"""Randomly pick two images from different classes using RNG seed."""
	n = len(dataset)
	gen = torch.Generator()
	gen.manual_seed(seed)
	for _ in range(10000):
		i = int(torch.randint(0, n, (1,), generator=gen).item())
		j = int(torch.randint(0, n, (1,), generator=gen).item())
		if i == j:
			continue
		img_i, label_i = dataset[i]
		img_j, label_j = dataset[j]
		if int(label_i) != int(label_j):
			return (img_i, int(label_i)), (img_j, int(label_j))
	raise RuntimeError("Could not randomly find two different classes.")


def encode(model: ConvVAE, device, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	model.eval()
	with torch.no_grad():
		x = x.to(device)
		mu, logvar = model.encode(x)
		z = model.reparameterize(mu, logvar)
		return z, mu, logvar


def main() -> None:
	args = parse_args()
	set_seed(args.seed)
	device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

	ckpt, ckpt_latent = load_checkpoint(args.ckpt)
	latent_dim = args.latent_dim or ckpt_latent
	if latent_dim is None:
		raise ValueError("latent_dim is not in checkpoint; please pass --latent-dim")

	model = ConvVAE(latent_dim=latent_dim).to(device)
	model.load_state_dict(ckpt["model_state_dict"])
	model.eval()

	transform = transforms.Compose([transforms.ToTensor()])
	dataset = datasets.STL10(root="~/.torch", split=args.split, download=True, transform=transform)

	(a_img, a_label), (b_img, b_label) = pick_two_different_classes(dataset)
	print(f"Top pair labels: A={a_label}, B={b_label}")

	# Pick a second independent pair (C, D) from different classes, randomly
	(c_img, c_label), (d_img, d_label) = pick_two_different_classes_random(dataset, seed=args.seed + 1)
	print(f"Bottom pair labels: C={c_label}, D={d_label}")

	# Add batch dimension (top row pair)
	a_img_b = a_img.unsqueeze(0)
	b_img_b = b_img.unsqueeze(0)
	# Add batch dimension (bottom row pair)
	c_img_b = c_img.unsqueeze(0)
	d_img_b = d_img.unsqueeze(0)

	z_a, _, _ = encode(model, device, a_img_b)
	z_b, _, _ = encode(model, device, b_img_b)
	z_c, _, _ = encode(model, device, c_img_b)
	z_d, _, _ = encode(model, device, d_img_b)

	# Interpolate A->B (top) and C->D (bottom), generate images for specified alphas
	images_rows: List[torch.Tensor] = []
	row_a_to_b = []
	row_c_to_d = []
	with torch.no_grad():
		for alpha in args.alphas:
			za = (1 - alpha) * z_a + alpha * z_b
			img_a2b = model.decode(za).cpu()
			row_a_to_b.append(img_a2b)
			zc = (1 - alpha) * z_c + alpha * z_d
			img_c2d = model.decode(zc).cpu()
			row_c_to_d.append(img_c2d)

	row_a_to_b = torch.cat(row_a_to_b, dim=0)  # N x C x H x W
	row_c_to_d = torch.cat(row_c_to_d, dim=0)
	grid = torch.cat([row_a_to_b, row_c_to_d], dim=0)

	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	# Save original source images for each row (A, B, C, D)
	# Top pair (A, B)
	top_pair = torch.cat([a_img_b.cpu(), b_img_b.cpu()], dim=0)
	save_image_grid(top_pair, str(out_dir / "orig_top_pair.png"), nrow=2)
	# Bottom pair (C, D)
	bottom_pair = torch.cat([c_img_b.cpu(), d_img_b.cpu()], dim=0)
	save_image_grid(bottom_pair, str(out_dir / "orig_bottom_pair.png"), nrow=2)
	# Individual originals
	save_image_grid(a_img_b.cpu(), str(out_dir / "orig_A.png"), nrow=1)
	save_image_grid(b_img_b.cpu(), str(out_dir / "orig_B.png"), nrow=1)
	save_image_grid(c_img_b.cpu(), str(out_dir / "orig_C.png"), nrow=1)
	save_image_grid(d_img_b.cpu(), str(out_dir / "orig_D.png"), nrow=1)
	out_path = out_dir / "interpolation_grid.png"
	save_image_grid(grid, str(out_path), nrow=len(args.alphas))
	print(f"Saved {out_path}")
	print(f"Row labels: top='{args.label_top}', bottom='{args.label_bottom}'")


if __name__ == "__main__":
	main()
