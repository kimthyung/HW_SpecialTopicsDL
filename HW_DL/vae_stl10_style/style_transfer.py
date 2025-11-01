import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vae_stl10.models import ConvVAE
from vae_stl10.utils import save_image_grid


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Latent style transfer for STL-10 VAE")
	parser.add_argument("--ckpt", type=str, required=True, help="Path to model_final.pt")
	parser.add_argument("--out-dir", type=str, required=True, help="Output directory for images")
	parser.add_argument("--latent-dim", type=int, default=None)
	parser.add_argument("--split", type=str, default="test", choices=["test", "train"]) 
	parser.add_argument("--device", type=str, default=None)
	parser.add_argument("--lambda-list", type=float, nargs="*", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
	parser.add_argument("--alpha-list", type=float, nargs="*", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--num-pairs", type=int, default=3, help="Number of A/B class pairs to process")
	return parser.parse_args()


def set_seed(seed: int) -> None:
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def load_checkpoint(ckpt_path: str) -> Tuple[dict, int]:
	ckpt = torch.load(ckpt_path, map_location="cpu")
	latent_dim = ckpt.get("latent_dim")
	return ckpt, latent_dim


def compute_class_means(model: ConvVAE, loader: DataLoader, device: str) -> Dict[int, torch.Tensor]:
	model.eval()
	label_to_sum: Dict[int, torch.Tensor] = defaultdict(lambda: None)
	label_to_count: Dict[int, int] = defaultdict(int)
	with torch.no_grad():
		for images, labels in loader:
			images = images.to(device)
			mu, logvar = model.encode(images)
			# Use mu as the embedding for mean computation
			for i in range(images.size(0)):
				label = int(labels[i].item())
				vec = mu[i].detach()
				if label_to_sum[label] is None:
					label_to_sum[label] = vec.clone()
				else:
					label_to_sum[label] += vec
				label_to_count[label] += 1
	class_means: Dict[int, torch.Tensor] = {}
	for label, s in label_to_sum.items():
		class_means[label] = s / max(1, label_to_count[label])
	return class_means


def pick_two_different_classes(dataset) -> Tuple[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int]]:
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


def pick_k_pairs(dataset, k: int) -> List[Tuple[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int]]]:
    label_to_image = {}
    for img, label in dataset:
        ilabel = int(label)
        if ilabel not in label_to_image:
            label_to_image[ilabel] = img
        # keep scanning full dataset to collect as many distinct labels as possible
    labels = sorted(label_to_image.keys())
    if len(labels) < 2:
        raise RuntimeError("Could not find two different classes in the split.")
    pairs: List[Tuple[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int]]] = []
    idx = 0
    while len(pairs) < k and (idx + 1) < len(labels):
        la = labels[idx]
        lb = labels[idx + 1]
        pairs.append(((label_to_image[la], la), (label_to_image[lb], lb)))
        idx += 2
    if len(pairs) < k:
        raise RuntimeError(f"Could not find {k} pairs from available classes (found {len(pairs)}).")
    return pairs


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
	loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

	# Compute class means on chosen split
	class_means = compute_class_means(model, loader, device)

	# Pick multiple A/B pairs from different classes
	pairs = pick_k_pairs(dataset, args.num_pairs)

	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	for pair_idx, ((a_img, a_label), (b_img, b_label)) in enumerate(pairs):
		print(f"Pair {pair_idx}: A={a_label}, B={b_label}")

		# Encode A and B (use mu as z)
		with torch.no_grad():
			a_mu, _ = model.encode(a_img.unsqueeze(0).to(device))
			b_mu, _ = model.encode(b_img.unsqueeze(0).to(device))
			z_a = a_mu
			z_b = b_mu

		z_mean_b = class_means[int(b_label)].unsqueeze(0).to(device)

		# Arithmetic: z_new = z_A + λ (z_B - z_mean)
		arith_images: List[torch.Tensor] = []
		with torch.no_grad():
			for lam in args.lambda_list:
				z_new = z_a + lam * (z_b - z_mean_b)
				img = model.decode(z_new).cpu()
				arith_images.append(img)
		arith_grid = torch.cat(arith_images, dim=0)

		# Interpolation: z'_new = z_A + α (z_B - z_A)
		interp_images: List[torch.Tensor] = []
		with torch.no_grad():
			for alpha in args.alpha_list:
				z_new = z_a + alpha * (z_b - z_a)
				img = model.decode(z_new).cpu()
				interp_images.append(img)
		interp_grid = torch.cat(interp_images, dim=0)

		# Save original images and grids for this pair
		save_image_grid(a_img.unsqueeze(0), str(out_dir / f"style_orig_A_label_{a_label}_p{pair_idx}.png"), nrow=1)
		save_image_grid(b_img.unsqueeze(0), str(out_dir / f"style_orig_B_label_{b_label}_p{pair_idx}.png"), nrow=1)
		save_image_grid(torch.stack([a_img, b_img], dim=0), str(out_dir / f"style_orig_A_B_pair_p{pair_idx}.png"), nrow=2)
		save_image_grid(arith_grid, str(out_dir / f"style_arithmetic_lambda_grid_p{pair_idx}.png"), nrow=len(args.lambda_list))
		save_image_grid(interp_grid, str(out_dir / f"style_interpolation_alpha_grid_p{pair_idx}.png"), nrow=len(args.alpha_list))

	print(f"Saved style grids to {out_dir}")


if __name__ == "__main__":
	main()
