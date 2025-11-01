import argparse
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from vae_stl10.data import get_stl10_loaders
from vae_stl10.models import ConvVAE
from vae_stl10.utils import save_image_grid


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train ConvVAE on STL-10")
	parser.add_argument("--epochs", type=int, default=50)
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--latent-dim", type=int, default=128)
	parser.add_argument("--recon-loss", type=str, default="bce", choices=["bce", "mse"])
	parser.add_argument("--beta", type=float, default=1.0, help="Weight for KL term (beta-VAE)")
	parser.add_argument("--kl-warmup-epochs", type=int, default=0, help="Linearly warm up beta over these epochs")
	parser.add_argument("--normalize-recon", action="store_true", help="Normalize reconstruction loss by pixels (per-sample mean)")
	parser.add_argument("--split", type=str, default="train", choices=["train", "train+unlabeled"]) 
	parser.add_argument("--device", type=str, default=None)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--out-dir", type=str, default="vae_stl10/outputs")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--sample-count", type=int, default=10)
	return parser.parse_args()


def set_seed(seed: int) -> None:
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def reconstruction_loss(x_recon: torch.Tensor, x: torch.Tensor, kind: str) -> torch.Tensor:
	# returns per-sample reconstruction loss (sum over pixels/channels)
	if kind == "bce":
		loss = F.binary_cross_entropy(x_recon, x, reduction="none")
	elif kind == "mse":
		loss = F.mse_loss(x_recon, x, reduction="none")
	else:
		raise ValueError(f"Unknown recon loss: {kind}")
	return loss.view(loss.size(0), -1).sum(dim=1)


def train_epoch(model: ConvVAE, loader, optimizer, device, recon_kind: str, normalize_recon: bool, beta: float) -> Dict[str, float]:
	model.train()
	recon_meter: List[float] = []
	kl_meter: List[float] = []
	total_meter: List[float] = []
	for images, _ in tqdm(loader, desc="Train", leave=False):
		images = images.to(device, non_blocking=True)
		recon, mu, logvar = model(images)
		recon_loss = reconstruction_loss(recon, images, recon_kind)  # (N,)
		if normalize_recon:
			# Per-sample mean over pixels/channels to keep scale comparable to KL
			numel_per_sample = images.size(1) * images.size(2) * images.size(3)
			recon_loss = recon_loss / float(numel_per_sample)
		kl_loss = model.kl_divergence(mu, logvar)  # (N,)
		total = recon_loss + beta * kl_loss

		loss = total.mean()
		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

		recon_meter.append(recon_loss.mean().item())
		kl_meter.append(kl_loss.mean().item())
		total_meter.append(loss.item())

	return {
		"recon": sum(recon_meter) / max(1, len(recon_meter)),
		"kl": sum(kl_meter) / max(1, len(kl_meter)),
		"total": sum(total_meter) / max(1, len(total_meter)),
	}


def plot_losses(history: Dict[str, List[float]], out_path: str) -> None:
	plt.figure(figsize=(7, 5))
	plt.plot(history["recon"], label="Reconstruction")
	plt.plot(history["kl"], label="KL")
	plt.plot(history["total"], label="Total")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("VAE Losses")
	plt.legend()
	plt.tight_layout()
	Path(out_path).parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(out_path)
	plt.close()


def save_reconstructions(model: ConvVAE, loader, device, out_path: str, count: int = 10) -> None:
	model.eval()
	with torch.no_grad():
		for images, _ in loader:
			images = images.to(device)
			recon, _, _ = model(images)
			orig = images[:count].cpu()
			rec = recon[:count].cpu()
			stack = torch.cat([orig, rec], dim=0)  # two rows when nrow=count
			save_image_grid(stack, out_path, nrow=count)
			break


def save_random_samples(model: ConvVAE, device, out_path: str, latent_dim: int, count: int = 10) -> None:
	model.eval()
	with torch.no_grad():
		z = torch.randn(count, latent_dim, device=device)
		images = model.decode(z).cpu()
		save_image_grid(images, out_path, nrow=count)


def main() -> None:
	args = parse_args()
	set_seed(args.seed)
	device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	train_loader, test_loader = get_stl10_loaders(
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		split=args.split,
	)

	model = ConvVAE(latent_dim=args.latent_dim).to(device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	history = {"recon": [], "kl": [], "total": []}
	for epoch in range(1, args.epochs + 1):
		# KL warm-up schedule for current epoch
		current_beta = args.beta
		if args.kl_warmup_epochs > 0:
			scale = min(1.0, epoch / float(args.kl_warmup_epochs))
			current_beta = args.beta * scale

		stats = train_epoch(
			model,
			train_loader,
			optimizer,
			device,
			args.recon_loss,
			args.normalize_recon,
			current_beta,
		)
		for k in history:
			history[k].append(stats[k])
		print(f"Epoch {epoch:03d} | recon={stats['recon']:.2f} kl={stats['kl']:.2f} total={stats['total']:.2f} beta={current_beta:.3f}")

	plot_losses(history, str(out_dir / "loss_curve.png"))

	# Save final reconstructions and samples
	save_reconstructions(model, test_loader, device, str(out_dir / "reconstructions_final.png"), count=min(10, args.sample_count))
	save_random_samples(model, device, str(out_dir / "samples_final.png"), latent_dim=args.latent_dim, count=args.sample_count)

	# Save model
	torch.save({
		"model_state_dict": model.state_dict(),
		"latent_dim": args.latent_dim,
		"history": history,
	}, str(out_dir / "model_final.pt"))
	print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
	main()
