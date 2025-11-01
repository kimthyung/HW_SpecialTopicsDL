import os
from typing import Tuple

import torch
from torch import nn


class ConvVAE(nn.Module):
	def __init__(self, latent_dim: int = 128):
		super().__init__()
		self.latent_dim = latent_dim

		# Encoder: 96x96 -> 3x3 via five stride-2 conv layers
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 96 -> 48
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),

			nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 48 -> 24
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),

			nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 24 -> 12
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),

			nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 12 -> 6
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),

			nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 6 -> 3
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)

		# Compute flattened feature size dynamically for safety (expects 96x96 input)
		with torch.no_grad():
			dummy = torch.zeros(1, 3, 96, 96)
			enc_out = self.encoder(dummy)
			self._enc_shape = enc_out.shape[1:]  # (C, H, W)
			self.flattened_size = int(enc_out.numel())

		self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
		self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

		self.fc_decode = nn.Linear(latent_dim, self.flattened_size)

		# Decoder mirrors the encoder using ConvTranspose2d
		C, H, W = self._enc_shape
		assert (C, H, W) == (512, 3, 3), "This architecture assumes 96x96 input -> 3x3x512 features."

		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # 3 -> 6
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),

			nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 6 -> 12
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),

			nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 12 -> 24
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),

			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 24 -> 48
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),

			nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),     # 48 -> 96
			nn.Sigmoid(),  # output in [0,1]
		)

	def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		features = self.encoder(x)
		flattened = features.view(features.size(0), -1)
		mu = self.fc_mu(flattened)
		logvar = self.fc_logvar(flattened)
		return mu, logvar

	@staticmethod
	def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std

	def decode(self, z: torch.Tensor) -> torch.Tensor:
		h = self.fc_decode(z)
		h = h.view(h.size(0), *self._enc_shape)
		x_recon = self.decoder(h)
		return x_recon

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		recon = self.decode(z)
		return recon, mu, logvar

	@staticmethod
	def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
		# KL divergence between N(mu, sigma^2) and N(0, I)
		# 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar )
		return 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=1)  # per-sample
