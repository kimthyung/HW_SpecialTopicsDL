import os
import math
import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt


def select_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        try:
            if torch.cuda.is_available():
                a = torch.ones(1, device="cuda")
                b = torch.randn(1, device="cuda")
                _ = (a + b).item()
                return torch.device("cuda")
        except Exception:
            pass
        return torch.device("cpu")
    if torch.cuda.is_available():
        try:
            a = torch.ones(1, device="cuda")
            b = torch.randn(1, device="cuda")
            _ = (a + b).item()
            return torch.device("cuda")
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def standardize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-8
    x_std = (x - mean) / std
    return x_std.astype(np.float32), mean.squeeze(), std.squeeze()


def smooth_series(y: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    if len(y) == 0:
        return y
    out = np.zeros_like(y, dtype=np.float32)
    ema = 0.0
    for i, v in enumerate(y):
        ema = alpha * ema + (1.0 - alpha) * float(v)
        out[i] = ema
    return out


@dataclass
class BetaSchedule:
    T: int = 50
    beta_min: float = 1e-4
    beta_max: float = 2e-2

    def build(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        betas = torch.linspace(self.beta_min, self.beta_max, self.T, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return betas, alphas, alpha_bars


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, depth: int = 3, out_dim: int = 2):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.ReLU())
            dim = hidden
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_dino_tsv(path: str) -> np.ndarray:
    raw = pd.read_csv(path, sep="\t")
    if {"dataset", "x", "y"}.issubset(set(raw.columns)):
        dino = raw[raw["dataset"] == "dino"][ ["x", "y"] ].to_numpy()
    else:
        num_cols = [c for c in raw.columns if np.issubdtype(raw[c].dtype, np.number)]
        dino = raw[num_cols[:2]].to_numpy()
    return dino


def make_loaders(x: np.ndarray, batch_size: int = 256, val_frac: float = 0.15) -> Tuple[DataLoader, DataLoader]:
    x_tensor = torch.from_numpy(x)
    n = x_tensor.shape[0]
    n_val = max(1, int(n * val_frac))
    train = TensorDataset(x_tensor[:-n_val]) if n_val < n else TensorDataset(x_tensor)
    val = TensorDataset(x_tensor[-n_val:]) if n_val < n else TensorDataset(x_tensor[:1])
    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(val, batch_size=batch_size, shuffle=False)


def q_sample(x0: torch.Tensor, alpha_bar_t: torch.Tensor):
    eps = torch.randn_like(x0)
    return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * eps, eps


def train_notime(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 betas: torch.Tensor, alphas: torch.Tensor, alpha_bars: torch.Tensor,
                 device: torch.device, epochs: int, lr: float, clip_grad: float,
                 verbose: bool, log_interval: int):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    T = betas.shape[0]
    tr_hist, va_hist = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        ep = []
        for (x0_batch,) in train_loader:
            B = x0_batch.shape[0]
            x0 = x0_batch.to(device)
            t = torch.randint(1, T + 1, (B,), device=device)
            alpha_bar_t = alpha_bars[t - 1].view(B, 1)
            x_t, eps = q_sample(x0, alpha_bar_t)
            pred = model(x_t)
            loss = mse(pred, eps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            opt.step()
            ep.append(loss.detach().item())
        tr_hist.append(float(np.mean(ep)))

        model.eval()
        with torch.inference_mode():
            epv = []
            for (x0_batch,) in val_loader:
                B = x0_batch.shape[0]
                x0 = x0_batch.to(device)
                t = torch.randint(1, T + 1, (B,), device=device)
                alpha_bar_t = alpha_bars[t - 1].view(B, 1)
                x_t, eps = q_sample(x0, alpha_bar_t)
                pred = model(x_t)
                loss = mse(pred, eps)
                epv.append(loss.detach().item())
            va_hist.append(float(np.mean(epv)))
        if verbose and (epoch % log_interval == 0 or epoch == 1 or epoch == epochs):
            print(f"epoch {epoch:4d}/{epochs}  train_mse={tr_hist[-1]:.6f}  val_mse={va_hist[-1]:.6f}")
    return np.array(tr_hist, dtype=np.float32), np.array(va_hist, dtype=np.float32)


@torch.no_grad()
def ancestral_sample(model: nn.Module, betas: torch.Tensor, alphas: torch.Tensor, alpha_bars: torch.Tensor,
                     num_samples: int, device: torch.device):
    T = betas.shape[0]
    x = torch.randn(num_samples, 2, device=device)
    for t_idx in reversed(range(1, T + 1)):
        beta_t = betas[t_idx - 1]
        alpha_t = alphas[t_idx - 1]
        alpha_bar_t = alpha_bars[t_idx - 1]
        eps = model(x)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps)
        if t_idx > 1:
            z = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * z
        else:
            x = mean
    return x.detach().cpu().numpy().astype(np.float32)


def plot_scatter(pts: np.ndarray, title: str, out_path: str):
    plt.figure(figsize=(4, 4))
    plt.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.85)
    plt.axis("off")
    plt.title(title)
    plt.gca().set_aspect("equal", "box")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_betas(betas: torch.Tensor, out_path: str):
    plt.figure(figsize=(5, 3))
    plt.plot(range(1, len(betas) + 1), betas.cpu().numpy())
    plt.xlabel("timestep"); plt.ylabel("beta_t"); plt.title("Linear beta schedule")
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def plot_alpha_bar(alpha_bars: torch.Tensor, out_path: str):
    plt.figure(figsize=(5, 3))
    plt.plot(range(1, len(alpha_bars) + 1), alpha_bars.cpu().numpy())
    plt.xlabel("timestep"); plt.ylabel("alpha_bar_t"); plt.title("alpha_bar")
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def build_alpha_bar_curve(T: int, beta_min: float, beta_max: float) -> np.ndarray:
    schedule = BetaSchedule(T=T, beta_min=beta_min, beta_max=beta_max)
    _, _, alpha_bars = schedule.build()
    return alpha_bars.cpu().numpy()


def plot_alpha_bar_compare(T: int, configs: list, out_path: str):
    plt.figure(figsize=(6.5, 3.5))
    timesteps = range(1, T + 1)
    for label, bmin, bmax in configs:
        curve = build_alpha_bar_curve(T, bmin, bmax)
        plt.plot(timesteps, curve, label=f"{label} ({bmin:.0e},{bmax:.0e})")
    plt.xlabel("timestep"); plt.ylabel("alpha_bar_t"); plt.title("alpha_bar compare")
    plt.legend()
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def plot_noise_power_compare(T: int, configs: list, out_path: str):
    plt.figure(figsize=(6.5, 3.5))
    timesteps = range(1, T + 1)
    for label, bmin, bmax in configs:
        curve = build_alpha_bar_curve(T, bmin, bmax)
        noise_power = 1.0 - curve
        plt.plot(timesteps, noise_power, label=f"{label} ({bmin:.0e},{bmax:.0e})")
    plt.xlabel("timestep"); plt.ylabel("1 - alpha_bar_t")
    plt.title("noise power compare (higher = more destroyed)")
    plt.legend()
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def plot_forward_panels(x0: np.ndarray, alpha_bars: torch.Tensor, timesteps: Tuple[int, ...], out_path: str, seed: int = 42):
    rng = np.random.default_rng(seed)
    x0_t = torch.from_numpy(x0)
    fig, axs = plt.subplots(1, len(timesteps), figsize=(4 * len(timesteps), 4))
    if len(timesteps) == 1:
        axs = [axs]
    for ax, t in zip(axs, timesteps):
        eps = torch.from_numpy(rng.standard_normal(size=x0.shape).astype(np.float32))
        a_bar = alpha_bars[t - 1].item()
        x_t = math.sqrt(a_bar) * x0_t + math.sqrt(1 - a_bar) * eps
        pts = x_t.numpy()
        ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.85)
        ax.set_title(f"t={t}"); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal", "box")
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/kth8606/DeepHome/DatasaurusDozen.tsv")
    parser.add_argument("--out_dir", type=str, default="/home/kth8606/DeepHome/outputs/ddpm_dinosaur_notime")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta_min", type=float, default=1e-4)
    parser.add_argument("--beta_max", type=float, default=2e-2)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    ensure_dir(args.out_dir)

    x = load_dino_tsv(args.data_path)
    x_std, _, _ = standardize(x)
    plot_scatter(x_std, "Standardized Dinosaur", os.path.join(args.out_dir, "dino_standardized.png"))

    schedule = BetaSchedule(T=args.T, beta_min=args.beta_min, beta_max=args.beta_max)
    betas, alphas, alpha_bars = schedule.build()
    plot_betas(betas, os.path.join(args.out_dir, "betas.png"))
    plot_alpha_bar(alpha_bars, os.path.join(args.out_dir, "alpha_bar.png"))
    plot_forward_panels(x_std, alpha_bars, (1, 6, 12, 25, args.T), os.path.join(args.out_dir, "forward_panels.png"), seed=args.seed)

    train_loader, val_loader = make_loaders(x_std, batch_size=args.batch_size)

    model = MLP(in_dim=2, hidden=args.width, depth=3, out_dim=2)
    tr, va = train_notime(model, train_loader, val_loader, betas.to(device), alphas.to(device), alpha_bars.to(device),
                          device, epochs=args.epochs, lr=args.lr, clip_grad=1.0, verbose=args.verbose, log_interval=args.log_interval)

    # plot loss
    plt.figure(figsize=(5.5, 3.5))
    plt.plot(tr, label="train", alpha=0.5)
    plt.plot(va, label="val", alpha=0.8)
    if len(va) > 2:
        plt.plot(smooth_series(va, alpha=0.9), label="val (EMA)", linewidth=2)
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("No-time denoiser loss")
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "loss_notime.png")); plt.close()

    # sampling
    model.eval()
    samples = ancestral_sample(model.to(device), betas.to(device), alphas.to(device), alpha_bars.to(device), args.num_samples, device)
    plot_scatter(samples, "Samples (no time)", os.path.join(args.out_dir, "samples_notime.png"))

    # compare schedules: gentle vs default vs aggressive
    compare_cfgs = [
        ("gentle", 1e-4, 2e-3),
        ("default", float(args.beta_min), float(args.beta_max)),
        ("aggressive", 5e-3, 5e-2),
    ]
    plot_alpha_bar_compare(args.T, compare_cfgs, os.path.join(args.out_dir, "alpha_bar_compare.png"))
    plot_noise_power_compare(args.T, compare_cfgs, os.path.join(args.out_dir, "noise_power_compare.png"))

    print("Done. Figures saved to:", args.out_dir)


if __name__ == "__main__":
    main()


