import os
import math
import argparse
import copy
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt


# -----------------------------
# Config and utils
# -----------------------------


def select_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        try:
            if torch.cuda.is_available():
                # try a tiny kernel op to ensure compatibility
                a = torch.ones(1, device="cuda")
                b = torch.randn(1, device="cuda")
                _ = (a + b).item()
                return torch.device("cuda")
        except Exception:
            pass
        return torch.device("cpu")
    # auto
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


# -----------------------------
# Beta schedule
# -----------------------------


@dataclass
class BetaSchedule:
    T: int = 100
    beta_min: float = 1e-4
    beta_max: float = 2e-2

    def build(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        betas = torch.linspace(self.beta_min, self.beta_max, self.T, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return betas, alphas, alpha_bars


# -----------------------------
# Models
# -----------------------------


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


def fourier_encode(v: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # v: (N, D), B: (L, D) random Gaussian, output: (N, 2L)
    # gamma(v) = [cos(2π B v), sin(2π B v)] using v @ B^T
    proj = v @ B.t()
    proj = 2 * math.pi * proj
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


def sinusoidal_time_embedding(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(1000.0), half, device=device)
    )
    t = t.float().unsqueeze(-1)  # (B, 1)
    args = t / freqs  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


# -----------------------------
# Data
# -----------------------------


def load_dino_tsv(path: str) -> np.ndarray:
    raw = pd.read_csv(path, sep="\t")
    if {"dataset", "x", "y"}.issubset(set(raw.columns)):
        dino = raw[raw["dataset"] == "dino"][["x", "y"]].to_numpy()
    else:
        num_cols = [c for c in raw.columns if np.issubdtype(raw[c].dtype, np.number)]
        dino = raw[num_cols[:2]].to_numpy()
    return dino


def make_loaders(x: np.ndarray, batch_size: int = 256, val_frac: float = 0.15) -> Tuple[DataLoader, DataLoader]:
    x_tensor = torch.from_numpy(x)
    ds = TensorDataset(x_tensor)
    n = len(ds)
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    # Simple split without shuffling to keep determinism consistent with seeding
    train_ds = TensorDataset(x_tensor[:n_train])
    val_ds = TensorDataset(x_tensor[n_train:])
    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train, val


# -----------------------------
# Forward process helpers
# -----------------------------


def q_sample(x0: torch.Tensor, alpha_bar_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    eps = torch.randn_like(x0)
    return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * eps, eps


# -----------------------------
# Training
# -----------------------------


def train_denoiser(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    device: torch.device,
    epochs: int = 300,
    lr: float = 1e-3,
    clip_grad: float = 1.0,
    time_dim: int = 0,
    time_embed: str = "none",
    verbose: bool = False,
    log_interval: int = 10,
    ema_decay: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ema_state = None
    if ema_decay > 0.0:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    mse = nn.MSELoss()
    T = betas.shape[0]

    train_losses = []
    val_losses = []
    best_val = float('inf')
    best_state = copy.deepcopy(model.state_dict())

    for epoch_idx in range(1, epochs + 1):
        model.train()
        epoch_train = []
        for (x0_batch,) in train_loader:
            B = x0_batch.shape[0]
            x0 = x0_batch.to(device)
            t = torch.randint(1, T + 1, (B,), device=device)
            alpha_bar_t = alpha_bars[t - 1].view(B, 1)
            x_t, eps = q_sample(x0, alpha_bar_t)

            if time_embed == "linear":
                t_norm = (t.float() - 1.0) / max(1, (T - 1))
                inp = torch.cat([x_t, t_norm.view(B, 1)], dim=1)
            elif time_embed == "sinusoidal":
                t_embed = sinusoidal_time_embedding(t, dim=time_dim)
                inp = torch.cat([x_t, t_embed], dim=1)
            else:
                inp = x_t

            eps_pred = model(inp)
            loss = mse(eps_pred, eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            opt.step()
            if ema_state is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)
            epoch_train.append(loss.detach().item())

        train_losses.append(float(np.mean(epoch_train)))

        # validation
        model.eval()
        with torch.inference_mode():
            epoch_val = []
            for (x0_batch,) in val_loader:
                B = x0_batch.shape[0]
                x0 = x0_batch.to(device)
                t = torch.randint(1, T + 1, (B,), device=device)
                alpha_bar_t = alpha_bars[t - 1].view(B, 1)
                x_t, eps = q_sample(x0, alpha_bar_t)
                if time_embed == "linear":
                    t_norm = (t.float() - 1.0) / max(1, (T - 1))
                    inp = torch.cat([x_t, t_norm.view(B, 1)], dim=1)
                elif time_embed == "sinusoidal":
                    t_embed = sinusoidal_time_embedding(t, dim=time_dim)
                    inp = torch.cat([x_t, t_embed], dim=1)
                else:
                    inp = x_t
                eps_pred = model(inp)
                loss = mse(eps_pred, eps)
                epoch_val.append(loss.detach().item())
            val_losses.append(float(np.mean(epoch_val)))

        # track best
        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch_idx % log_interval == 0 or epoch_idx == 1 or epoch_idx == epochs):
            print(f"epoch {epoch_idx:4d}/{epochs}  train_mse={train_losses[-1]:.6f}  val_mse={val_losses[-1]:.6f}")

    # prefer EMA state if enabled
    final_state = best_state
    if ema_state is not None:
        final_state = ema_state
    return np.array(train_losses, dtype=np.float32), np.array(val_losses, dtype=np.float32), final_state


# -----------------------------
# Sampling (ancestral)
# -----------------------------


@torch.no_grad()
def ancestral_sample(
    model: nn.Module,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    num_samples: int,
    device: torch.device,
    time_dim: int = 0,
    time_embed: str = "none",
) -> np.ndarray:
    T = betas.shape[0]
    x = torch.randn(num_samples, 2, device=device)
    for t_idx in reversed(range(1, T + 1)):
        beta_t = betas[t_idx - 1]
        alpha_t = alphas[t_idx - 1]
        alpha_bar_t = alpha_bars[t_idx - 1]
        if time_embed == "linear":
            t_tensor = torch.full((num_samples,), t_idx, device=device, dtype=torch.long)
            t_norm = (t_tensor.float() - 1.0) / max(1, (T - 1))
            inp = torch.cat([x, t_norm.view(num_samples, 1)], dim=1)
        elif time_embed == "sinusoidal" and time_dim > 0:
            t_tensor = torch.full((num_samples,), t_idx, device=device, dtype=torch.long)
            t_emb = sinusoidal_time_embedding(t_tensor, dim=time_dim)
            inp = torch.cat([x, t_emb], dim=1)
        else:
            inp = x

        eps_theta = model(inp)
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_theta
        )
        if t_idx > 1:
            z = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
            x = mean + sigma_t * z
        else:
            x = mean
    return x.detach().cpu().numpy().astype(np.float32)


# -----------------------------
# Plotting helpers
# -----------------------------


def plot_betas(betas: torch.Tensor, out_path: str):
    plt.figure(figsize=(5, 3))
    plt.plot(range(1, len(betas) + 1), betas.cpu().numpy())
    plt.xlabel("timestep")
    plt.ylabel("beta_t")
    plt.title("Linear beta schedule")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_alpha_bar(alpha_bars: torch.Tensor, out_path: str):
    plt.figure(figsize=(5, 3))
    plt.plot(range(1, len(alpha_bars) + 1), alpha_bars.cpu().numpy())
    plt.xlabel("timestep")
    plt.ylabel("alpha_bar_t")
    plt.title("Cumulative product of alphas (alpha_bar)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_scatter(pts: np.ndarray, title: str, out_path: str):
    plt.figure(figsize=(4, 4))
    plt.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.85)
    plt.axis("off")
    plt.title(title)
    plt.gca().set_aspect("equal", "box")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_forward_panels(x0: np.ndarray, alpha_bars: torch.Tensor, timesteps: Tuple[int, ...], out_path: str, seed: int = 42):
    rng = np.random.default_rng(seed)
    x0_t = torch.from_numpy(x0)
    fig, axs = plt.subplots(1, len(timesteps), figsize=(4 * len(timesteps), 4))
    if len(timesteps) == 1:
        axs = [axs]
    for ax, t in zip(axs, timesteps):
        B = x0_t.shape[0]
        eps = torch.from_numpy(rng.standard_normal(size=x0.shape).astype(np.float32))
        a_bar = alpha_bars[t - 1].item()
        x_t = math.sqrt(a_bar) * x0_t + math.sqrt(1 - a_bar) * eps
        pts = x_t.numpy()
        ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.85)
        ax.set_title(f"t={t}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", "box")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_losses(train: np.ndarray, val: np.ndarray, title: str, out_path: str, smooth: float = 0.9):
    plt.figure(figsize=(5.5, 3.5))
    plt.plot(train, label="train", alpha=0.5)
    plt.plot(val, label="val", alpha=0.8)
    if len(val) > 2:
        plt.plot(smooth_series(val, alpha=smooth), label="val (EMA)", linewidth=2)
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -----------------------------
# Main
# -----------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/kth8606/DeepHome/DatasaurusDozen.tsv")
    parser.add_argument("--out_dir", type=str, default="/home/kth8606/DeepHome/outputs/ddpm_dinosaur")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta_min", type=float, default=1e-4)
    parser.add_argument("--beta_max", type=float, default=2e-2)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--time_embed", type=str, default="linear", choices=["linear", "sinusoidal"]) 
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--lr_notime", type=float, default=1e-3)
    parser.add_argument("--lr_time", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--width_time", type=int, default=128)
    parser.add_argument("--fourier_pos_L", type=int, default=64)
    parser.add_argument("--fourier_time_L", type=int, default=32)
    parser.add_argument("--sigma_pos", type=float, default=10.0)
    parser.add_argument("--sigma_time", type=float, default=5.0)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--run_notime", type=int, default=1, choices=[0,1])
    parser.add_argument("--run_time", type=int, default=1, choices=[0,1])
    parser.add_argument("--run_fourier", type=int, default=1, choices=[0,1])
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    ensure_dir(args.out_dir)

    # Load and standardize data
    x = load_dino_tsv(args.data_path)
    x_std, mean_xy, std_xy = standardize(x)
    plot_scatter(x_std, "Standardized Dinosaur", os.path.join(args.out_dir, "dino_standardized.png"))

    # Schedule
    schedule = BetaSchedule(T=args.T, beta_min=args.beta_min, beta_max=args.beta_max)
    betas, alphas, alpha_bars = schedule.build()
    plot_betas(betas, os.path.join(args.out_dir, "betas.png"))
    plot_alpha_bar(alpha_bars, os.path.join(args.out_dir, "alpha_bar.png"))

    # Forward noising panels
    forward_ts = (1, 6, 12, 25, args.T)
    plot_forward_panels(x_std, alpha_bars, forward_ts, os.path.join(args.out_dir, "forward_panels.png"), seed=args.seed)

    # Dataloaders
    train_loader, val_loader = make_loaders(x_std, batch_size=args.batch_size)

    tag_str = ("_" + args.tag) if len(args.tag) > 0 else ""

    # Train model without time
    if args.run_notime:
        model_notime = MLP(in_dim=2, hidden=args.width, depth=3, out_dim=2)
        tr_nt, va_nt, best_nt = train_denoiser(
            model_notime, train_loader, val_loader, betas.to(device), alphas.to(device), alpha_bars.to(device), device,
            epochs=args.epochs, lr=args.lr_notime, clip_grad=1.0, time_dim=0, time_embed="none", verbose=args.verbose, log_interval=args.log_interval,
            ema_decay=(args.ema_decay if args.ema else 0.0)
        )
        plot_losses(tr_nt, va_nt, "No-time denoiser loss", os.path.join(args.out_dir, f"loss_notime{tag_str}.png"))
        model_notime.load_state_dict(best_nt)

    # Part 2(b): linear time embedding t~ concat to x
    if args.run_time:
        if args.time_embed == "linear":
            in_dim_time = 2 + 1
        else:
            in_dim_time = 2 + args.time_dim
        model_time = MLP(in_dim=in_dim_time, hidden=args.width_time, depth=3, out_dim=2)
        tr_t, va_t, best_t = train_denoiser(
            model_time, train_loader, val_loader, betas.to(device), alphas.to(device), alpha_bars.to(device), device,
            epochs=args.epochs, lr=args.lr_time, clip_grad=1.0, time_dim=(1 if args.time_embed=="linear" else args.time_dim), time_embed=args.time_embed, verbose=args.verbose, log_interval=args.log_interval,
            ema_decay=(args.ema_decay if args.ema else 0.0)
        )
        plot_losses(tr_t, va_t, "Time-conditioned denoiser loss", os.path.join(args.out_dir, f"loss_time{tag_str}.png"))
        model_time.load_state_dict(best_t)

    # Part 2(c): Fourier features for position (L=64) and time (L=32)
    if args.run_fourier:
        Lp = args.fourier_pos_L
        Lt = args.fourier_time_L
        # fixed random Gaussian B matrices
        B_pos = torch.randn(Lp, 2) * args.sigma_pos
        B_time = torch.randn(Lt, 1) * args.sigma_time

        # gamma(x) uses B_pos (Lp x 2) -> proj dim Lp, after cos+sin -> 2*Lp
        # For position, we encode both coordinates jointly via v=(x1,x2), so out is 2*Lp
        # For time, v=(t~) so out is 2*Lt
        in_dim_ff = 2 * Lp + 2 * Lt
        model_time_ff = MLP(in_dim=in_dim_ff, hidden=args.width_time, depth=3, out_dim=2)

        def ff_input(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            # build Fourier features for x and t~
            x_enc = fourier_encode(x_t, B_pos.to(x_t.device))  # (B, 2*Lp)
            t_norm = (t.float() - 1.0) / max(1, (betas.shape[0] - 1))
            t_enc = fourier_encode(t_norm.view(-1, 1), B_time.to(x_t.device))  # (B, 2*Lt)
            return torch.cat([x_enc, t_enc], dim=1)

        # wrap a tiny adapter module to feed Fourier features through the same training loop
        class FFWrapper(nn.Module):
            def __init__(self, base: nn.Module):
                super().__init__()
                self.base = base
            def forward(self, xt_enc: torch.Tensor) -> torch.Tensor:
                return self.base(xt_enc)

        ff_model = FFWrapper(model_time_ff)

        # custom train loop for Fourier-encoded inputs
        def train_ff():
            ff_model.to(device)
            opt = torch.optim.AdamW(ff_model.parameters(), lr=args.lr_time)
            mse = nn.MSELoss()
            T = betas.shape[0]
            # move schedules to device for consistency
            betas_d = betas.to(device)
            alphas_d = alphas.to(device)
            alpha_bars_d = alpha_bars.to(device)
            best_val = float('inf')
            best_state = copy.deepcopy(ff_model.state_dict())
            tr_hist, va_hist = [], []
            for epoch_idx in range(1, args.epochs + 1):
                ff_model.train()
                ep = []
                for (x0_batch,) in train_loader:
                    Bsz = x0_batch.shape[0]
                    x0 = x0_batch.to(device)
                    t = torch.randint(1, T + 1, (Bsz,), device=device)
                    alpha_bar_t = alpha_bars_d[t - 1].view(Bsz, 1)
                    x_t, eps = q_sample(x0, alpha_bar_t)
                    inp = ff_input(x_t, t)
                    eps_pred = ff_model(inp)
                    loss = mse(eps_pred, eps)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(ff_model.parameters(), max_norm=1.0)
                    opt.step()
                    ep.append(loss.detach().item())
                tr_hist.append(float(np.mean(ep)))
                ff_model.eval()
                with torch.inference_mode():
                    epv = []
                    for (x0_batch,) in val_loader:
                        Bsz = x0_batch.shape[0]
                        x0 = x0_batch.to(device)
                        t = torch.randint(1, T + 1, (Bsz,), device=device)
                        alpha_bar_t = alpha_bars_d[t - 1].view(Bsz, 1)
                        x_t, eps = q_sample(x0, alpha_bar_t)
                        inp = ff_input(x_t, t)
                        eps_pred = ff_model(inp)
                        loss = mse(eps_pred, eps)
                        epv.append(loss.detach().item())
                    va_hist.append(float(np.mean(epv)))
                if args.verbose and (epoch_idx % args.log_interval == 0 or epoch_idx == 1 or epoch_idx == args.epochs):
                    print(f"[FF] epoch {epoch_idx:4d}/{args.epochs}  train_mse={tr_hist[-1]:.6f}  val_mse={va_hist[-1]:.6f}")
                if va_hist[-1] < best_val:
                    best_val = va_hist[-1]
                    best_state = {k: v.detach().cpu().clone() for k, v in ff_model.state_dict().items()}
            return np.array(tr_hist, dtype=np.float32), np.array(va_hist, dtype=np.float32), best_state, (B_pos, B_time)

        tr_ff, va_ff, best_ff, (B_pos, B_time) = train_ff()
        plot_losses(tr_ff, va_ff, "Fourier-conditioned denoiser loss", os.path.join(args.out_dir, f"loss_fourier{tag_str}.png"))
        ff_model.load_state_dict(best_ff)
        # Save B matrices for reproducibility/inspection
        try:
            np.save(os.path.join(args.out_dir, f"B_pos{tag_str}.npy"), B_pos.cpu().numpy())
            np.save(os.path.join(args.out_dir, f"B_time{tag_str}.npy"), B_time.cpu().numpy())
        except Exception:
            pass

    # Sampling with ancestral sampler
    with torch.inference_mode():
        if args.run_notime:
            samples_nt = ancestral_sample(
                model_notime.to(device).eval(), betas.to(device), alphas.to(device), alpha_bars.to(device),
                num_samples=args.num_samples, device=device, time_dim=0, time_embed="none"
            )
        if args.run_time:
            samples_time = ancestral_sample(
                model_time.to(device).eval(), betas.to(device), alphas.to(device), alpha_bars.to(device),
                num_samples=args.num_samples, device=device, time_dim=(1 if args.time_embed=="linear" else args.time_dim), time_embed=args.time_embed
            )

        # Fourier-conditioned sampling: build per-step Fourier inputs
        def ff_step_input(x: torch.Tensor, t_idx: int) -> torch.Tensor:
            t_tensor = torch.full((x.shape[0],), t_idx, device=x.device, dtype=torch.long)
            t_norm = (t_tensor.float() - 1.0) / max(1, (betas.shape[0] - 1))
            x_enc = fourier_encode(x, B_pos.to(x.device))
            t_enc = fourier_encode(t_norm.view(-1, 1), B_time.to(x.device))
            return torch.cat([x_enc, t_enc], dim=1)

        if args.run_fourier:
            T_steps = betas.shape[0]
            betas_d = betas.to(device)
            alphas_d = alphas.to(device)
            alpha_bars_d = alpha_bars.to(device)
            x_cur = torch.randn(args.num_samples, 2, device=device)
            ff_local = ff_model.to(device).eval()
            for t_idx in reversed(range(1, T_steps + 1)):
                beta_t = betas_d[t_idx - 1]
                alpha_t = alphas_d[t_idx - 1]
                alpha_bar_t = alpha_bars_d[t_idx - 1]
                inp = ff_step_input(x_cur, t_idx)
                eps_theta = ff_local(inp)
                mean = (1.0 / torch.sqrt(alpha_t)) * (
                    x_cur - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_theta
                )
                if t_idx > 1:
                    z = torch.randn_like(x_cur)
                    sigma_t = torch.sqrt(beta_t)
                    x_cur = mean + sigma_t * z
                else:
                    x_cur = mean
            samples_fourier = x_cur.detach().cpu().numpy().astype(np.float32)
    if args.run_notime:
        plot_scatter(samples_nt, "Samples (no time)", os.path.join(args.out_dir, f"samples_notime{tag_str}.png"))
    if args.run_time:
        plot_scatter(samples_time, "Samples (with time)", os.path.join(args.out_dir, f"samples_time{tag_str}.png"))
    if args.run_fourier:
        plot_scatter(samples_fourier, "Samples (Fourier-conditioned)", os.path.join(args.out_dir, f"samples_fourier{tag_str}.png"))

    # Notes
    note = []
    note.append("Effect of beta_min/beta_max in linear schedule: higher values speed up information destruction per step;\n")
    note.append("- Increasing beta_min raises early noise; the structure fades quickly from the start.\n")
    note.append("- Increasing beta_max raises late noise; the last steps become very destructive.\n")
    note.append("Limitation without time: a single mapping x_t -> epsilon cannot represent different noise levels across timesteps;\n")
    note.append("it must average over t, leading to blurry or biased denoising and poorer samples.\n")
    with open(os.path.join(args.out_dir, "notes.txt"), "w") as f:
        f.writelines(note)

    print("Done. Figures saved to:", args.out_dir)
    print("- betas.png, alpha_bar.png, forward_panels.png, dino_standardized.png")
    prints = []
    if args.run_notime:
        prints += [f"loss_notime{tag_str}.png", f"samples_notime{tag_str}.png"]
    if args.run_time:
        prints += [f"loss_time{tag_str}.png", f"samples_time{tag_str}.png"]
    if args.run_fourier:
        prints += [f"loss_fourier{tag_str}.png", f"samples_fourier{tag_str}.png"]
    print("- " + ", ".join(prints))
    print("See notes.txt for brief explanations.")


if __name__ == "__main__":
    main()



