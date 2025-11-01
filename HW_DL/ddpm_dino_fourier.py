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


def fourier_encode(v: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    proj = v @ B.t()
    proj = 2 * math.pi * proj
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


def encode_with_mapping(x: torch.Tensor, mapping: str, B: torch.Tensor | None) -> torch.Tensor:
    if mapping == "none" or B is None:
        return x
    return fourier_encode(x, B)


def load_dino_tsv(path: str) -> np.ndarray:
    raw = pd.read_csv(path, sep="\t")
    if {"dataset", "x", "y"}.issubset(set(raw.columns)):
        dino = raw[raw["dataset"] == "dino"][ ["x", "y"] ].to_numpy()
    else:
        num_cols = [c for c in raw.columns if np.issubdtype(raw[c].dtype, np.number)]
        dino = raw[num_cols[:2]].to_numpy()
    return dino


def make_loaders(x: np.ndarray, batch_size: int = 256, val_frac: float = 0.15):
    x_tensor = torch.from_numpy(x)
    n = x_tensor.shape[0]
    n_val = max(1, int(n * val_frac))
    train = TensorDataset(x_tensor[:-n_val]) if n_val < n else TensorDataset(x_tensor)
    val = TensorDataset(x_tensor[-n_val:]) if n_val < n else TensorDataset(x_tensor[:1])
    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(val, batch_size=batch_size, shuffle=False)


def q_sample(x0: torch.Tensor, alpha_bar_t: torch.Tensor):
    eps = torch.randn_like(x0)
    return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * eps, eps


def train_fourier(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                  betas: torch.Tensor, alphas: torch.Tensor, alpha_bars: torch.Tensor,
                  device: torch.device, epochs: int, lr: float, clip_grad: float,
                  B_pos: torch.Tensor | None, B_time: torch.Tensor | None,
                  mapping_pos: str, mapping_time: str):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    T = betas.shape[0]
    betas_d, alphas_d, alpha_bars_d = betas.to(device), alphas.to(device), alpha_bars.to(device)
    tr_hist, va_hist = [], []

    def ff_input(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_enc = encode_with_mapping(x_t, mapping_pos, None if B_pos is None else B_pos.to(x_t.device))
        t_norm = (t.float() - 1.0) / max(1, (T - 1))
        t_in = t_norm.view(-1, 1)
        t_enc = encode_with_mapping(t_in, mapping_time, None if B_time is None else B_time.to(x_t.device))
        return torch.cat([x_enc, t_enc], dim=1)

    for epoch in range(1, epochs + 1):
        model.train()
        ep = []
        for (x0_batch,) in train_loader:
            B = x0_batch.shape[0]
            x0 = x0_batch.to(device)
            t = torch.randint(1, T + 1, (B,), device=device)
            alpha_bar_t = alpha_bars_d[t - 1].view(B, 1)
            x_t, eps = q_sample(x0, alpha_bar_t)
            inp = ff_input(x_t, t)
            pred = model(inp)
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
                alpha_bar_t = alpha_bars_d[t - 1].view(B, 1)
                x_t, eps = q_sample(x0, alpha_bar_t)
                inp = ff_input(x_t, t)
                pred = model(inp)
                loss = mse(pred, eps)
                epv.append(loss.detach().item())
            va_hist.append(float(np.mean(epv)))
    return np.array(tr_hist, dtype=np.float32), np.array(va_hist, dtype=np.float32)


@torch.no_grad()
def ancestral_sample_fourier(model: nn.Module, betas: torch.Tensor, alphas: torch.Tensor, alpha_bars: torch.Tensor,
                             num_samples: int, device: torch.device, B_pos: torch.Tensor | None, B_time: torch.Tensor | None,
                             mapping_pos: str, mapping_time: str):
    T = betas.shape[0]
    betas_d, alphas_d, alpha_bars_d = betas.to(device), alphas.to(device), alpha_bars.to(device)
    x = torch.randn(num_samples, 2, device=device)

    def ff_step_input(x_cur: torch.Tensor, t_idx: int) -> torch.Tensor:
        t_tensor = torch.full((x_cur.shape[0],), t_idx, device=x_cur.device, dtype=torch.long)
        t_norm = (t_tensor.float() - 1.0) / max(1, (T - 1))
        x_enc = encode_with_mapping(x_cur, mapping_pos, None if B_pos is None else B_pos.to(x_cur.device))
        t_in = t_norm.view(-1, 1)
        t_enc = encode_with_mapping(t_in, mapping_time, None if B_time is None else B_time.to(x_cur.device))
        return torch.cat([x_enc, t_enc], dim=1)

    model_local = model.to(device).eval()
    for t_idx in reversed(range(1, T + 1)):
        beta_t = betas_d[t_idx - 1]
        alpha_t = alphas_d[t_idx - 1]
        alpha_bar_t = alpha_bars_d[t_idx - 1]
        inp = ff_step_input(x, t_idx)
        eps = model_local(inp)
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
    plt.axis("off"); plt.title(title); plt.gca().set_aspect("equal", "box")
    plt.tight_layout(); plt.savefig(out_path); plt.close()


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
    parser.add_argument("--out_dir", type=str, default="/home/kth8606/DeepHome/outputs/ddpm_dinosaur_fourier")
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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--fourier_pos_L", type=int, default=64)
    parser.add_argument("--fourier_time_L", type=int, default=32)
    parser.add_argument("--sigma_pos", type=float, default=10.0)
    parser.add_argument("--sigma_time", type=float, default=5.0)
    parser.add_argument("--mapping_pos", type=str, default="gaussian", choices=["gaussian", "basic", "none"], help="Position mapping per fourier-feature-networks: gaussian/basic/none")
    parser.add_argument("--mapping_time", type=str, default="gaussian", choices=["gaussian", "basic", "none"], help="Time mapping per fourier-feature-networks: gaussian/basic/none")
    parser.add_argument("--basic_scale_pos", type=float, default=1.0, help="Scale for basic pos mapping (B = scale * I)")
    parser.add_argument("--basic_scale_time", type=float, default=1.0, help="Scale for basic time mapping (B = scale * I)")
    parser.add_argument("--load_B_dir", type=str, default="", help="Optional directory to load B_pos.npy / B_time.npy. If not found, will generate.")
    parser.add_argument("--save_B", action="store_true", help="If set, saves B_pos.npy and B_time.npy to out_dir")
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

    # Build mapping matrices following https://github.com/tancik/fourier-feature-networks
    def maybe_load_B(path: str) -> np.ndarray | None:
        try:
            if os.path.isfile(path):
                return np.load(path)
        except Exception:
            return None
        return None

    B_pos, B_time = None, None
    pos_feat_dim, time_feat_dim = 0, 0

    if args.mapping_pos == "gaussian":
        Lp = args.fourier_pos_L
        B_pos = torch.randn(Lp, 2) * args.sigma_pos
        pos_feat_dim = 2 * Lp
        if args.load_B_dir:
            arr = maybe_load_B(os.path.join(args.load_B_dir, "B_pos.npy"))
            if arr is not None:
                B_pos = torch.from_numpy(arr.astype(np.float32))
    elif args.mapping_pos == "basic":
        B_pos = torch.eye(2) * args.basic_scale_pos
        pos_feat_dim = 2 * 2
    else:  # none
        B_pos = None
        pos_feat_dim = 2

    if args.mapping_time == "gaussian":
        Lt = args.fourier_time_L
        B_time = torch.randn(Lt, 1) * args.sigma_time
        time_feat_dim = 2 * Lt
        if args.load_B_dir:
            arr = maybe_load_B(os.path.join(args.load_B_dir, "B_time.npy"))
            if arr is not None:
                B_time = torch.from_numpy(arr.astype(np.float32))
    elif args.mapping_time == "basic":
        B_time = torch.eye(1) * args.basic_scale_time
        time_feat_dim = 2 * 1
    else:  # none
        B_time = None
        time_feat_dim = 1

    in_dim_ff = pos_feat_dim + time_feat_dim
    model = MLP(in_dim=in_dim_ff, hidden=args.width, depth=3, out_dim=2)

    tr, va = train_fourier(model, train_loader, val_loader, betas, alphas, alpha_bars, device,
                           epochs=args.epochs, lr=args.lr, clip_grad=1.0,
                           B_pos=B_pos, B_time=B_time,
                           mapping_pos=args.mapping_pos, mapping_time=args.mapping_time)

    plt.figure(figsize=(5.5, 3.5))
    plt.plot(tr, label="train", alpha=0.5)
    plt.plot(va, label="val", alpha=0.8)
    if len(va) > 2:
        plt.plot(smooth_series(va, alpha=0.9), label="val (EMA)", linewidth=2)
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("Fourier-conditioned denoiser loss")
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "loss_fourier.png")); plt.close()

    if args.save_B:
        if B_pos is not None:
            np.save(os.path.join(args.out_dir, "B_pos.npy"), B_pos.cpu().numpy())
        if B_time is not None:
            np.save(os.path.join(args.out_dir, "B_time.npy"), B_time.cpu().numpy())

    samples = ancestral_sample_fourier(model, betas, alphas, alpha_bars, args.num_samples, device,
                                      B_pos, B_time, mapping_pos=args.mapping_pos, mapping_time=args.mapping_time)
    plot_scatter(samples, "Samples (Fourier-conditioned)", os.path.join(args.out_dir, "samples_fourier.png"))

    print("Done. Figures saved to:", args.out_dir)


if __name__ == "__main__":
    main()


