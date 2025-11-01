import os
import torch
import matplotlib.pyplot as plt

ckpt_path = "/home/kth8606/DeepHome/vae_stl10/outputs/719394/model_final.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")
h = ckpt["history"]

epochs = list(range(1, len(h["recon"]) + 1))

fig, ax1 = plt.subplots(figsize=(7, 5))

# Normalize Reconstruction and Total by number of pixels (3*96*96)
num_pixels = 3 * 96 * 96
recon_per_pixel = [v / num_pixels for v in h["recon"]]
total_per_pixel = [v / num_pixels for v in h["total"]]

# Left axis: Reconstruction + Total (per pixel)
l1 = ax1.plot(epochs, recon_per_pixel, label="Reconstruction (per-pixel)", color="tab:blue")
l2 = ax1.plot(epochs, total_per_pixel, label="Total (per-pixel)", color="tab:orange")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Reconstruction/Total (per-pixel)")
ax1.set_ylim(0, 1.0)

# Right axis: KL
ax2 = ax1.twinx()
l3 = ax2.plot(epochs, h["kl"], label="KL", color="tab:green")
ax2.set_ylabel("KL")
ax2.set_ylim(0, 500)

# Combined legend
lines = l1 + l2 + l3
labels = [ln.get_label() for ln in lines]
ax1.legend(lines, labels, loc="upper right")

plt.title("VAE Losses (Dual Axes)")
plt.tight_layout()

out_dir = os.path.dirname(ckpt_path)
out_path = os.path.join(out_dir, "loss_curve_dual_axes.png")
plt.savefig(out_path, dpi=150)
print(f"Saved {out_path}")