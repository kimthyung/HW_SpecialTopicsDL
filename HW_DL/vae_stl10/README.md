# ConvVAE on STL-10

This project implements a convolutional Variational Autoencoder (VAE) for the STL-10 dataset with training, loss plotting, and random image generation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r vae_stl10/requirements.txt
```

## Train (50 epochs default)

```bash
python vae_stl10/train.py --epochs 50 --batch-size 128 --latent-dim 128 --recon-loss bce --out-dir vae_stl10/outputs
```

Flags:
- `--epochs`: number of epochs (default 50)
- `--batch-size`: global batch size (default 128)
- `--lr`: learning rate (default 1e-3)
- `--latent-dim`: latent dimension (default 128)
- `--recon-loss`: `bce` or `mse` (default `bce`)
- `--split`: STL10 split: `train` or `train+unlabeled` (default `train`)
- `--device`: e.g., `cuda`, `cuda:0`, or `cpu` (auto if omitted)
- `--num-workers`: dataloader workers (default 4)
- `--out-dir`: directory to write outputs (images, plots, checkpoints)

On first run the STL-10 dataset will be downloaded automatically to `~/.torch` by torchvision.

## Outputs
- `loss_curve.png`: plot of reconstruction, KL, and total loss per epoch
- `reconstructions_final.png`: grid of input vs reconstruction samples
- `samples_final.png`: 10 random images from latent samples
- `model_final.pt`: trained model checkpoint

## Model architecture (ConvVAE)

- **Input**: images `3x96x96`
- **Latent dimension**: default 128

| Stage | Layer | Details | Output Shape |
|---|---|---|---|
| Encoder | Conv2d | 3→64, k=4,s=2,p=1 + BN + ReLU | 64×48×48 |
|  | Conv2d | 64→128, k=4,s=2,p=1 + BN + ReLU | 128×24×24 |
|  | Conv2d | 128→256, k=4,s=2,p=1 + BN + ReLU | 256×12×12 |
|  | Conv2d | 256→512, k=4,s=2,p=1 + BN + ReLU | 512×6×6 |
|  | Conv2d | 512→512, k=4,s=2,p=1 + BN + ReLU | 512×3×3 |
|  | Flatten |  | 4608 |
|  | Linear μ | 4608→latent_dim | latent_dim |
|  | Linear logσ² | 4608→latent_dim | latent_dim |
| Reparam. | z = μ + σ⊙ε | ε∼N(0,I) | latent_dim |
| Decoder | Linear | latent_dim→4608 | 4608 |
|  | Reshape |  | 512×3×3 |
|  | ConvTranspose2d | 512→512, k=4,s=2,p=1 + BN + ReLU | 512×6×6 |
|  | ConvTranspose2d | 512→256, k=4,s=2,p=1 + BN + ReLU | 256×12×12 |
|  | ConvTranspose2d | 256→128, k=4,s=2,p=1 + BN + ReLU | 128×24×24 |
|  | ConvTranspose2d | 128→64, k=4,s=2,p=1 + BN + ReLU | 64×48×48 |
|  | ConvTranspose2d | 64→3, k=4,s=2,p=1 + Sigmoid | 3×96×96 |

## Notes
- Images are normalized to [0,1] and the decoder ends with `Sigmoid` when using `bce`. If you choose `mse`, the same output is used but the loss changes.
- Default split is `train`. You can use `train+unlabeled` to leverage more data.
