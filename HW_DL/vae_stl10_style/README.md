# VAE STL-10: Latent Style Transfer

Transfer style-like attributes from image B to A using latent vector arithmetic and interpolation.

## Usage

After training, run:

```bash
python /home/kth8606/DeepHome/vae_stl10_style/style_transfer.py \
  --ckpt /home/kth8606/DeepHome/vae_stl10/outputs/model_final.pt \
  --out-dir /home/kth8606/DeepHome/vae_stl10/outputs \
  --split test \
  --device cuda \
  --lambda-list 0.0 0.2 0.4 0.6 0.8 1.0 \
  --alpha-list 0.0 0.2 0.4 0.6 0.8 1.0
```

Notes:
- Class means are computed from the chosen split (`test` or `train`); do not use `train+unlabeled` here.
- Two grids are saved: arithmetic (λ sweep) and interpolation (α sweep).
