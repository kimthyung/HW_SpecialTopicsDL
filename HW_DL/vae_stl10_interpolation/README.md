# VAE STL-10: Latent Space Interpolation

This experiment interpolates in the latent space between two STL-10 images from different classes and saves the generated images.

## Usage

After training, run:

```bash
python /home/kth8606/DeepHome/vae_stl10_interpolation/interpolate.py \
  --ckpt /home/kth8606/DeepHome/vae_stl10/outputs/model_final.pt \
  --out-dir /home/kth8606/DeepHome/vae_stl10/outputs \
  --split test \
  --device cuda
```

Notes:
- `--ckpt` must point to the saved checkpoint from training.
- `--split` should be `test` or `train` (do not use `train+unlabeled`, as unlabeled images lack class labels).
- If `--device` is omitted, it auto-selects CUDA if available.

Outputs:
- `interpolation_grid.png`: a 2-row grid. Top row: interpolation from image A→B, bottom row: B→A, for α ∈ [0, 0.25, 0.5, 0.75, 1.0].
