#!/bin/bash
#SBATCH -p base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=vae_stl10
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Paths and params
VENV_DIR="/home/kth8606/DeepHome/vae_stl10/.venv"
SCRIPT_PATH="/home/kth8606/DeepHome/vae_stl10/train.py"
OUT_DIR="/home/kth8606/DeepHome/vae_stl10/outputs/${SLURM_JOB_ID:-manual}"

# Allow overrides via environment variables
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LATENT_DIM="${LATENT_DIM:-128}"
RECON_LOSS="${RECON_LOSS:-bce}"
SPLIT="${SPLIT:-train}"
LR="${LR:-1e-3}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-}"

# Activate environment
if [ -d "${VENV_DIR}" ]; then
	source "${VENV_DIR}/bin/activate"
else
	echo "Warning: VENV_DIR not found at ${VENV_DIR}. Continuing without explicit venv activation."
fi

# Use local project modules
export PYTHONPATH="/home/kth8606/DeepHome:${PYTHONPATH:-}"

mkdir -p "${OUT_DIR}"

echo "### VAE STL-10 sbatch job start: $(date)"
echo "### Node: $(hostname)"
echo "### CUDA devices: ${CUDA_VISIBLE_DEVICES:-}"
echo "### Python: $(command -v python)"
python --version || true

# Optional: show GPU info
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

# Optional device argument
if [ -n "${DEVICE}" ]; then
	DEVICE_OPT=(--device "${DEVICE}")
else
	DEVICE_OPT=()
fi

# Train
python "${SCRIPT_PATH}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --latent-dim "${LATENT_DIM}" \
  --recon-loss "${RECON_LOSS}" \
  --split "${SPLIT}" \
  --num-workers "${NUM_WORKERS}" \
  --out-dir "${OUT_DIR}" \
  "${DEVICE_OPT[@]}"

echo "### VAE STL-10 sbatch job end: $(date)"
