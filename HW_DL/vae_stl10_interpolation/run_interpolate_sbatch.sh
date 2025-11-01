#!/bin/bash
#SBATCH -p suma_rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=vae_interp
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Paths and params
VENV_DIR="/home/kth8606/DeepHome/vae_stl10/.venv"
SCRIPT_PATH="/home/kth8606/DeepHome/vae_stl10_interpolation/interpolate.py"
OUT_DIR="/home/kth8606/DeepHome/vae_stl10/outputs/interp_${SLURM_JOB_ID:-manual}"

# Allow overrides via environment variables
CKPT="${CKPT:-}"
SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-}"
ALPHAS="${ALPHAS:-0.0 0.25 0.5 0.75 1.0}"

# Activate environment
if [ -d "${VENV_DIR}" ]; then
	source "${VENV_DIR}/bin/activate"
else
	echo "Warning: VENV_DIR not found at ${VENV_DIR}. Continuing without explicit venv activation."
fi

# Use local project modules
export PYTHONPATH="/home/kth8606/DeepHome:${PYTHONPATH:-}"

mkdir -p "${OUT_DIR}"

# Resolve checkpoint if not provided
if [ -z "${CKPT}" ]; then
	CKPT=$(ls -1 /home/kth8606/DeepHome/vae_stl10/outputs/*/model_final.pt 2>/dev/null | tail -n1 || true)
fi
if [ ! -f "${CKPT}" ]; then
	echo "Error: checkpoint not found. Set CKPT env var or ensure outputs exist."
	exit 1
fi

echo "### VAE Interp sbatch start: $(date)"
echo "Node: $(hostname)"
echo "CUDA: ${CUDA_VISIBLE_DEVICES:-}"
echo "Python: $(command -v python)"
python --version || true
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

# Optional device argument
if [ -n "${DEVICE}" ]; then
	DEVICE_OPT=(--device "${DEVICE}")
else
	DEVICE_OPT=()
fi

# Run interpolation
python "${SCRIPT_PATH}" \
  --ckpt "${CKPT}" \
  --out-dir "${OUT_DIR}" \
  --split "${SPLIT}" \
  --alphas ${ALPHAS} \
  "${DEVICE_OPT[@]}"

echo "### VAE Interp sbatch end: $(date)"




