#!/bin/bash

#SBATCH --job-name=drusen_gpu_check

#SBATCH --account=eungjoolee

#SBATCH --partition=gpu_standard

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=4G

#SBATCH --time=00:15:00

#SBATCH --output=logs/gpu_check_%j.out

#SBATCH --error=logs/gpu_check_%j.err



source /opt/ohpc/pub/apps/anaconda/2024.06/etc/profile.d/conda.sh

conda activate /groups/eungjoolee/puneethsamarla/conda_envs/eda310



cd /groups/eungjoolee/puneethsamarla/drusen_volume_project



nvidia-smi



python - <<'PY'

import torch

print("Torch:", torch.__version__)

print("CUDA build:", torch.version.cuda)

print("CUDA available:", torch.cuda.is_available())

print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():

    print("GPU:", torch.cuda.get_device_name(0))

PY

