#!/bin/bash

#SBATCH --job-name=img_repro

#SBATCH --account=eungjoolee

#SBATCH --partition=standard

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=4G

#SBATCH --time=04:00:00

#SBATCH --output=logs/img_repro_%j.out

#SBATCH --error=logs/img_repro_%j.err



set -euo pipefail



source /opt/ohpc/pub/apps/anaconda/2024.06/etc/profile.d/conda.sh



conda activate /groups/eungjoolee/puneethsamarla/conda_envs/eda310



cd /groups/eungjoolee/puneethsamarla/drusen_volume_project



export PYTHONUNBUFFERED=1



python -u src/check_all_old_new_images.py

