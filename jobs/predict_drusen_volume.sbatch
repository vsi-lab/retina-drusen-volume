#!/bin/bash

#SBATCH --job-name=drusen_vol

#SBATCH --account=eungjoolee

#SBATCH --partition=gpu_standard

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8

#SBATCH --mem-per-cpu=6G

#SBATCH --time=08:00:00

#SBATCH --output=logs/drusen_volume_%A_%a.out

#SBATCH --error=logs/drusen_volume_%A_%a.err

#SBATCH --array=0-1



source /opt/ohpc/pub/apps/anaconda/2024.06/etc/profile.d/conda.sh

conda activate /groups/eungjoolee/puneethsamarla/conda_envs/eda310



cd /groups/eungjoolee/puneethsamarla/drusen_volume_project



export PYTHONUNBUFFERED=1



MODELS=("unetpp" "deeplabv3plus")

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}



echo "======================================"

echo "Model: ${MODEL}"

echo "Node: $(hostname)"

echo "Array task: ${SLURM_ARRAY_TASK_ID}"

echo "======================================"



python - <<'PY'

import torch

print("Torch:", torch.__version__)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():

    print("GPU:", torch.cuda.get_device_name(0))

PY



python -u src/predict_rpedc_drusen_volume.py --manifest data/manifests/duke_rpedc_manifest.csv --splits data/splits/duke_subject_split.csv --thickness_manifest reports/normal_reference/rpedc_thickness_manifest.csv --reference_csv reports/drusen_volume/reference_formula/drusen_volume_threshold_sensitivity.csv --checkpoint models/${MODEL}/best.pt --model_name ${MODEL} --out_dir reports/drusen_volume/predicted_formula --batch_size 4

