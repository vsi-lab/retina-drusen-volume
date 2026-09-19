#!/bin/bash

#SBATCH --job-name=rpedc_model

#SBATCH --account=eungjoolee

#SBATCH --partition=gpu_standard

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8

#SBATCH --mem-per-cpu=6G

#SBATCH --time=24:00:00

#SBATCH --output=logs/train_%A_%a.out

#SBATCH --error=logs/train_%A_%a.err

#SBATCH --array=0-3



set -euo pipefail



export PYTHONUNBUFFERED=1



source /opt/ohpc/pub/apps/anaconda/2024.06/etc/profile.d/conda.sh



conda activate /groups/eungjoolee/puneethsamarla/conda_envs/eda310



cd /groups/eungjoolee/puneethsamarla/drusen_volume_project



mkdir -p logs

mkdir -p models



ARCHITECTURES=(

    unet

    unetpp

    fpn

    deeplabv3plus

)



ARCHITECTURE="${ARCHITECTURES[$SLURM_ARRAY_TASK_ID]}"



echo "Architecture: ${ARCHITECTURE}"

echo "Array task: ${SLURM_ARRAY_TASK_ID}"

echo "Node: $(hostname)"

echo "Start time: $(date)"

echo "Python: $(which python)"



nvidia-smi



python -u src/train_rpedc_models.py --manifest data/manifests/duke_rpedc_manifest.csv --splits data/splits/duke_subject_split.csv --out_dir "models/${ARCHITECTURE}" --arch "${ARCHITECTURE}" --encoder resnet34 --encoder_weights imagenet --epochs 10 --bs 4 --workers 8 --lr 3e-4 --weight_decay 1e-4 --bce_weight 0.4 --dice_weight 0.6 --pos_weight 3.0 --threshold 0.5 --patience 4 --seed 42 --amp



echo "Completed architecture: ${ARCHITECTURE}"

echo "End time: $(date)"

