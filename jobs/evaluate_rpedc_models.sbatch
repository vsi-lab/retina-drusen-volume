#!/bin/bash

#SBATCH --job-name=rpedc_eval

#SBATCH --account=eungjoolee

#SBATCH --partition=gpu_standard

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8

#SBATCH --mem-per-cpu=6G

#SBATCH --time=12:00:00

#SBATCH --output=logs/eval_%A_%a.out

#SBATCH --error=logs/eval_%A_%a.err

#SBATCH --array=0-3



set -euo pipefail



export PYTHONUNBUFFERED=1



source /opt/ohpc/pub/apps/anaconda/2024.06/etc/profile.d/conda.sh



conda activate /groups/eungjoolee/puneethsamarla/conda_envs/eda310



cd /groups/eungjoolee/puneethsamarla/drusen_volume_project



mkdir -p logs

mkdir -p reports/segmentation



ARCHITECTURES=(

    unet

    unetpp

    fpn

    deeplabv3plus

)



ARCHITECTURE="${ARCHITECTURES[$SLURM_ARRAY_TASK_ID]}"



echo "Evaluating: ${ARCHITECTURE}"

echo "Node: $(hostname)"

echo "Start time: $(date)"



python -u src/evaluate_rpedc_model.py --manifest data/manifests/duke_rpedc_manifest.csv --splits data/splits/duke_subject_split.csv --split_name test --ckpt "models/${ARCHITECTURE}/best.pt" --out_dir "reports/segmentation/${ARCHITECTURE}" --bs 4 --workers 8



echo "Evaluation completed: ${ARCHITECTURE}"

echo "End time: $(date)"

