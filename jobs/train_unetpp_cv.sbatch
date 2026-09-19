#!/bin/bash
#SBATCH --job-name=unetpp_cv
#SBATCH --account=eungjoolee
#SBATCH --partition=gpu_standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --time=05:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/cv_train/unetpp_%A_%a.out
#SBATCH --error=logs/cv_train/unetpp_%A_%a.err

set -euo pipefail

source /opt/ohpc/pub/apps/anaconda/2024.06/etc/profile.d/conda.sh

conda activate /groups/eungjoolee/puneethsamarla/conda_envs/eda310

cd /groups/eungjoolee/puneethsamarla/drusen_volume_project

export PYTHONUNBUFFERED=1


# ------------------------------------------------------------
# 15 tasks:
#
# task 0-2   -> fold 1, seeds 42/43/44
# task 3-5   -> fold 2
# task 6-8   -> fold 3
# task 9-11  -> fold 4
# task 12-14 -> fold 5
# ------------------------------------------------------------

FOLD=$((SLURM_ARRAY_TASK_ID / 3 + 1))

SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))

SEEDS=(42 43 44)

SEED=${SEEDS[$SEED_INDEX]}


EPOCHS=${EPOCHS:-20}
WORKERS=${WORKERS:-4}

OUT_ROOT=${OUT_ROOT:-models/cv/unetpp}

OUT_DIR=${OUT_ROOT}/fold_${FOLD}/seed_${SEED}

MANIFEST=data/manifests/cv/fold_${FOLD}.csv


echo "============================================================"
echo "U-NET++ FIVE-FOLD UNCERTAINTY TRAINING"
echo "============================================================"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Fold: ${FOLD}"
echo "Seed: ${SEED}"
echo "Epochs: ${EPOCHS}"
echo "Manifest: ${MANIFEST}"
echo "Output: ${OUT_DIR}"

echo


if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: Missing manifest ${MANIFEST}"
    exit 1
fi


echo "Torch / GPU check:"

python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY


mkdir -p "${OUT_DIR}"


python -u src/train_unetpp_cv.py --manifest "${MANIFEST}" --fold "${FOLD}" --seed "${SEED}" --out_dir "${OUT_DIR}" --epochs "${EPOCHS}" --batch_size 4 --workers "${WORKERS}" --lr 0.0003 --weight_decay 0.0001 --patience 5 --threshold 0.5 --encoder resnet34 --encoder_weights imagenet


echo
echo "============================================================"
echo "COMPLETED"
echo "Fold: ${FOLD}"
echo "Seed: ${SEED}"
echo "============================================================"