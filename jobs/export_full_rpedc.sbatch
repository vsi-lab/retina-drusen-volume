#!/bin/bash

#SBATCH --job-name=export_rpedc

#SBATCH --account=eungjoolee

#SBATCH --partition=standard

#SBATCH --cpus-per-task=8

#SBATCH --mem-per-cpu=6G

#SBATCH --time=12:00:00

#SBATCH --output=logs/export_full_rpedc_%j.out

#SBATCH --error=logs/export_full_rpedc_%j.err



set -euo pipefail



source /opt/ohpc/pub/apps/anaconda/2024.06/etc/profile.d/conda.sh

conda activate /groups/eungjoolee/puneethsamarla/conda_envs/eda310



cd /groups/eungjoolee/puneethsamarla/drusen_volume_project



export PYTHONUNBUFFERED=1



echo "============================================================"

echo "FULL DUKE RPEDC EXPORT"

echo "============================================================"

echo "Started: $(date)"

echo "Node: $(hostname)"

echo "Python: $(which python)"

echo



AMD_DIR=$(find data/raw/duke_full_release -type d -iname "AMD" | head -1)

CONTROL_DIR=$(find data/raw/duke_full_release -type d -iname "Control" | head -1)



echo "AMD directory: $AMD_DIR"

echo "Control directory: $CONTROL_DIR"



if [ -z "$AMD_DIR" ]; then

    echo "ERROR: AMD directory not found."

    exit 1

fi



if [ -z "$CONTROL_DIR" ]; then

    echo "ERROR: Control directory not found."

    exit 1

fi



AMD_COUNT=$(find "$AMD_DIR" -type f -iname "*.mat" | wc -l)

CONTROL_COUNT=$(find "$CONTROL_DIR" -type f -iname "*.mat" | wc -l)



echo

echo "AMD MAT files: $AMD_COUNT"

echo "Control MAT files: $CONTROL_COUNT"



if [ "$AMD_COUNT" -ne 269 ]; then

    echo "ERROR: Expected 269 AMD MAT files, found $AMD_COUNT"

    exit 1

fi



if [ "$CONTROL_COUNT" -ne 115 ]; then

    echo "ERROR: Expected 115 Control MAT files, found $CONTROL_COUNT"

    exit 1

fi



echo

echo "Dataset count validation: PASS"

echo



mkdir -p data/processed_full

mkdir -p data/manifests



# Remove only the previous incomplete manifest.

# Existing exported image/mask files can safely be overwritten.

rm -f data/manifests/duke_full_rpedc_manifest.csv



echo "Starting RPEDC export..."

echo



python -u src/export_duke_rpedc.py --amd_dir "$AMD_DIR" --control_dir "$CONTROL_DIR" --out_root data/processed_full --manifest data/manifests/duke_full_rpedc_manifest.csv --upper_channel 1 --lower_channel 2 --minimum_valid_columns 250



echo

echo "============================================================"

echo "EXPORT COMMAND COMPLETED"

echo "============================================================"



if [ ! -f data/manifests/duke_full_rpedc_manifest.csv ]; then

    echo "ERROR: Manifest was not created."

    exit 1

fi



python - <<'PY'

import pandas as pd



p = "data/manifests/duke_full_rpedc_manifest.csv"

df = pd.read_csv(p)



print()

print("============================================================")

print("FULL EXPORT SUMMARY")

print("============================================================")

print("Manifest:", p)

print("Total exported slices:", len(df))

print("Total subjects:", df["subject"].nunique())



print("\nSubjects by group:")

print(df.groupby("group")["subject"].nunique())



print("\nSlices by group:")

print(df.groupby("group").size())



print("\nValid columns:")

print(df["valid_columns"].describe())



print("\nMask pixels:")

print(df["mask_pixels"].describe())

PY



echo

echo "Finished: $(date)"

echo "FULL RPEDC EXPORT: SUCCESS"

