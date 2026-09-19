#!/usr/bin/env python3

import re
from pathlib import Path

import pandas as pd


MANIFEST = Path(
    "data/manifests/duke_full_rpedc_manifest.csv"
)

SPLIT = Path(
    "data/splits/duke_5fold_subject_roles.csv"
)


pattern = re.compile(
    r"Farsiu_Ophthalmology_2013_"
    r"(AMD|Control)_Subject_(\d+)"
)


def make_subject_key(name):

    match = pattern.search(str(name))

    if match is None:
        return None

    group = match.group(1)
    subject_id = int(match.group(2))

    return f"{group}_{subject_id}"


manifest = pd.read_csv(MANIFEST)

manifest["subject_key"] = (
    manifest["subject"]
    .apply(make_subject_key)
)


print("=" * 80)
print("FULL RPEDC EXPORT VALIDATION")
print("=" * 80)


print("\nTOTAL SLICES")
print(len(manifest))


print("\nSUBJECTS BY GROUP")
print(
    manifest.groupby("group")[
        "subject_key"
    ].nunique()
)


print("\nSLICES BY GROUP")
print(
    manifest.groupby("group").size()
)


print("\nIMAGE SHAPES")
print(
    manifest[
        [
            "height",
            "width",
        ]
    ].drop_duplicates()
)


print("\nVALID COLUMNS")
print(
    manifest[
        "valid_columns"
    ].describe()
)


print("\nMASK PIXELS")
print(
    manifest[
        "mask_pixels"
    ].describe()
)


# ---------------------------------------
# Structural checks
# ---------------------------------------

assert (
    manifest["height"] == 512
).all()

assert (
    manifest["width"] == 1000
).all()

assert (
    manifest["valid_columns"] >= 250
).all()

assert (
    manifest["mask_pixels"] > 0
).all()

assert (
    manifest["subject_key"]
    .notna()
).all()


duplicates = manifest.duplicated(
    subset=[
        "subject_key",
        "bscan_idx",
    ],
    keep=False,
)

assert not duplicates.any(), (
    "Duplicate subject/B-scan pairs detected."
)


print("\nStructural validation: PASS")


# ---------------------------------------
# CV coverage
# ---------------------------------------

split = pd.read_csv(SPLIT)

expected = set(
    split["subject"].unique()
)

observed = set(
    manifest["subject_key"].unique()
)


missing = sorted(
    expected - observed
)

extra = sorted(
    observed - expected
)


print("\nCV SUBJECTS EXPECTED")
print(len(expected))


print("\nCV SUBJECTS FOUND")
print(
    len(expected & observed)
)


print("\nMISSING CV SUBJECTS")
print(
    missing if missing else "None"
)


print("\nNON-CV SUBJECTS PRESENT IN EXPORT")
print(
    extra if extra else "None"
)


assert len(missing) == 0, (
    "At least one CV subject has no exported slices."
)


# The expected extra subject is Control_1088,
# which remains outside quantitative CV pending
# spacing verification.

unexpected_extra = set(extra) - {
    "Control_1088"
}

assert not unexpected_extra, (
    f"Unexpected extra subjects: "
    f"{unexpected_extra}"
)


print("\nCV cohort coverage: PASS")


# ---------------------------------------
# File existence
# ---------------------------------------

for column in [
    "image_path",
    "rpedc_mask_path",
    "boundary_path",
]:

    missing_files = (
        ~manifest[column]
        .apply(
            lambda x:
            Path(x).exists()
        )
    ).sum()

    print(
        f"{column} missing files:",
        missing_files,
    )

    assert missing_files == 0


print("\nFile existence validation: PASS")


print("\nFINAL STATUS: PASS")