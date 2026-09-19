#!/usr/bin/env python3

from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image


OLD_MANIFEST = Path(
    "reports/pre_cv_archive/"
    "duke_rpedc_manifest_185subjects.csv"
)

NEW_MANIFEST = Path(
    "data/manifests/"
    "duke_full_rpedc_manifest.csv"
)

SEED = 42
PIXEL_SAMPLE_SIZE = 500


# ---------------------------------------------------------
# Basic file checks
# ---------------------------------------------------------

if not OLD_MANIFEST.exists():
    raise FileNotFoundError(
        f"Old manifest not found: {OLD_MANIFEST}"
    )

if not NEW_MANIFEST.exists():
    raise FileNotFoundError(
        f"New manifest not found: {NEW_MANIFEST}"
    )


old = pd.read_csv(OLD_MANIFEST)
new = pd.read_csv(NEW_MANIFEST)


print("=" * 80)
print("OLD VS FULL RPEDC EXPORT REPRODUCIBILITY")
print("=" * 80)

print("\nOld manifest rows:", len(old))
print("Old subjects:", old["subject"].nunique())

print("\nNew manifest rows:", len(new))
print("New subjects:", new["subject"].nunique())


# ---------------------------------------------------------
# Check unique subject + B-scan combinations
# ---------------------------------------------------------

old_duplicates = old.duplicated(
    subset=["subject", "bscan_idx"],
    keep=False,
)

new_duplicates = new.duplicated(
    subset=["subject", "bscan_idx"],
    keep=False,
)

if old_duplicates.any():
    raise RuntimeError(
        "Duplicate subject/B-scan pairs found in old manifest."
    )

if new_duplicates.any():
    raise RuntimeError(
        "Duplicate subject/B-scan pairs found in new manifest."
    )


# ---------------------------------------------------------
# Join old slices to corresponding full-release slices
# ---------------------------------------------------------

merged = old.merge(
    new,
    on=[
        "subject",
        "bscan_idx",
    ],
    how="left",
    suffixes=(
        "_old",
        "_new",
    ),
    indicator=True,
    validate="one_to_one",
)


found = (
    merged["_merge"] == "both"
).sum()

missing = (
    merged["_merge"] != "both"
).sum()


print("\n=== SLICE MATCHING ===")
print("Old manifest rows:", len(old))
print("Rows found in new export:", found)
print("Rows missing from new export:", missing)


if missing > 0:

    print("\nMissing examples:")

    print(
        merged[
            merged["_merge"] != "both"
        ][
            [
                "subject",
                "bscan_idx",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    raise RuntimeError(
        "Some old exported slices are missing from the full export."
    )


# ---------------------------------------------------------
# Compare manifest-level values for ALL overlapping slices
# ---------------------------------------------------------

print("\n=== MANIFEST VALUE CHECKS ===")


columns_to_compare = [
    "height",
    "width",
    "valid_columns",
    "mask_pixels",
]


for column in columns_to_compare:

    old_values = merged[
        f"{column}_old"
    ].to_numpy()

    new_values = merged[
        f"{column}_new"
    ].to_numpy()

    identical = np.array_equal(
        old_values,
        new_values,
    )

    print(
        f"{column} identical:",
        identical,
    )

    if not identical:

        differences = np.where(
            old_values != new_values
        )[0]

        print(
            f"  Differing rows: {len(differences)}"
        )

        raise RuntimeError(
            f"{column} differs between exports."
        )


# ---------------------------------------------------------
# Compare actual PNG/NPZ outputs on fixed random sample
# ---------------------------------------------------------

rng = random.Random(SEED)

sample_size = min(
    PIXEL_SAMPLE_SIZE,
    len(merged),
)

sample_indices = rng.sample(
    range(len(merged)),
    sample_size,
)


image_failures = []
mask_failures = []
boundary_failures = []


print()
print(
    f"Comparing actual exported files for "
    f"{sample_size} reproducibly sampled slices..."
)


for count, i in enumerate(
    sample_indices,
    start=1,
):

    row = merged.iloc[i]

    # -----------------------------------------------------
    # Images
    # -----------------------------------------------------

    old_image_path = Path(
        row["image_path_old"]
    )

    new_image_path = Path(
        row["image_path_new"]
    )

    if not old_image_path.exists():
        raise FileNotFoundError(
            f"Missing old image: {old_image_path}"
        )

    if not new_image_path.exists():
        raise FileNotFoundError(
            f"Missing new image: {new_image_path}"
        )

    old_image = np.asarray(
        Image.open(old_image_path)
    )

    new_image = np.asarray(
        Image.open(new_image_path)
    )

    if not np.array_equal(
        old_image,
        new_image,
    ):

        image_failures.append(
            (
                row["subject"],
                row["bscan_idx"],
            )
        )


    # -----------------------------------------------------
    # RPEDC masks
    # -----------------------------------------------------

    old_mask_path = Path(
        row["rpedc_mask_path_old"]
    )

    new_mask_path = Path(
        row["rpedc_mask_path_new"]
    )

    old_mask = np.asarray(
        Image.open(old_mask_path)
    )

    new_mask = np.asarray(
        Image.open(new_mask_path)
    )

    if not np.array_equal(
        old_mask,
        new_mask,
    ):

        mask_failures.append(
            (
                row["subject"],
                row["bscan_idx"],
            )
        )


    # -----------------------------------------------------
    # Boundary files
    # -----------------------------------------------------

    old_boundary_path = Path(
        row["boundary_path_old"]
    )

    new_boundary_path = Path(
        row["boundary_path_new"]
    )

    old_boundary = np.load(
        old_boundary_path
    )

    new_boundary = np.load(
        new_boundary_path
    )


    required_keys = [
        "upper_boundary",
        "lower_boundary",
        "valid_columns",
    ]


    boundary_same = True


    for key in required_keys:

        if key not in old_boundary.files:
            raise KeyError(
                f"{key} missing from {old_boundary_path}"
            )

        if key not in new_boundary.files:
            raise KeyError(
                f"{key} missing from {new_boundary_path}"
            )


        a = old_boundary[key]
        b = new_boundary[key]


        # Works for floating arrays containing NaNs
        # as well as integer/bool arrays.
        if (
            np.issubdtype(
                a.dtype,
                np.floating,
            )
            or
            np.issubdtype(
                b.dtype,
                np.floating,
            )
        ):

            same = np.allclose(
                a,
                b,
                rtol=0,
                atol=0,
                equal_nan=True,
            )

        else:

            same = np.array_equal(
                a,
                b,
            )


        if not same:
            boundary_same = False
            break


    if not boundary_same:

        boundary_failures.append(
            (
                row["subject"],
                row["bscan_idx"],
            )
        )


    if (
        count % 100 == 0
        or
        count == sample_size
    ):

        print(
            f"Checked {count}/{sample_size}"
        )


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n=== PIXEL/BOUNDARY COMPARISON ===")

print(
    "Image failures:",
    len(image_failures),
)

print(
    "Mask failures:",
    len(mask_failures),
)

print(
    "Boundary failures:",
    len(boundary_failures),
)


if image_failures:

    print(
        "\nFirst image failures:",
        image_failures[:10],
    )


if mask_failures:

    print(
        "\nFirst mask failures:",
        mask_failures[:10],
    )


if boundary_failures:

    print(
        "\nFirst boundary failures:",
        boundary_failures[:10],
    )


assert len(image_failures) == 0
assert len(mask_failures) == 0
assert len(boundary_failures) == 0


print()
print("=" * 80)
print("OLD → FULL EXPORT REPRODUCIBILITY: PASS")
print("=" * 80)