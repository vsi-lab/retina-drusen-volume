#!/usr/bin/env python3

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


NAME_PATTERN = re.compile(
    r"Farsiu_Ophthalmology_2013_"
    r"(AMD|Control)_Subject_(\d+)",
    re.IGNORECASE,
)


def sha256_file(path, block_size=4 * 1024 * 1024):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def parse_name(path):
    match = NAME_PATTERN.search(path.stem)

    if match is None:
        return None, None

    raw_group = match.group(1)

    group = (
        "AMD"
        if raw_group.lower() == "amd"
        else "Control"
    )

    subject_id = int(
        match.group(2)
    )

    return group, subject_id


def inspect_mat(path, source_tag):
    group, subject_id = parse_name(path)

    row = {
        "subject_key": "",
        "subject_id": subject_id,
        "group": group,
        "filename": path.name,
        "full_path": str(path.resolve()),
        "source_tag": source_tag,
        "size_mb": path.stat().st_size / (1024 ** 2),
        "sha256": "",
        "read_status": "unknown",
        "age": np.nan,
        "image_shape": "",
        "layer_shape": "",
        "num_bscans": np.nan,
        "image_height": np.nan,
        "num_ascans": np.nan,
        "layer_channels": np.nan,
        "has_images": False,
        "has_layerMaps": False,
        "has_age": False,
        "inclusion_status": "review",
        "exclusion_reason": "",
    }

    if group is not None and subject_id is not None:
        row["subject_key"] = (
            f"{group}_{subject_id}"
        )
    else:
        row["exclusion_reason"] = (
            "filename_not_recognized"
        )

    try:
        data = sio.loadmat(
            path,
            variable_names=[
                "images",
                "layerMaps",
                "Age",
            ],
        )

        images = data.get("images")
        layers = data.get("layerMaps")
        age = data.get("Age")

        row["has_images"] = (
            images is not None
        )

        row["has_layerMaps"] = (
            layers is not None
        )

        row["has_age"] = (
            age is not None
        )

        if images is None:
            row["read_status"] = "missing_images"
            row["inclusion_status"] = "exclude"
            row["exclusion_reason"] = "missing_images"
            return row

        if layers is None:
            row["read_status"] = "missing_layerMaps"
            row["inclusion_status"] = "exclude"
            row["exclusion_reason"] = "missing_layerMaps"
            return row

        row["image_shape"] = str(
            tuple(images.shape)
        )

        row["layer_shape"] = str(
            tuple(layers.shape)
        )

        if images.ndim == 3:
            row["image_height"] = images.shape[0]
            row["num_ascans"] = images.shape[1]
            row["num_bscans"] = images.shape[2]

        if layers.ndim == 3:
            row["layer_channels"] = layers.shape[2]

        if age is not None:
            try:
                row["age"] = float(
                    np.asarray(age).squeeze()
                )
            except Exception:
                pass

        row["read_status"] = "ok"

        # Structural consistency checks
        if images.ndim != 3 or layers.ndim != 3:
            row["inclusion_status"] = "review"
            row["exclusion_reason"] = (
                "unexpected_dimensions"
            )

        elif images.shape[1] != layers.shape[1]:
            row["inclusion_status"] = "review"
            row["exclusion_reason"] = (
                "ascan_dimension_mismatch"
            )

        elif images.shape[2] != layers.shape[0]:
            row["inclusion_status"] = "review"
            row["exclusion_reason"] = (
                "bscan_dimension_mismatch"
            )

        elif layers.shape[2] < 3:
            row["inclusion_status"] = "review"
            row["exclusion_reason"] = (
                "fewer_than_3_boundaries"
            )

        elif images.shape[2] != 100:
            # Do NOT automatically exclude it.
            # Its physical spacing first needs verification.
            row["inclusion_status"] = "review_spacing"
            row["exclusion_reason"] = (
                "nonstandard_bscan_count"
            )

        else:
            row["inclusion_status"] = "eligible"
            row["exclusion_reason"] = ""

    except Exception as exc:
        row["read_status"] = (
            f"{type(exc).__name__}: {exc}"
        )

        row["inclusion_status"] = "exclude"
        row["exclusion_reason"] = "mat_read_failure"

    return row


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True,
    )

    parser.add_argument(
        "--source_tag",
        required=True,
    )

    parser.add_argument(
        "--out_csv",
        required=True,
    )

    parser.add_argument(
        "--hash",
        action="store_true",
    )

    args = parser.parse_args()

    root = Path(args.root)

    if not root.exists():
        raise FileNotFoundError(root)

    files = sorted(
        root.rglob("*.mat")
    )

    print("Root:", root)
    print("MAT files found:", len(files))
    print("Source tag:", args.source_tag)

    rows = []

    for index, path in enumerate(
        files,
        start=1,
    ):
        print(
            f"\rInspecting {index}/{len(files)}",
            end="",
            flush=True,
        )

        row = inspect_mat(
            path,
            args.source_tag,
        )

        if args.hash:
            try:
                row["sha256"] = sha256_file(path)
            except Exception as exc:
                row["sha256"] = (
                    f"HASH_ERROR:{exc}"
                )

        rows.append(row)

    print()

    df = pd.DataFrame(rows)

    output = Path(args.out_csv)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output,
        index=False,
    )

    print()
    print("=" * 75)
    print("GROUP COUNTS")
    print("=" * 75)

    print(
        df.groupby("group")
        .size()
    )

    print()
    print("=" * 75)
    print("INCLUSION STATUS")
    print("=" * 75)

    print(
        df.groupby(
            [
                "group",
                "inclusion_status",
            ]
        ).size()
    )

    print()
    print("=" * 75)
    print("B-SCAN COUNTS")
    print("=" * 75)

    print(
        df.groupby(
            [
                "group",
                "num_bscans",
            ]
        ).size()
    )

    duplicate_subjects = df[
        df["subject_key"].duplicated(
            keep=False
        )
    ]

    print()
    print("=" * 75)
    print("DUPLICATE SUBJECT KEYS")
    print("=" * 75)

    if duplicate_subjects.empty:
        print("None")
    else:
        print(
            duplicate_subjects[
                [
                    "subject_key",
                    "filename",
                    "full_path",
                ]
            ].to_string(index=False)
        )

    print()
    print("Saved:", output)


if __name__ == "__main__":
    main()