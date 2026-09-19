#!/usr/bin/env python3

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


def file_hash(path, block_size=1024 * 1024):
    h = hashlib.md5()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(block_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def extract_age(age_value):
    if age_value is None:
        return np.nan

    try:
        arr = np.asarray(age_value).squeeze()

        if arr.size == 0:
            return np.nan

        return float(arr.flat[0])

    except Exception:
        return np.nan


def audit_file(path, group):
    row = {
        "group": group,
        "filename": path.name,
        "path": str(path),
        "size_mb": path.stat().st_size / (1024 ** 2),
        "status": "unknown",
        "age": np.nan,
        "image_shape": "",
        "layer_shape": "",
        "image_dtype": "",
        "layer_dtype": "",
        "layer_nan_fraction": np.nan,
        "keys": "",
        "md5": "",
    }

    try:
        data = sio.loadmat(path)

        keys = [
            key
            for key in data.keys()
            if not key.startswith("__")
        ]

        row["keys"] = ",".join(keys)

        images = data.get("images")
        layers = data.get("layerMaps")
        age = data.get("Age")

        if images is None:
            row["status"] = "missing_images"
            return row

        if layers is None:
            row["status"] = "missing_layerMaps"
            return row

        row["image_shape"] = str(tuple(images.shape))
        row["layer_shape"] = str(tuple(layers.shape))

        row["image_dtype"] = str(images.dtype)
        row["layer_dtype"] = str(layers.dtype)

        row["layer_nan_fraction"] = float(
            np.isnan(layers).mean()
        )

        row["age"] = extract_age(age)

        row["status"] = "ok"

    except Exception as exc:
        row["status"] = (
            "read_error:"
            + type(exc).__name__
            + ":"
            + str(exc)
        )

    try:
        row["md5"] = file_hash(path)

    except Exception:
        row["md5"] = "HASH_ERROR"

    return row


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--amd_dir",
        required=True,
    )

    parser.add_argument(
        "--control_dir",
        required=True,
    )

    parser.add_argument(
        "--out_csv",
        required=True,
    )

    args = parser.parse_args()

    amd_dir = Path(args.amd_dir)
    control_dir = Path(args.control_dir)

    if not amd_dir.exists():
        raise FileNotFoundError(
            f"AMD directory not found: {amd_dir}"
        )

    if not control_dir.exists():
        raise FileNotFoundError(
            f"Control directory not found: {control_dir}"
        )

    amd_files = sorted(
        amd_dir.rglob("*.mat")
    )

    control_files = sorted(
        control_dir.rglob("*.mat")
    )

    print("AMD MAT files found:", len(amd_files))
    print("Control MAT files found:", len(control_files))

    rows = []

    for index, path in enumerate(
        amd_files,
        start=1,
    ):
        print(
            f"\rAuditing AMD: {index}/{len(amd_files)}",
            end="",
            flush=True,
        )

        rows.append(
            audit_file(
                path=path,
                group="AMD",
            )
        )

    print()

    for index, path in enumerate(
        control_files,
        start=1,
    ):
        print(
            f"\rAuditing Control: {index}/{len(control_files)}",
            end="",
            flush=True,
        )

        rows.append(
            audit_file(
                path=path,
                group="Control",
            )
        )

    print()

    dataframe = pd.DataFrame(rows)

    output_path = Path(
        args.out_csv
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print_section(
        "FILE STATUS"
    )

    print(
        dataframe.groupby(
            [
                "group",
                "status",
            ]
        ).size()
    )

    print_section(
        "EXPECTED PUBLISHED COHORT"
    )

    print("AMD expected:     269")
    print("Control expected: 115")

    print_section(
        "CURRENT FILE COUNTS"
    )

    print(
        dataframe.groupby(
            "group"
        ).size()
    )

    print_section(
        "SUCCESSFULLY READ FILES"
    )

    print(
        dataframe[
            dataframe["status"] == "ok"
        ]
        .groupby("group")
        .size()
    )

    print_section(
        "IMAGE SHAPES"
    )

    print(
        dataframe.groupby(
            [
                "group",
                "image_shape",
            ]
        ).size()
    )

    print_section(
        "LAYER MAP SHAPES"
    )

    print(
        dataframe.groupby(
            [
                "group",
                "layer_shape",
            ]
        ).size()
    )

    print_section(
        "AGE SUMMARY"
    )

    print(
        dataframe.groupby(
            "group"
        )["age"]
        .describe()
    )

    print_section(
        "DUPLICATE FILE CHECK"
    )

    valid_hashes = dataframe[
        dataframe["md5"].notna()
        &
        (dataframe["md5"] != "")
        &
        (dataframe["md5"] != "HASH_ERROR")
    ]

    duplicates = valid_hashes[
        valid_hashes.duplicated(
            subset="md5",
            keep=False,
        )
    ].sort_values(
        "md5"
    )

    if duplicates.empty:
        print("No duplicate MAT files detected.")

    else:
        print(
            duplicates[
                [
                    "group",
                    "filename",
                    "md5",
                ]
            ].to_string(
                index=False
            )
        )

    print_section(
        "MISSING OR INVALID FILES"
    )

    invalid = dataframe[
        dataframe["status"] != "ok"
    ]

    if invalid.empty:
        print("No unreadable or structurally invalid MAT files.")

    else:
        print(
            invalid[
                [
                    "group",
                    "filename",
                    "status",
                ]
            ].to_string(
                index=False
            )
        )

    print_section(
        "SUMMARY"
    )

    amd_present = int(
        (
            dataframe["group"]
            == "AMD"
        ).sum()
    )

    control_present = int(
        (
            dataframe["group"]
            == "Control"
        ).sum()
    )

    print(
        f"AMD present: {amd_present}/269"
    )

    print(
        f"Control present: {control_present}/115"
    )

    print(
        f"AMD missing relative to published cohort: "
        f"{269 - amd_present}"
    )

    print(
        f"Control missing relative to published cohort: "
        f"{115 - control_present}"
    )

    print()

    if amd_present == 92 and control_present == 93:
        print(
            "IMPORTANT: the current extracted dataset itself "
            "contains only 92 AMD and 93 Control MAT files."
        )

        print(
            "This means the missing subjects are not being removed "
            "by the RPEDC exporter or train/test split."
        )

    print()

    print(
        "Saved inventory:",
        output_path,
    )


if __name__ == "__main__":
    main()