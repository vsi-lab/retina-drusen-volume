#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


AXIAL_SPACING_UM = 3.23


def build_rpedc_map(layer_maps):
    """
    layerMaps:
        [B-scan, A-scan, boundary]

    RPEDC thickness:
        boundary 2 - boundary 1

    Output:
        thickness map in microns
        shape = [B-scan, A-scan]
    """

    if layer_maps.ndim != 3:
        raise ValueError(
            f"Unexpected layerMaps shape: {layer_maps.shape}"
        )

    if layer_maps.shape[2] < 3:
        raise ValueError(
            "Need at least 3 boundary channels."
        )

    upper = layer_maps[:, :, 1].astype(
        np.float32
    )

    lower = layer_maps[:, :, 2].astype(
        np.float32
    )

    thickness_px = lower - upper

    invalid = (
        ~np.isfinite(upper)
        |
        ~np.isfinite(lower)
        |
        (thickness_px <= 0)
    )

    thickness_um = (
        thickness_px
        *
        AXIAL_SPACING_UM
    )

    thickness_um[
        invalid
    ] = np.nan

    return thickness_um


def process_group(root, group, out_dir):
    root = Path(root)

    files = sorted(
        root.rglob("*.mat")
    )

    rows = []

    for index, path in enumerate(
        files,
        start=1,
    ):

        print(
            f"\r{group}: "
            f"{index}/{len(files)}",
            end="",
            flush=True,
        )

        row = {
            "subject": path.stem,
            "filename": path.name,
            "group": group,
            "status": "unknown",
            "age": np.nan,
            "bscans": np.nan,
            "ascans": np.nan,
            "valid_fraction": np.nan,
            "mean_rpedc_um": np.nan,
            "median_rpedc_um": np.nan,
        }

        try:
            data = sio.loadmat(path)

            layer_maps = data[
                "layerMaps"
            ]

            age = float(
                np.asarray(
                    data["Age"]
                ).squeeze()
            )

            thickness = build_rpedc_map(
                layer_maps
            )

            valid = np.isfinite(
                thickness
            )

            row["age"] = age
            row["bscans"] = (
                thickness.shape[0]
            )
            row["ascans"] = (
                thickness.shape[1]
            )

            row["valid_fraction"] = (
                float(valid.mean())
            )

            row["mean_rpedc_um"] = (
                float(
                    np.nanmean(
                        thickness
                    )
                )
            )

            row["median_rpedc_um"] = (
                float(
                    np.nanmedian(
                        thickness
                    )
                )
            )

            output = (
                out_dir
                /
                group.lower()
                /
                f"{path.stem}.npy"
            )

            output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.save(
                output,
                thickness.astype(
                    np.float32
                ),
            )

            row["thickness_map"] = str(
                output
            )

            row["status"] = "ok"

        except Exception as exc:
            row["status"] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        rows.append(row)

    print()

    return rows


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
        "--out_dir",
        default=(
            "data/processed/"
            "rpedc_thickness_maps"
        ),
    )

    parser.add_argument(
        "--manifest",
        default=(
            "reports/normal_reference/"
            "rpedc_thickness_manifest.csv"
        ),
    )

    args = parser.parse_args()

    out_dir = Path(
        args.out_dir
    )

    rows = []

    rows.extend(
        process_group(
            args.amd_dir,
            "AMD",
            out_dir,
        )
    )

    rows.extend(
        process_group(
            args.control_dir,
            "Control",
            out_dir,
        )
    )

    df = pd.DataFrame(rows)

    manifest = Path(
        args.manifest
    )

    manifest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        manifest,
        index=False,
    )

    print()
    print("=" * 70)
    print("RPEDC THICKNESS MAP SUMMARY")
    print("=" * 70)

    valid = df[
        df["status"] == "ok"
    ]

    print(
        valid.groupby("group")[
            [
                "valid_fraction",
                "mean_rpedc_um",
                "median_rpedc_um",
            ]
        ].agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "median",
                "max",
            ]
        )
    )

    print()
    print(
        "Saved manifest:",
        manifest,
    )


if __name__ == "__main__":
    main()