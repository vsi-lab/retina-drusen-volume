#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.ndimage import gaussian_filter


def fill_nan_1d(values):
    values = np.asarray(values, dtype=float)
    x = np.arange(len(values))

    valid = np.isfinite(values)

    if valid.sum() < 2:
        return None

    out = values.copy()

    out[~valid] = np.interp(
        x[~valid],
        x[valid],
        values[valid],
    )

    return out


def estimate_fovea(layer_maps):
    """
    layer_maps expected shape:
        B-scans x A-scans x boundaries

    We use boundary 0 to boundary 2 as a
    full-retinal-thickness proxy.

    Fovea is estimated as the minimum of a
    heavily smoothed 2-D retinal thickness map.
    """

    if layer_maps.ndim != 3:
        raise ValueError(
            f"Unexpected layerMaps ndim: {layer_maps.ndim}"
        )

    if layer_maps.shape[2] < 3:
        raise ValueError(
            f"Expected >=3 boundaries, got {layer_maps.shape}"
        )

    upper = layer_maps[:, :, 0].astype(float)
    lower = layer_maps[:, :, 2].astype(float)

    thickness = lower - upper

    thickness[
        ~np.isfinite(thickness)
    ] = np.nan

    thickness[
        thickness <= 0
    ] = np.nan

    # Fill gaps independently along A-scan direction
    filled = np.full_like(
        thickness,
        np.nan,
        dtype=float,
    )

    for bscan in range(thickness.shape[0]):
        row = fill_nan_1d(
            thickness[bscan]
        )

        if row is not None:
            filled[bscan] = row

    # Fill remaining missing values by median
    finite = np.isfinite(filled)

    if finite.sum() == 0:
        raise ValueError(
            "No valid retinal thickness values."
        )

    median_value = np.nanmedian(filled)

    filled[
        ~finite
    ] = median_value

    # Smooth strongly because we want the broad foveal depression,
    # not local layer irregularities.
    smoothed = gaussian_filter(
        filled,
        sigma=(3.0, 25.0),
    )

    # Avoid image borders where interpolation can create false minima.
    b_margin = max(
        5,
        int(round(smoothed.shape[0] * 0.10)),
    )

    x_margin = max(
        50,
        int(round(smoothed.shape[1] * 0.10)),
    )

    search = smoothed[
        b_margin:
        smoothed.shape[0] - b_margin,
        x_margin:
        smoothed.shape[1] - x_margin,
    ]

    index = np.nanargmin(search)

    local_bscan, local_x = np.unravel_index(
        index,
        search.shape,
    )

    fovea_bscan = (
        local_bscan
        + b_margin
    )

    fovea_x = (
        local_x
        + x_margin
    )

    return {
        "fovea_bscan": int(fovea_bscan),
        "fovea_x": int(fovea_x),
        "minimum_smoothed_thickness_px":
            float(
                smoothed[
                    fovea_bscan,
                    fovea_x,
                ]
            ),
    }


def process_directory(root, group):
    rows = []

    files = sorted(
        Path(root).rglob("*.mat")
    )

    for i, path in enumerate(files, 1):
        print(
            f"\r{group}: {i}/{len(files)}",
            end="",
            flush=True,
        )

        row = {
            "subject": path.stem,
            "filename": path.name,
            "group": group,
            "status": "unknown",
            "age": np.nan,
            "fovea_bscan": np.nan,
            "fovea_x": np.nan,
            "minimum_smoothed_thickness_px": np.nan,
        }

        try:
            data = sio.loadmat(path)

            layers = data["layerMaps"]

            age = np.asarray(
                data["Age"]
            ).squeeze()

            row["age"] = float(age)

            result = estimate_fovea(
                layers
            )

            row.update(result)
            row["status"] = "ok"

        except Exception as exc:
            row["status"] = (
                f"{type(exc).__name__}: {exc}"
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
        "--out_csv",
        required=True,
    )

    args = parser.parse_args()

    rows = []

    rows.extend(
        process_directory(
            args.amd_dir,
            "AMD",
        )
    )

    rows.extend(
        process_directory(
            args.control_dir,
            "Control",
        )
    )

    df = pd.DataFrame(rows)

    out = Path(args.out_csv)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        out,
        index=False,
    )

    print()
    print("=" * 70)
    print("FOVEAL CENTER SUMMARY")
    print("=" * 70)

    valid = df[
        df["status"] == "ok"
    ]

    print(
        valid.groupby("group")[
            [
                "fovea_bscan",
                "fovea_x",
                "minimum_smoothed_thickness_px",
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
    print("Saved:", out)


if __name__ == "__main__":
    main()