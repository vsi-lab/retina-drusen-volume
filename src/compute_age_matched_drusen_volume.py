#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Physical sampling supplied for this Duke dataset
LATERAL_SPACING_UM = 6.54
BSCAN_SPACING_UM = 67.0

# Threshold sensitivity analysis
K_VALUES = [
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
]

# Require a reasonable number of valid controls
# at each spatial location.
MIN_VALID_CONTROLS_PER_PIXEL = 3


def choose_age_matched_controls(
    controls,
    amd_age,
    min_controls=5,
):
    """
    Adaptive matching strategy.

    Start at ±5 years.
    If fewer than min_controls are available,
    expand to ±7, then ±10 years.
    """

    for window in [5, 7, 10]:

        matched = controls[
            (
                controls["age"]
                >= amd_age - window
            )
            &
            (
                controls["age"]
                <= amd_age + window
            )
        ].copy()

        if len(matched) >= min_controls:
            return matched, window

    # Extremely defensive fallback:
    # choose nearest controls by age.
    controls = controls.copy()

    controls["age_distance"] = np.abs(
        controls["age"] - amd_age
    )

    matched = (
        controls
        .sort_values("age_distance")
        .head(min_controls)
        .copy()
    )

    return matched, -1


def load_map(path):
    array = np.load(path)

    if array.ndim != 2:
        raise ValueError(
            f"Expected 2-D map, got {array.shape}: {path}"
        )

    return array.astype(
        np.float32
    )


def build_reference_maps(
    matched_controls,
):
    """
    Stack age-matched controls on the native
    fovea-centered Duke coordinate grid.

    NaNs remain missing.

    Returns:
      mean_map
      std_map
      count_map
    """

    arrays = []

    for _, row in matched_controls.iterrows():

        array = load_map(
            row["thickness_map"]
        )

        if array.shape != (100, 1000):
            continue

        arrays.append(array)

    if len(arrays) < 3:
        raise RuntimeError(
            "Fewer than 3 usable 100x1000 "
            "age-matched control maps."
        )

    stack = np.stack(
        arrays,
        axis=0,
    )

    count_map = np.sum(
        np.isfinite(stack),
        axis=0,
    )

    # Avoid noisy RuntimeWarnings in locations where
    # every control is missing.
    with np.errstate(
        invalid="ignore",
        divide="ignore",
    ):
        mean_map = np.nanmean(
            stack,
            axis=0,
        )

        std_map = np.nanstd(
            stack,
            axis=0,
            ddof=1,
        )

    insufficient = (
        count_map
        <
        MIN_VALID_CONTROLS_PER_PIXEL
    )

    mean_map[
        insufficient
    ] = np.nan

    std_map[
        insufficient
    ] = np.nan

    return (
        mean_map.astype(np.float32),
        std_map.astype(np.float32),
        count_map.astype(np.int16),
        len(arrays),
    )


def calculate_abnormal_volume(
    amd_map,
    mean_map,
    std_map,
    k,
):
    """
    Abnormal RPEDC thickness is defined as:

        AMD RPEDC >
        control mean + k * control SD

    Excess thickness is:

        AMD thickness - threshold

    Only positive excess contributes to volume.

    Volume element:
        excess axial thickness (um)
        * lateral spacing (um)
        * B-scan spacing (um)

    Result converted from um^3 to mm^3.
    """

    threshold = (
        mean_map
        +
        k * std_map
    )

    valid = (
        np.isfinite(amd_map)
        &
        np.isfinite(threshold)
    )

    excess_um = np.full(
        amd_map.shape,
        np.nan,
        dtype=np.float32,
    )

    difference = (
        amd_map[valid]
        -
        threshold[valid]
    )

    excess_um[valid] = np.maximum(
        difference,
        0.0,
    )

    abnormal = (
        valid
        &
        (amd_map > threshold)
    )

    excess_sum_um = float(
        np.nansum(excess_um)
    )

    volume_um3 = (
        excess_sum_um
        *
        LATERAL_SPACING_UM
        *
        BSCAN_SPACING_UM
    )

    volume_mm3 = (
        volume_um3
        / 1_000_000_000.0
    )

    valid_locations = int(
        valid.sum()
    )

    abnormal_locations = int(
        abnormal.sum()
    )

    abnormal_fraction = (
        abnormal_locations
        / valid_locations
        if valid_locations > 0
        else np.nan
    )

    return {
        "volume_mm3": volume_mm3,
        "abnormal_locations": abnormal_locations,
        "valid_locations": valid_locations,
        "abnormal_fraction": abnormal_fraction,
        "maximum_excess_um": (
            float(np.nanmax(excess_um))
            if np.any(np.isfinite(excess_um))
            else np.nan
        ),
        "mean_positive_excess_um": (
            float(
                np.nanmean(
                    excess_um[
                        excess_um > 0
                    ]
                )
            )
            if np.any(excess_um > 0)
            else 0.0
        ),
        "threshold_map": threshold,
        "excess_map": excess_um,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default=(
            "reports/normal_reference/"
            "rpedc_thickness_manifest.csv"
        ),
    )

    parser.add_argument(
        "--out_dir",
        default=(
            "reports/drusen_volume/"
            "reference_formula"
        ),
    )

    parser.add_argument(
        "--save_k3_maps",
        action="store_true",
    )

    args = parser.parse_args()

    manifest = pd.read_csv(
        args.manifest
    )

    manifest = manifest[
        manifest["status"] == "ok"
    ].copy()

    amd = manifest[
        manifest["group"] == "AMD"
    ].copy()

    controls = manifest[
        manifest["group"] == "Control"
    ].copy()

    # The normal atlas must use the common
    # 100 x 1000 coordinate grid.
    controls_common = controls[
        (controls["bscans"] == 100)
        &
        (controls["ascans"] == 1000)
    ].copy()

    excluded_controls = controls[
        ~controls.index.isin(
            controls_common.index
        )
    ].copy()

    print("=" * 72)
    print("NORMAL REFERENCE INPUT")
    print("=" * 72)

    print(
        "AMD subjects:",
        len(amd),
    )

    print(
        "Control subjects total:",
        len(controls),
    )

    print(
        "Controls on common 100x1000 grid:",
        len(controls_common),
    )

    print(
        "Controls excluded for shape mismatch:",
        len(excluded_controls),
    )

    if len(excluded_controls):

        print(
            excluded_controls[
                [
                    "subject",
                    "age",
                    "bscans",
                    "ascans",
                ]
            ].to_string(
                index=False
            )
        )

    output_root = Path(
        args.out_dir
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    map_root = (
        output_root
        /
        "k3_maps"
    )

    if args.save_k3_maps:
        map_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    rows = []
    matching_rows = []

    for index, (_, amd_row) in enumerate(
        amd.iterrows(),
        start=1,
    ):

        print(
            f"\rAMD {index}/{len(amd)}: "
            f"{amd_row['subject']}",
            end="",
            flush=True,
        )

        amd_age = float(
            amd_row["age"]
        )

        amd_map = load_map(
            amd_row["thickness_map"]
        )

        if amd_map.shape != (100, 1000):
            print()
            print(
                "Skipping non-standard AMD map:",
                amd_row["subject"],
                amd_map.shape,
            )
            continue

        matched, age_window = (
            choose_age_matched_controls(
                controls_common,
                amd_age,
                min_controls=5,
            )
        )

        (
            mean_map,
            std_map,
            count_map,
            number_used,
        ) = build_reference_maps(
            matched
        )

        matching_rows.append(
            {
                "subject": amd_row["subject"],
                "amd_age": amd_age,
                "age_window_years": age_window,
                "number_matched_controls": len(
                    matched
                ),
                "number_controls_used": number_used,
                "matched_control_subjects": "|".join(
                    matched[
                        "subject"
                    ].astype(str)
                ),
            }
        )

        for k in K_VALUES:

            result = (
                calculate_abnormal_volume(
                    amd_map,
                    mean_map,
                    std_map,
                    k,
                )
            )

            rows.append(
                {
                    "subject": amd_row[
                        "subject"
                    ],
                    "age": amd_age,
                    "k_sd": k,
                    "age_window_years": (
                        age_window
                    ),
                    "number_controls": (
                        number_used
                    ),
                    "volume_mm3": result[
                        "volume_mm3"
                    ],
                    "abnormal_locations": (
                        result[
                            "abnormal_locations"
                        ]
                    ),
                    "valid_locations": result[
                        "valid_locations"
                    ],
                    "abnormal_fraction": (
                        result[
                            "abnormal_fraction"
                        ]
                    ),
                    "maximum_excess_um": (
                        result[
                            "maximum_excess_um"
                        ]
                    ),
                    "mean_positive_excess_um": (
                        result[
                            "mean_positive_excess_um"
                        ]
                    ),
                }
            )

            # Save only the primary 3-SD maps
            # to avoid unnecessary storage use.
            if (
                args.save_k3_maps
                and
                np.isclose(k, 3.0)
            ):

                subject_dir = (
                    map_root
                    /
                    amd_row["subject"]
                )

                subject_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                np.save(
                    subject_dir
                    /
                    "control_mean_um.npy",
                    mean_map,
                )

                np.save(
                    subject_dir
                    /
                    "control_std_um.npy",
                    std_map,
                )

                np.save(
                    subject_dir
                    /
                    "valid_control_count.npy",
                    count_map,
                )

                np.save(
                    subject_dir
                    /
                    "threshold_3sd_um.npy",
                    result[
                        "threshold_map"
                    ].astype(
                        np.float32
                    ),
                )

                np.save(
                    subject_dir
                    /
                    "excess_3sd_um.npy",
                    result[
                        "excess_map"
                    ].astype(
                        np.float32
                    ),
                )

    print()

    results = pd.DataFrame(
        rows
    )

    matching = pd.DataFrame(
        matching_rows
    )

    result_csv = (
        output_root
        /
        "drusen_volume_threshold_sensitivity.csv"
    )

    matching_csv = (
        output_root
        /
        "age_matching_details.csv"
    )

    results.to_csv(
        result_csv,
        index=False,
    )

    matching.to_csv(
        matching_csv,
        index=False,
    )

    print()
    print("=" * 72)
    print("THRESHOLD SENSITIVITY SUMMARY")
    print("=" * 72)

    summary = (
        results
        .groupby(
            "k_sd"
        )[
            "volume_mm3"
        ]
        .agg(
            [
                "count",
                "mean",
                "std",
                "median",
                "min",
                "max",
            ]
        )
    )

    print(summary)

    summary.to_csv(
        output_root
        /
        "threshold_sensitivity_summary.csv"
    )

    print()
    print("=" * 72)
    print("AGE WINDOW USE")
    print("=" * 72)

    print(
        matching[
            "age_window_years"
        ].value_counts(
            dropna=False
        ).sort_index()
    )

    print()
    print("Saved:")
    print(result_csv)
    print(matching_csv)


if __name__ == "__main__":
    main()