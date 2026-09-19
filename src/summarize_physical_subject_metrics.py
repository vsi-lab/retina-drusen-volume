#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


AXIAL_SPACING_UM = 3.23

MODELS = [
    "unet",
    "unetpp",
    "fpn",
    "deeplabv3plus",
]


def bootstrap_mean_ci(values, n_boot=10000, seed=42):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        values,
        size=(n_boot, len(values)),
        replace=True,
    )

    boot_means = samples.mean(axis=1)

    low, high = np.percentile(
        boot_means,
        [2.5, 97.5],
    )

    return float(low), float(high)


def bootstrap_paired_difference_ci(
    values_a,
    values_b,
    n_boot=10000,
    seed=42,
):
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)

    valid = (
        np.isfinite(values_a)
        &
        np.isfinite(values_b)
    )

    differences = (
        values_a[valid]
        -
        values_b[valid]
    )

    if len(differences) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        differences,
        size=(n_boot, len(differences)),
        replace=True,
    )

    means = samples.mean(axis=1)

    low, high = np.percentile(
        means,
        [2.5, 97.5],
    )

    return float(low), float(high)


def add_physical_thickness_columns(df):
    df = df.copy()

    df["thickness_mae_um"] = (
        df["thickness_mae"]
        * AXIAL_SPACING_UM
    )

    df["thickness_rmse_um"] = (
        df["thickness_rmse"]
        * AXIAL_SPACING_UM
    )

    df["thickness_bias_um"] = (
        df["thickness_bias"]
        * AXIAL_SPACING_UM
    )

    return df


def summarize_metric(df, metric):
    values = df[metric].to_numpy(dtype=float)

    values = values[np.isfinite(values)]

    ci_low, ci_high = bootstrap_mean_ci(values)

    return {
        "metric": metric,
        "n_subjects": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def main():

    reports_root = Path(
        "reports/segmentation"
    )

    output_root = Path(
        "reports/physical_metrics"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_data = {}
    summary_rows = []

    metrics_to_summarize = [
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "balanced_accuracy",
        "mcc",
        "thickness_mae_um",
        "thickness_rmse_um",
        "thickness_bias_um",
    ]

    print("=" * 80)
    print("SUBJECT-LEVEL PHYSICAL METRIC SUMMARY")
    print("=" * 80)

    for model in MODELS:

        csv_path = (
            reports_root
            / model
            / "subject_metrics.csv"
        )

        if not csv_path.exists():
            print(
                f"WARNING: missing {csv_path}"
            )
            continue

        df = pd.read_csv(csv_path)

        required = {
            "subject",
            "group",
            "dice",
            "iou",
            "thickness_mae",
            "thickness_rmse",
            "thickness_bias",
        }

        missing = (
            required
            -
            set(df.columns)
        )

        if missing:
            raise RuntimeError(
                f"{model} missing columns: "
                f"{sorted(missing)}"
            )

        if df["subject"].duplicated().any():
            duplicates = df.loc[
                df["subject"].duplicated(),
                "subject",
            ].tolist()

            raise RuntimeError(
                f"{model}: duplicate subject rows: "
                f"{duplicates}"
            )

        df = add_physical_thickness_columns(
            df
        )

        model_data[model] = df

        physical_csv = (
            output_root
            / f"{model}_subject_metrics_physical.csv"
        )

        df.to_csv(
            physical_csv,
            index=False,
        )

        print()
        print(
            f"{model.upper()} | "
            f"n={len(df)}"
        )

        for metric in metrics_to_summarize:

            if metric not in df.columns:
                continue

            result = summarize_metric(
                df,
                metric,
            )

            result["model"] = model

            summary_rows.append(result)

            if metric in [
                "thickness_mae_um",
                "thickness_rmse_um",
                "thickness_bias_um",
            ]:
                units = "µm"
            else:
                units = ""

            print(
                f"{metric:26s} "
                f"{result['mean']:.3f} "
                f"[95% CI "
                f"{result['ci95_low']:.3f}, "
                f"{result['ci95_high']:.3f}] "
                f"{units}"
            )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        output_root
        / "model_subject_summary_95ci.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("PAIRED U-NET++ VS DEEPLABV3+")
    print("=" * 80)

    a = model_data["unetpp"]
    b = model_data["deeplabv3plus"]

    merged = a.merge(
        b,
        on=[
            "subject",
            "group",
        ],
        suffixes=(
            "_unetpp",
            "_deeplab",
        ),
        validate="one_to_one",
    )

    print(
        "Paired subjects:",
        len(merged),
    )

    paired_metrics = [
        "dice",
        "iou",
        "mcc",
        "thickness_mae_um",
        "thickness_rmse_um",
    ]

    paired_rows = []

    for metric in paired_metrics:

        a_col = (
            f"{metric}_unetpp"
        )

        b_col = (
            f"{metric}_deeplab"
        )

        values_a = merged[
            a_col
        ].to_numpy(dtype=float)

        values_b = merged[
            b_col
        ].to_numpy(dtype=float)

        valid = (
            np.isfinite(values_a)
            &
            np.isfinite(values_b)
        )

        values_a = values_a[valid]
        values_b = values_b[valid]

        differences = (
            values_a
            -
            values_b
        )

        ci_low, ci_high = (
            bootstrap_paired_difference_ci(
                values_a,
                values_b,
            )
        )

        try:
            statistic, p_value = wilcoxon(
                values_a,
                values_b,
                alternative="two-sided",
            )
        except ValueError:
            statistic = np.nan
            p_value = np.nan

        row = {
            "metric": metric,
            "n_subjects": len(values_a),
            "unetpp_mean": float(
                np.mean(values_a)
            ),
            "deeplabv3plus_mean": float(
                np.mean(values_b)
            ),
            "paired_difference_mean": float(
                np.mean(differences)
            ),
            "paired_difference_median": float(
                np.median(differences)
            ),
            "difference_95ci_low": ci_low,
            "difference_95ci_high": ci_high,
            "wilcoxon_statistic": statistic,
            "wilcoxon_p_value": p_value,
        }

        paired_rows.append(row)

        print()
        print(metric)

        print(
            "  U-Net++ mean:       ",
            f"{row['unetpp_mean']:.4f}",
        )

        print(
            "  DeepLabV3+ mean:    ",
            f"{row['deeplabv3plus_mean']:.4f}",
        )

        print(
            "  Paired difference:  ",
            f"{row['paired_difference_mean']:.4f}",
        )

        print(
            "  Difference 95% CI:  ",
            f"[{ci_low:.4f}, "
            f"{ci_high:.4f}]",
        )

        print(
            "  Wilcoxon p-value:   ",
            f"{p_value:.6f}",
        )

    paired_df = pd.DataFrame(
        paired_rows
    )

    paired_path = Path(
        "reports/paired_comparison/"
        "unetpp_vs_deeplabv3plus.csv"
    )

    paired_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paired_df.to_csv(
        paired_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(summary_path)
    print(paired_path)


if __name__ == "__main__":
    main()