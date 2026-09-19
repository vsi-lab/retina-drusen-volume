#!/usr/bin/env python3

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    spearmanr,
    wilcoxon,
)


INPUT_DIR = Path(
    "reports/drusen_volume/predicted_formula"
)

OUTPUT_DIR = Path(
    "reports/drusen_volume/agreement_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS = [
    "unetpp",
    "deeplabv3plus",
]

K_VALUES = [
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
]

BOOTSTRAP_ITERATIONS = 10000
SEED = 42


def rmse(reference, prediction):

    reference = np.asarray(
        reference,
        dtype=float,
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                (
                    prediction
                    -
                    reference
                ) ** 2
            )
        )
    )


def mae(reference, prediction):

    return float(
        np.mean(
            np.abs(
                np.asarray(prediction)
                -
                np.asarray(reference)
            )
        )
    )


def bland_altman(reference, prediction):

    reference = np.asarray(
        reference,
        dtype=float,
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    differences = (
        prediction
        -
        reference
    )

    averages = (
        prediction
        +
        reference
    ) / 2.0

    bias = float(
        np.mean(differences)
    )

    sd_difference = float(
        np.std(
            differences,
            ddof=1,
        )
    )

    lower_loa = (
        bias
        -
        1.96 * sd_difference
    )

    upper_loa = (
        bias
        +
        1.96 * sd_difference
    )

    return {
        "bias_mm3": bias,
        "difference_sd_mm3": sd_difference,
        "lower_loa_mm3": lower_loa,
        "upper_loa_mm3": upper_loa,
        "averages": averages,
        "differences": differences,
    }


def icc_absolute_agreement(reference, prediction):
    """
    ICC(A,1):
    two-way random-effects,
    absolute-agreement,
    single-measure ICC.

    Rows = subjects
    Columns = methods:
        reference and prediction.
    """

    data = np.column_stack(
        [
            np.asarray(
                reference,
                dtype=float,
            ),
            np.asarray(
                prediction,
                dtype=float,
            ),
        ]
    )

    n, k = data.shape

    if n < 2:
        return np.nan

    grand_mean = np.mean(data)

    row_means = np.mean(
        data,
        axis=1,
    )

    column_means = np.mean(
        data,
        axis=0,
    )

    ss_rows = (
        k
        *
        np.sum(
            (
                row_means
                -
                grand_mean
            ) ** 2
        )
    )

    ss_columns = (
        n
        *
        np.sum(
            (
                column_means
                -
                grand_mean
            ) ** 2
        )
    )

    ss_total = np.sum(
        (
            data
            -
            grand_mean
        ) ** 2
    )

    ss_error = (
        ss_total
        -
        ss_rows
        -
        ss_columns
    )

    ms_rows = (
        ss_rows
        /
        (n - 1)
    )

    ms_columns = (
        ss_columns
        /
        (k - 1)
    )

    ms_error = (
        ss_error
        /
        (
            (n - 1)
            *
            (k - 1)
        )
    )

    denominator = (
        ms_rows
        +
        (k - 1) * ms_error
        +
        (
            k
            *
            (ms_columns - ms_error)
            /
            n
        )
    )

    if denominator == 0:
        return np.nan

    return float(
        (
            ms_rows
            -
            ms_error
        )
        /
        denominator
    )


def safe_pearson(reference, prediction):

    try:
        result = pearsonr(
            reference,
            prediction,
        )

        return float(
            result.statistic
        )

    except Exception:
        return np.nan


def safe_spearman(reference, prediction):

    try:
        result = spearmanr(
            reference,
            prediction,
        )

        return float(
            result.statistic
        )

    except Exception:
        return np.nan


def calculate_metrics(reference, prediction):

    reference = np.asarray(
        reference,
        dtype=float,
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    signed_error = (
        prediction
        -
        reference
    )

    absolute_error = np.abs(
        signed_error
    )

    relative_error = (
        100.0
        *
        absolute_error
        /
        reference
    )

    ba = bland_altman(
        reference,
        prediction,
    )

    return {
        "n_subjects": len(reference),

        "reference_mean_mm3": float(
            np.mean(reference)
        ),

        "reference_median_mm3": float(
            np.median(reference)
        ),

        "prediction_mean_mm3": float(
            np.mean(prediction)
        ),

        "prediction_median_mm3": float(
            np.median(prediction)
        ),

        "mae_mm3": mae(
            reference,
            prediction,
        ),

        "rmse_mm3": rmse(
            reference,
            prediction,
        ),

        "mean_signed_bias_mm3": float(
            np.mean(signed_error)
        ),

        "median_absolute_error_mm3": float(
            np.median(
                absolute_error
            )
        ),

        "mean_relative_error_percent": float(
            np.mean(
                relative_error
            )
        ),

        "median_relative_error_percent": float(
            np.median(
                relative_error
            )
        ),

        "pearson_r": safe_pearson(
            reference,
            prediction,
        ),

        "spearman_rho": safe_spearman(
            reference,
            prediction,
        ),

        "icc_a1": icc_absolute_agreement(
            reference,
            prediction,
        ),

        "bland_altman_bias_mm3": (
            ba["bias_mm3"]
        ),

        "bland_altman_lower_loa_mm3": (
            ba["lower_loa_mm3"]
        ),

        "bland_altman_upper_loa_mm3": (
            ba["upper_loa_mm3"]
        ),
    }


def bootstrap_ci(
    reference,
    prediction,
    metric_function,
    iterations=BOOTSTRAP_ITERATIONS,
    seed=SEED,
):

    reference = np.asarray(
        reference,
        dtype=float,
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    n = len(reference)

    rng = np.random.default_rng(
        seed
    )

    values = []

    for _ in range(iterations):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        ref_sample = reference[
            indices
        ]

        pred_sample = prediction[
            indices
        ]

        try:

            value = metric_function(
                ref_sample,
                pred_sample,
            )

            if np.isfinite(value):
                values.append(
                    value
                )

        except Exception:
            pass

    if len(values) < 100:
        return np.nan, np.nan

    low, high = np.percentile(
        values,
        [
            2.5,
            97.5,
        ],
    )

    return (
        float(low),
        float(high),
    )


def metric_ci_table(
    reference,
    prediction,
):

    metric_functions = {

        "mae_mm3":
            lambda r, p:
            mae(r, p),

        "rmse_mm3":
            lambda r, p:
            rmse(r, p),

        "mean_signed_bias_mm3":
            lambda r, p:
            float(
                np.mean(
                    np.asarray(p)
                    -
                    np.asarray(r)
                )
            ),

        "pearson_r":
            lambda r, p:
            safe_pearson(r, p),

        "spearman_rho":
            lambda r, p:
            safe_spearman(r, p),

        "icc_a1":
            lambda r, p:
            icc_absolute_agreement(
                r,
                p,
            ),
    }

    output = {}

    for name, function in (
        metric_functions.items()
    ):

        low, high = bootstrap_ci(
            reference,
            prediction,
            function,
        )

        output[
            f"{name}_ci95_low"
        ] = low

        output[
            f"{name}_ci95_high"
        ] = high

    return output


def assign_volume_strata(reference_values):
    """
    Because actual clinical mild/advanced labels
    are not available in the provided MAT files,
    use reference-volume tertiles.

    These must be called low/middle/high
    volume strata, NOT mild/moderate/advanced AMD.
    """

    reference_values = pd.Series(
        reference_values
    )

    try:

        labels = pd.qcut(
            reference_values,
            q=3,
            labels=[
                "low",
                "middle",
                "high",
            ],
            duplicates="drop",
        )

        return labels.astype(str)

    except Exception:

        return pd.Series(
            ["unassigned"]
            *
            len(reference_values)
        )


def main():

    all_model_data = {}
    summary_rows = []

    print(
        "=" * 90
    )

    print(
        "DRUSEN VOLUME AGREEMENT ANALYSIS"
    )

    print(
        "=" * 90
    )

    for model in MODELS:

        path = (
            INPUT_DIR
            /
            f"{model}_volume_agreement.csv"
        )

        df = pd.read_csv(
            path
        )

        all_model_data[
            model
        ] = df

        print()
        print(
            "#" * 90
        )

        print(
            model.upper()
        )

        print(
            "#" * 90
        )

        for k in K_VALUES:

            subset = df[
                np.isclose(
                    df["k_sd"],
                    k,
                )
            ].copy()

            reference = subset[
                "reference_volume_mm3"
            ].to_numpy(
                dtype=float
            )

            prediction = subset[
                "predicted_volume_mm3"
            ].to_numpy(
                dtype=float
            )

            metrics = (
                calculate_metrics(
                    reference,
                    prediction,
                )
            )

            cis = metric_ci_table(
                reference,
                prediction,
            )

            row = {
                "model": model,
                "k_sd": k,
                **metrics,
                **cis,
            }

            summary_rows.append(
                row
            )

            print()
            print(
                f"k = {k}"
            )

            print(
                f"  MAE: "
                f"{metrics['mae_mm3']:.6f} mm3"
            )

            print(
                f"  RMSE: "
                f"{metrics['rmse_mm3']:.6f} mm3"
            )

            print(
                f"  Bias: "
                f"{metrics['mean_signed_bias_mm3']:.6f} mm3"
            )

            print(
                f"  Median relative error: "
                f"{metrics['median_relative_error_percent']:.2f}%"
            )

            print(
                f"  Pearson r: "
                f"{metrics['pearson_r']:.4f}"
            )

            print(
                f"  Spearman rho: "
                f"{metrics['spearman_rho']:.4f}"
            )

            print(
                f"  ICC(A,1): "
                f"{metrics['icc_a1']:.4f}"
            )

            print(
                "  Bland-Altman: "
                f"{metrics['bland_altman_bias_mm3']:.6f} "
                f"["
                f"{metrics['bland_altman_lower_loa_mm3']:.6f}, "
                f"{metrics['bland_altman_upper_loa_mm3']:.6f}"
                f"] mm3"
            )

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUTPUT_DIR
        /
        "volume_agreement_all_thresholds.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # -----------------------------------------
    # PRIMARY 3-SD MODEL COMPARISON
    # -----------------------------------------

    print()
    print(
        "=" * 90
    )

    print(
        "PAIRED MODEL COMPARISON AT 3 SD"
    )

    print(
        "=" * 90
    )

    unetpp = all_model_data[
        "unetpp"
    ]

    deeplab = all_model_data[
        "deeplabv3plus"
    ]

    unetpp = unetpp[
        np.isclose(
            unetpp["k_sd"],
            3.0,
        )
    ].copy()

    deeplab = deeplab[
        np.isclose(
            deeplab["k_sd"],
            3.0,
        )
    ].copy()

    merged = unetpp.merge(
        deeplab,
        on="subject",
        suffixes=(
            "_unetpp",
            "_deeplab",
        ),
        validate="one_to_one",
    )

    if len(merged) != 14:
        raise RuntimeError(
            f"Expected 14 paired AMD subjects, "
            f"found {len(merged)}"
        )

    error_unetpp = (
        merged[
            "absolute_error_mm3_unetpp"
        ].to_numpy(
            dtype=float
        )
    )

    error_deeplab = (
        merged[
            "absolute_error_mm3_deeplab"
        ].to_numpy(
            dtype=float
        )
    )

    difference = (
        error_unetpp
        -
        error_deeplab
    )

    try:

        statistic, p_value = wilcoxon(
            error_unetpp,
            error_deeplab,
            alternative="two-sided",
        )

    except ValueError:

        statistic = np.nan
        p_value = np.nan

    rng = np.random.default_rng(
        SEED
    )

    boot_differences = []

    for _ in range(
        BOOTSTRAP_ITERATIONS
    ):

        indices = rng.integers(
            0,
            len(difference),
            size=len(difference),
        )

        boot_differences.append(
            np.mean(
                difference[
                    indices
                ]
            )
        )

    ci_low, ci_high = np.percentile(
        boot_differences,
        [
            2.5,
            97.5,
        ],
    )

    paired_summary = pd.DataFrame(
        [
            {
                "n_subjects": len(merged),

                "unetpp_mae_mm3":
                    float(
                        np.mean(
                            error_unetpp
                        )
                    ),

                "deeplabv3plus_mae_mm3":
                    float(
                        np.mean(
                            error_deeplab
                        )
                    ),

                "paired_mae_difference_unetpp_minus_deeplab":
                    float(
                        np.mean(
                            difference
                        )
                    ),

                "difference_ci95_low":
                    float(ci_low),

                "difference_ci95_high":
                    float(ci_high),

                "wilcoxon_statistic":
                    statistic,

                "wilcoxon_p_value":
                    p_value,
            }
        ]
    )

    print(
        paired_summary.to_string(
            index=False
        )
    )

    paired_summary.to_csv(
        OUTPUT_DIR
        /
        "paired_unetpp_vs_deeplab_3sd.csv",
        index=False,
    )

    # -----------------------------------------
    # VOLUME STRATA
    # -----------------------------------------

    print()
    print(
        "=" * 90
    )

    print(
        "REFERENCE-VOLUME STRATA AT 3 SD"
    )

    print(
        "=" * 90
    )

    strata_rows = []

    for model in MODELS:

        df = all_model_data[
            model
        ]

        subset = df[
            np.isclose(
                df["k_sd"],
                3.0,
            )
        ].copy()

        subset = subset.sort_values(
            "reference_volume_mm3"
        ).reset_index(
            drop=True
        )

        subset["volume_stratum"] = (
            assign_volume_strata(
                subset[
                    "reference_volume_mm3"
                ]
            )
        )

        subset.to_csv(
            OUTPUT_DIR
            /
            f"{model}_3sd_subject_strata.csv",
            index=False,
        )

        for stratum in [
            "low",
            "middle",
            "high",
        ]:

            group = subset[
                subset[
                    "volume_stratum"
                ]
                ==
                stratum
            ]

            if len(group) == 0:
                continue

            strata_rows.append(
                {
                    "model": model,
                    "volume_stratum": stratum,
                    "n_subjects": len(group),

                    "reference_volume_mean_mm3":
                        group[
                            "reference_volume_mm3"
                        ].mean(),

                    "reference_volume_median_mm3":
                        group[
                            "reference_volume_mm3"
                        ].median(),

                    "volume_mae_mm3":
                        group[
                            "absolute_error_mm3"
                        ].mean(),

                    "median_relative_error_percent":
                        group[
                            "relative_error_percent"
                        ].median(),

                    "mean_signed_error_mm3":
                        group[
                            "signed_error_mm3"
                        ].mean(),
                }
            )

    strata = pd.DataFrame(
        strata_rows
    )

    print(
        strata.to_string(
            index=False
        )
    )

    strata.to_csv(
        OUTPUT_DIR
        /
        "volume_strata_3sd.csv",
        index=False,
    )

    print()
    print(
        "=" * 90
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 90
    )

    print(summary_path)

    print(
        OUTPUT_DIR
        /
        "paired_unetpp_vs_deeplab_3sd.csv"
    )

    print(
        OUTPUT_DIR
        /
        "volume_strata_3sd.csv"
    )


if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    main()