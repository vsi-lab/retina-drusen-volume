#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_DIR = Path(
    "reports/drusen_volume/predicted_formula"
)

OUTPUT_DIR = Path(
    "reports/drusen_volume/agreement_analysis/plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


for model in [
    "unetpp",
    "deeplabv3plus",
]:

    df = pd.read_csv(
        INPUT_DIR
        /
        f"{model}_volume_agreement.csv"
    )

    df = df[
        np.isclose(
            df["k_sd"],
            3.0,
        )
    ].copy()

    reference = df[
        "reference_volume_mm3"
    ].to_numpy()

    prediction = df[
        "predicted_volume_mm3"
    ].to_numpy()

    averages = (
        reference
        +
        prediction
    ) / 2

    differences = (
        prediction
        -
        reference
    )

    bias = np.mean(
        differences
    )

    sd = np.std(
        differences,
        ddof=1,
    )

    lower = (
        bias
        -
        1.96 * sd
    )

    upper = (
        bias
        +
        1.96 * sd
    )

    # Bland-Altman
    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        averages,
        differences,
    )

    plt.axhline(
        bias,
        linestyle="-",
    )

    plt.axhline(
        lower,
        linestyle="--",
    )

    plt.axhline(
        upper,
        linestyle="--",
    )

    plt.xlabel(
        "Mean reference and predicted volume (mm³)"
    )

    plt.ylabel(
        "Predicted − reference volume (mm³)"
    )

    plt.title(
        f"{model}: Bland–Altman, 3 SD"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        /
        f"{model}_bland_altman_3sd.png",
        dpi=200,
    )

    plt.close()

    # Agreement scatter
    plt.figure(
        figsize=(7, 7)
    )

    plt.scatter(
        reference,
        prediction,
    )

    maximum = max(
        reference.max(),
        prediction.max(),
    )

    plt.plot(
        [0, maximum],
        [0, maximum],
        linestyle="--",
    )

    plt.xlabel(
        "Reference abnormal RPEDC volume (mm³)"
    )

    plt.ylabel(
        "Predicted abnormal RPEDC volume (mm³)"
    )

    plt.title(
        f"{model}: Reference vs predicted volume"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        /
        f"{model}_volume_agreement_3sd.png",
        dpi=200,
    )

    plt.close()

    print(
        model,
        "plots saved."
    )