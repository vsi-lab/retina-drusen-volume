#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio


CSV = Path("reports/normal_reference/foveal_centers.csv")

CONTROL_ROOT = Path(
    "data/raw/duke_amd/extracted/control"
)

OUT_DIR = Path(
    "reports/normal_reference/fovea_qc"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def find_mat_file(root, filename):
    matches = list(
        root.rglob(filename)
    )

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find {filename} under {root}"
        )

    if len(matches) > 1:
        print(
            f"WARNING: multiple matches for {filename}; "
            f"using {matches[0]}"
        )

    return matches[0]


df = pd.read_csv(CSV)

controls = df[
    (df["group"] == "Control")
    &
    (df["status"] == "ok")
].copy()

controls = controls.sort_values(
    "fovea_bscan"
)

indices = np.linspace(
    0,
    len(controls) - 1,
    12,
).astype(int)

selected = controls.iloc[
    indices
]

print("Selected QC subjects:", len(selected))
print()


for _, row in selected.iterrows():

    try:

        path = find_mat_file(
            CONTROL_ROOT,
            row["filename"],
        )

        data = sio.loadmat(path)

        images = data["images"]
        layers = data["layerMaps"]

        bscan = int(
            row["fovea_bscan"]
        )

        x = int(
            row["fovea_x"]
        )

        if bscan < 0 or bscan >= images.shape[2]:
            print(
                "SKIP invalid B-scan:",
                row["subject"],
                bscan,
            )
            continue

        image = images[
            :,
            :,
            bscan,
        ]

        plt.figure(
            figsize=(14, 6)
        )

        plt.imshow(
            image,
            cmap="gray",
        )

        for channel in range(
            layers.shape[2]
        ):

            y = layers[
                bscan,
                :,
                channel,
            ]

            valid = np.isfinite(y)

            plt.plot(
                np.arange(len(y))[valid],
                y[valid],
                linewidth=1.2,
                label=f"Boundary {channel}",
            )

        plt.axvline(
            x,
            linestyle="--",
            linewidth=2,
            label="Estimated foveal x",
        )

        plt.title(
            f"{row['subject']} | "
            f"B-scan={bscan}, "
            f"x={x}, "
            f"age={row['age']}"
        )

        plt.xlim(
            0,
            image.shape[1] - 1,
        )

        plt.ylim(
            image.shape[0] - 1,
            0,
        )

        plt.legend(
            loc="best"
        )

        plt.tight_layout()

        out = (
            OUT_DIR
            /
            f"{row['subject']}_fovea.png"
        )

        plt.savefig(
            out,
            dpi=150,
        )

        plt.close()

        print(
            "Saved:",
            out,
        )

    except Exception as exc:

        print(
            "ERROR:",
            row["subject"],
            type(exc).__name__,
            exc,
        )


print()
print("QC complete.")
print("Output:", OUT_DIR)