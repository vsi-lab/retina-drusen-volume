#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import segmentation_models_pytorch as smp


AXIAL_SPACING_UM = 3.23
LATERAL_SPACING_UM = 6.54
BSCAN_SPACING_UM = 67.0

K_VALUES = [2.0, 2.5, 3.0, 3.5, 4.0]
MIN_VALID_CONTROLS_PER_PIXEL = 3


def pad_to_multiple(array, multiple=32):

    height, width = array.shape[-2:]

    pad_height = (-height) % multiple
    pad_width = (-width) % multiple

    if array.ndim == 2:
        padding = (
            (0, pad_height),
            (0, pad_width),
        )

    elif array.ndim == 3:
        padding = (
            (0, 0),
            (0, pad_height),
            (0, pad_width),
        )

    else:
        raise ValueError(
            f"Unsupported shape: {array.shape}"
        )

    return np.pad(
        array,
        padding,
        mode="constant",
        constant_values=0,
    )


def build_model(architecture, encoder):

    common_arguments = {
        "encoder_name": encoder,
        "encoder_weights": None,
        "in_channels": 3,
        "classes": 1,
        "activation": None,
    }

    if architecture == "unet":
        return smp.Unet(
            **common_arguments
        )

    if architecture == "unetpp":
        return smp.UnetPlusPlus(
            **common_arguments
        )

    if architecture == "fpn":
        return smp.FPN(
            **common_arguments
        )

    if architecture == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            **common_arguments
        )

    raise ValueError(
        f"Unknown architecture: {architecture}"
    )


def load_checkpoint(checkpoint_path, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    architecture = checkpoint["architecture"]
    encoder = checkpoint["encoder"]

    threshold = float(
        checkpoint.get(
            "threshold",
            0.5,
        )
    )

    model = build_model(
        architecture,
        encoder,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.to(device)
    model.eval()

    return (
        model,
        architecture,
        encoder,
        threshold,
    )


def load_thickness_map(path):

    array = np.load(path)

    if array.ndim != 2:
        raise ValueError(
            f"Expected 2-D thickness map: "
            f"{path}, got {array.shape}"
        )

    return array.astype(
        np.float32
    )


def choose_age_matched_controls(
    controls,
    amd_age,
    min_controls=5,
):

    for window in [5, 7, 10]:

        matched = controls[
            (controls["age"] >= amd_age - window)
            &
            (controls["age"] <= amd_age + window)
        ].copy()

        if len(matched) >= min_controls:
            return matched, window

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


def build_reference_maps(
    matched_controls,
):

    arrays = []

    for _, row in matched_controls.iterrows():

        array = load_thickness_map(
            row["thickness_map"]
        )

        if array.shape == (100, 1000):
            arrays.append(array)

    if len(arrays) < 3:
        raise RuntimeError(
            "Not enough common-grid control maps."
        )

    stack = np.stack(
        arrays,
        axis=0,
    )

    count_map = np.sum(
        np.isfinite(stack),
        axis=0,
    )

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
        len(arrays),
    )


def predict_subject_thickness(
    subject_rows,
    model,
    threshold,
    device,
    batch_size=4,
):

    """
    Returns a 100 x 1000 predicted RPEDC
    thickness map in microns.

    Only B-scans available in the manifest
    are populated.
    """

    predicted_map = np.full(
        (100, 1000),
        np.nan,
        dtype=np.float32,
    )

    subject_rows = (
        subject_rows
        .sort_values("bscan_idx")
        .reset_index(drop=True)
    )

    for start in range(
        0,
        len(subject_rows),
        batch_size,
    ):

        batch_rows = subject_rows.iloc[
            start:start + batch_size
        ]

        tensors = []
        metadata = []

        for _, row in batch_rows.iterrows():

            image = np.asarray(
                Image.open(
                    row["image_path"]
                ).convert("L"),
                dtype=np.float32,
            ) / 255.0

            original_height = image.shape[0]
            original_width = image.shape[1]

            image = np.stack(
                [image, image, image],
                axis=0,
            )

            image = pad_to_multiple(
                image,
                multiple=32,
            )

            tensors.append(
                torch.from_numpy(
                    np.ascontiguousarray(
                        image
                    )
                ).float()
            )

            metadata.append(
                {
                    "bscan_idx": int(
                        row["bscan_idx"]
                    ),
                    "height": original_height,
                    "width": original_width,
                }
            )

        batch_tensor = torch.stack(
            tensors,
            dim=0,
        ).to(device)

        with torch.no_grad():

            logits = model(
                batch_tensor
            )

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities
                >= threshold
            ).to(
                torch.uint8
            )

        predictions = (
            predictions
            .cpu()
            .numpy()
        )

        for prediction, meta in zip(
            predictions,
            metadata,
        ):

            mask = prediction[
                0,
                :meta["height"],
                :meta["width"],
            ]

            # RPEDC mask thickness for each A-scan.
            thickness_px = mask.sum(
                axis=0,
                dtype=np.float32,
            )

            thickness_um = (
                thickness_px
                *
                AXIAL_SPACING_UM
            )

            bscan_idx = meta[
                "bscan_idx"
            ]

            predicted_map[
                bscan_idx,
                :meta["width"],
            ] = thickness_um

    return predicted_map


def calculate_volume(
    thickness_map,
    ground_truth_map,
    mean_map,
    std_map,
    k,
):
    """
    For fair reference comparison, evaluate
    only spatial locations for which:

    1. supplied AMD RPEDC thickness is valid,
    2. predicted thickness exists,
    3. normal control reference is valid.

    This prevents unlabeled regions from being
    treated as prediction failures.
    """

    threshold_map = (
        mean_map
        +
        k * std_map
    )

    valid = (
        np.isfinite(thickness_map)
        &
        np.isfinite(ground_truth_map)
        &
        np.isfinite(threshold_map)
    )

    difference = np.full(
        thickness_map.shape,
        np.nan,
        dtype=np.float32,
    )

    difference[valid] = (
        thickness_map[valid]
        -
        threshold_map[valid]
    )

    excess_um = np.maximum(
        difference,
        0.0,
    )

    abnormal = (
        valid
        &
        (thickness_map > threshold_map)
    )

    volume_um3 = (
        float(
            np.nansum(excess_um)
        )
        *
        LATERAL_SPACING_UM
        *
        BSCAN_SPACING_UM
    )

    volume_mm3 = (
        volume_um3
        /
        1_000_000_000.0
    )

    return {
        "volume_mm3": volume_mm3,
        "valid_locations": int(
            valid.sum()
        ),
        "abnormal_locations": int(
            abnormal.sum()
        ),
        "abnormal_fraction": (
            float(
                abnormal.sum()
                /
                valid.sum()
            )
            if valid.sum() > 0
            else np.nan
        ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default=(
            "data/manifests/"
            "duke_rpedc_manifest.csv"
        ),
    )

    parser.add_argument(
        "--splits",
        default=(
            "data/splits/"
            "duke_subject_split.csv"
        ),
    )

    parser.add_argument(
        "--thickness_manifest",
        default=(
            "reports/normal_reference/"
            "rpedc_thickness_manifest.csv"
        ),
    )

    parser.add_argument(
        "--reference_csv",
        default=(
            "reports/drusen_volume/"
            "reference_formula/"
            "drusen_volume_threshold_sensitivity.csv"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--model_name",
        required=True,
    )

    parser.add_argument(
        "--out_dir",
        default=(
            "reports/drusen_volume/"
            "predicted_formula"
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("PREDICTED RPEDC → DRUSEN VOLUME")
    print("=" * 80)

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    (
        model,
        architecture,
        encoder,
        threshold,
    ) = load_checkpoint(
        args.checkpoint,
        device,
    )

    print(
        "Architecture:",
        architecture,
    )

    print(
        "Encoder:",
        encoder,
    )

    print(
        "Threshold:",
        threshold,
    )

    manifest = pd.read_csv(
        args.manifest
    )

    splits = pd.read_csv(
        args.splits
    )

    thickness_manifest = pd.read_csv(
        args.thickness_manifest
    )

    reference_results = pd.read_csv(
        args.reference_csv
    )

    test_amd = splits[
        (splits["split"] == "test")
        &
        (splits["group"] == "AMD")
    ].copy()

    print(
        "Locked AMD test subjects:",
        len(test_amd),
    )

    if len(test_amd) != 14:
        raise RuntimeError(
            f"Expected 14 AMD test subjects, "
            f"found {len(test_amd)}"
        )

    controls = thickness_manifest[
        (thickness_manifest["group"] == "Control")
        &
        (thickness_manifest["status"] == "ok")
        &
        (thickness_manifest["bscans"] == 100)
        &
        (thickness_manifest["ascans"] == 1000)
    ].copy()

    amd_reference_manifest = thickness_manifest[
        (thickness_manifest["group"] == "AMD")
        &
        (thickness_manifest["status"] == "ok")
    ].copy()

    output_root = Path(
        args.out_dir
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for subject_index, test_row in enumerate(
        test_amd.itertuples(
            index=False
        ),
        start=1,
    ):

        subject = test_row.subject

        print()
        print(
            f"[{subject_index}/{len(test_amd)}] "
            f"{subject}"
        )

        subject_manifest = manifest[
            manifest["subject"]
            ==
            subject
        ].copy()

        if subject_manifest.empty:
            raise RuntimeError(
                f"No manifest slices for {subject}"
            )

        amd_meta = amd_reference_manifest[
            amd_reference_manifest["subject"]
            ==
            subject
        ]

        if len(amd_meta) != 1:
            raise RuntimeError(
                f"Thickness-map lookup failed "
                f"for {subject}"
            )

        amd_meta = amd_meta.iloc[0]

        amd_age = float(
            amd_meta["age"]
        )

        ground_truth_map = (
            load_thickness_map(
                amd_meta[
                    "thickness_map"
                ]
            )
        )

        predicted_map = (
            predict_subject_thickness(
                subject_manifest,
                model,
                threshold,
                device,
                batch_size=args.batch_size,
            )
        )

        matched_controls, age_window = (
            choose_age_matched_controls(
                controls,
                amd_age,
                min_controls=5,
            )
        )

        (
            mean_map,
            std_map,
            number_controls,
        ) = build_reference_maps(
            matched_controls
        )

        for k in K_VALUES:

            prediction_result = (
                calculate_volume(
                    predicted_map,
                    ground_truth_map,
                    mean_map,
                    std_map,
                    k,
                )
            )

            reference_row = (
                reference_results[
                    (
                        reference_results[
                            "subject"
                        ]
                        ==
                        subject
                    )
                    &
                    (
                        np.isclose(
                            reference_results[
                                "k_sd"
                            ],
                            k,
                        )
                    )
                ]
            )

            if len(reference_row) != 1:
                raise RuntimeError(
                    f"Reference volume not found: "
                    f"{subject}, k={k}"
                )

            reference_volume = float(
                reference_row.iloc[0][
                    "volume_mm3"
                ]
            )

            predicted_volume = (
                prediction_result[
                    "volume_mm3"
                ]
            )

            signed_error = (
                predicted_volume
                -
                reference_volume
            )

            absolute_error = abs(
                signed_error
            )

            relative_error_percent = (
                100.0
                *
                absolute_error
                /
                reference_volume
                if reference_volume > 0
                else np.nan
            )

            rows.append(
                {
                    "subject": subject,
                    "group": "AMD",
                    "age": amd_age,
                    "model": args.model_name,
                    "architecture": architecture,
                    "encoder": encoder,
                    "threshold": threshold,
                    "k_sd": k,
                    "age_window_years": age_window,
                    "number_controls": number_controls,
                    "reference_volume_mm3": (
                        reference_volume
                    ),
                    "predicted_volume_mm3": (
                        predicted_volume
                    ),
                    "signed_error_mm3": (
                        signed_error
                    ),
                    "absolute_error_mm3": (
                        absolute_error
                    ),
                    "relative_error_percent": (
                        relative_error_percent
                    ),
                    "valid_locations": (
                        prediction_result[
                            "valid_locations"
                        ]
                    ),
                    "predicted_abnormal_locations": (
                        prediction_result[
                            "abnormal_locations"
                        ]
                    ),
                    "predicted_abnormal_fraction": (
                        prediction_result[
                            "abnormal_fraction"
                        ]
                    ),
                }
            )

        del predicted_map

        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = pd.DataFrame(
        rows
    )

    output_csv = (
        output_root
        /
        f"{args.model_name}_volume_agreement.csv"
    )

    results.to_csv(
        output_csv,
        index=False,
    )

    print()
    print("=" * 80)
    print("PRIMARY 3-SD SUMMARY")
    print("=" * 80)

    primary = results[
        np.isclose(
            results["k_sd"],
            3.0,
        )
    ]

    print(
        primary[
            [
                "reference_volume_mm3",
                "predicted_volume_mm3",
                "absolute_error_mm3",
                "relative_error_percent",
            ]
        ].describe()
    )

    print()
    print("Saved:", output_csv)


if __name__ == "__main__":
    main()