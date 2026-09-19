#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
)

from scipy.stats import (
    pearsonr,
    spearmanr,
)

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
)

import segmentation_models_pytorch as smp


def pad_to_multiple(
    array,
    multiple=32,
):

    height, width = array.shape[-2:]

    pad_height = (
        -height
    ) % multiple

    pad_width = (
        -width
    ) % multiple

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


class EvaluationDataset(Dataset):

    def __init__(
        self,
        dataframe,
    ):

        self.dataframe = (
            dataframe.reset_index(
                drop=True
            )
        )

    def __len__(self):

        return len(
            self.dataframe
        )

    def __getitem__(
        self,
        index,
    ):

        row = self.dataframe.iloc[
            index
        ]

        image = np.asarray(
            Image.open(
                row["image_path"]
            ).convert("L"),
            dtype=np.float32,
        ) / 255.0

        mask = np.asarray(
            Image.open(
                row["rpedc_mask_path"]
            ).convert("L"),
            dtype=np.uint8,
        )

        mask = (
            mask > 0
        ).astype(np.uint8)

        original_height, original_width = (
            image.shape
        )

        image = np.stack(
            [image, image, image],
            axis=0,
        )

        image = pad_to_multiple(
            image,
            multiple=32,
        )

        mask = pad_to_multiple(
            mask,
            multiple=32,
        )

        return {
            "image": torch.from_numpy(
                np.ascontiguousarray(
                    image
                )
            ).float(),
            "mask": torch.from_numpy(
                np.ascontiguousarray(
                    mask
                )
            ).long(),
            "original_height": (
                original_height
            ),
            "original_width": (
                original_width
            ),
            "subject": str(
                row["subject"]
            ),
            "group": str(
                row["group"]
            ),
            "bscan_idx": int(
                row["bscan_idx"]
            ),
        }


def build_model(
    architecture,
    encoder,
):

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
        architecture
    )


def overlap_metrics(ground_truth, prediction):



    ground_truth = ground_truth.astype(bool)

    prediction = prediction.astype(bool)



    true_positive = int(

        np.logical_and(ground_truth, prediction).sum()

    )

    true_negative = int(

        np.logical_and(~ground_truth, ~prediction).sum()

    )

    false_positive = int(

        np.logical_and(~ground_truth, prediction).sum()

    )

    false_negative = int(

        np.logical_and(ground_truth, ~prediction).sum()

    )



    epsilon = 1e-8



    dice = (

        2.0 * true_positive

        / (

            2.0 * true_positive

            + false_positive

            + false_negative

            + epsilon

        )

    )



    iou = (

        true_positive

        / (

            true_positive

            + false_positive

            + false_negative

            + epsilon

        )

    )



    precision = (

        true_positive

        / (

            true_positive

            + false_positive

            + epsilon

        )

    )



    recall = (

        true_positive

        / (

            true_positive

            + false_negative

            + epsilon

        )

    )



    specificity = (

        true_negative

        / (

            true_negative

            + false_positive

            + epsilon

        )

    )



    negative_predictive_value = (

        true_negative

        / (

            true_negative

            + false_negative

            + epsilon

        )

    )



    accuracy = (

        (true_positive + true_negative)

        / (

            true_positive

            + true_negative

            + false_positive

            + false_negative

            + epsilon

        )

    )



    balanced_accuracy = (

        recall + specificity

    ) / 2.0



    false_positive_rate = (

        false_positive

        / (

            false_positive

            + true_negative

            + epsilon

        )

    )



    false_negative_rate = (

        false_negative

        / (

            false_negative

            + true_positive

            + epsilon

        )

    )



    # Float conversion prevents large-integer NumPy sqrt errors.

    mcc_product = (

        float(true_positive + false_positive)

        * float(true_positive + false_negative)

        * float(true_negative + false_positive)

        * float(true_negative + false_negative)

    )



    mcc_denominator = (

        float(np.sqrt(max(mcc_product, 0.0)))

        + epsilon

    )



    mcc = (

        (

            true_positive * true_negative

            - false_positive * false_negative

        )

        / mcc_denominator

    )



    return {

        "true_positive": true_positive,

        "true_negative": true_negative,

        "false_positive": false_positive,

        "false_negative": false_negative,

        "dice": float(dice),

        "iou": float(iou),

        "precision": float(precision),

        "recall": float(recall),

        "specificity": float(specificity),

        "negative_predictive_value": float(

            negative_predictive_value

        ),

        "accuracy": float(accuracy),

        "balanced_accuracy": float(balanced_accuracy),

        "false_positive_rate": float(false_positive_rate),

        "false_negative_rate": float(false_negative_rate),

        "mcc": float(mcc),

    }





def calculate_surface_metrics(
    ground_truth,
    prediction,
):

    ground_truth = (
        ground_truth.astype(bool)
    )

    prediction = (
        prediction.astype(bool)
    )

    if (
        not ground_truth.any()
        and
        not prediction.any()
    ):

        return 0.0, 0.0

    if (
        not ground_truth.any()
        or
        not prediction.any()
    ):

        return np.nan, np.nan

    ground_truth_surface = (
        np.logical_xor(
            ground_truth,
            binary_erosion(
                ground_truth
            ),
        )
    )

    prediction_surface = (
        np.logical_xor(
            prediction,
            binary_erosion(
                prediction
            ),
        )
    )

    prediction_to_ground_truth = (
        distance_transform_edt(
            ~ground_truth_surface
        )[prediction_surface]
    )

    ground_truth_to_prediction = (
        distance_transform_edt(
            ~prediction_surface
        )[ground_truth_surface]
    )

    all_distances = np.concatenate(
        [
            prediction_to_ground_truth,
            ground_truth_to_prediction,
        ]
    )

    hd95 = float(
        np.percentile(
            all_distances,
            95,
        )
    )

    assd = float(
        (
            prediction_to_ground_truth.mean()
            +
            ground_truth_to_prediction.mean()
        )
        / 2.0
    )

    return hd95, assd


def calculate_thickness_metrics(
    ground_truth,
    prediction,
):

    ground_truth_thickness = (
        ground_truth.astype(
            np.float32
        ).sum(axis=0)
    )

    prediction_thickness = (
        prediction.astype(
            np.float32
        ).sum(axis=0)
    )

    valid_columns = np.logical_or(
        ground_truth_thickness > 0,
        prediction_thickness > 0,
    )

    if valid_columns.sum() == 0:

        return {
            "thickness_mae": 0.0,
            "thickness_rmse": 0.0,
            "thickness_bias": 0.0,
            "thickness_pearson": np.nan,
            "thickness_spearman": np.nan,
        }

    ground_truth_values = (
        ground_truth_thickness[
            valid_columns
        ]
    )

    prediction_values = (
        prediction_thickness[
            valid_columns
        ]
    )

    errors = (
        prediction_values
        -
        ground_truth_values
    )

    if (
        np.std(
            ground_truth_values
        ) > 0
        and
        np.std(
            prediction_values
        ) > 0
    ):

        pearson_value = float(
            pearsonr(
                ground_truth_values,
                prediction_values,
            ).statistic
        )

        spearman_value = float(
            spearmanr(
                ground_truth_values,
                prediction_values,
            ).statistic
        )

    else:

        pearson_value = np.nan
        spearman_value = np.nan

    return {
        "thickness_mae": float(
            np.mean(
                np.abs(errors)
            )
        ),

        "thickness_rmse": float(
            np.sqrt(
                np.mean(
                    errors ** 2
                )
            )
        ),

        "thickness_bias": float(
            np.mean(errors)
        ),

        "thickness_pearson": (
            pearson_value
        ),

        "thickness_spearman": (
            spearman_value
        ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--splits",
        required=True,
    )

    parser.add_argument(
        "--split_name",
        default="test",
    )

    parser.add_argument(
        "--ckpt",
        required=True,
    )

    parser.add_argument(
        "--out_dir",
        required=True,
    )

    parser.add_argument(
        "--bs",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    output_directory = Path(
        args.out_dir
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(
        args.manifest
    )

    splits = pd.read_csv(
        args.splits
    )

    data = manifest.merge(
        splits,
        on=[
            "subject",
            "group",
        ],
        how="inner",
        validate="many_to_one",
    )

    data = data[
        data["split"]
        == args.split_name
    ].copy()

    if data.empty:

        raise RuntimeError(
            f"No rows found for "
            f"split={args.split_name}"
        )

    checkpoint = torch.load(
        args.ckpt,
        map_location="cpu",
    )

    architecture = checkpoint[
        "architecture"
    ]

    encoder = checkpoint[
        "encoder"
    ]

    checkpoint_threshold = float(
        checkpoint.get(
            "threshold",
            0.5,
        )
    )

    threshold = (
        args.threshold
        if args.threshold is not None
        else checkpoint_threshold
    )

    model = build_model(
        architecture=architecture,
        encoder=encoder,
    )

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(
        device
    )

    model.eval()

    dataset = EvaluationDataset(
        data
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            args.workers > 0
        ),
    )

    rows = []

    sampled_probabilities = []
    sampled_targets = []

    random_generator = (
        np.random.default_rng(42)
    )

    with torch.no_grad():

        for batch in dataloader:

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            logits = model(
                images
            )

            probabilities = (
                torch.sigmoid(logits)[
                    :, 0
                ]
                .cpu()
                .numpy()
            )

            masks = (
                batch["mask"]
                .numpy()
            )

            for index in range(
                len(probabilities)
            ):

                original_height = int(
                    batch[
                        "original_height"
                    ][index]
                )

                original_width = int(
                    batch[
                        "original_width"
                    ][index]
                )

                # Crop away padding before calculating metrics.
                probability = probabilities[
                    index
                ][
                    :original_height,
                    :original_width,
                ]

                ground_truth = masks[
                    index
                ][
                    :original_height,
                    :original_width,
                ].astype(np.uint8)

                prediction = (
                    probability
                    >= threshold
                ).astype(np.uint8)

                metrics = overlap_metrics(
                    ground_truth,
                    prediction,
                )

                hd95, assd = (
                    calculate_surface_metrics(
                        ground_truth,
                        prediction,
                    )
                )

                metrics[
                    "hd95_pixels"
                ] = hd95

                metrics[
                    "assd_pixels"
                ] = assd

                metrics.update(
                    calculate_thickness_metrics(
                        ground_truth,
                        prediction,
                    )
                )

                ground_truth_area = int(
                    ground_truth.sum()
                )

                prediction_area = int(
                    prediction.sum()
                )

                metrics.update(
                    {
                        "subject": batch[
                            "subject"
                        ][index],

                        "group": batch[
                            "group"
                        ][index],

                        "bscan_idx": int(
                            batch[
                                "bscan_idx"
                            ][index]
                        ),

                        "ground_truth_area_pixels": (
                            ground_truth_area
                        ),

                        "prediction_area_pixels": (
                            prediction_area
                        ),

                        "area_absolute_error": abs(
                            prediction_area
                            -
                            ground_truth_area
                        ),

                        "area_relative_error": (
                            abs(
                                prediction_area
                                -
                                ground_truth_area
                            )
                            /
                            max(
                                ground_truth_area,
                                1,
                            )
                        ),
                    }
                )

                rows.append(
                    metrics
                )

                flat_probability = (
                    probability.ravel()
                )

                flat_target = (
                    ground_truth.ravel()
                )

                sample_size = min(
                    5000,
                    len(flat_target),
                )

                sample_indices = (
                    random_generator.choice(
                        len(flat_target),
                        size=sample_size,
                        replace=False,
                    )
                )

                sampled_probabilities.append(
                    flat_probability[
                        sample_indices
                    ]
                )

                sampled_targets.append(
                    flat_target[
                        sample_indices
                    ]
                )

    slice_metrics = pd.DataFrame(
        rows
    )

    slice_metrics.to_csv(
        output_directory
        / "slice_metrics.csv",
        index=False,
    )

    metric_columns = [
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "negative_predictive_value",
        "accuracy",
        "balanced_accuracy",
        "false_positive_rate",
        "false_negative_rate",
        "mcc",
        "hd95_pixels",
        "assd_pixels",
        "thickness_mae",
        "thickness_rmse",
        "thickness_bias",
        "thickness_pearson",
        "thickness_spearman",
        "area_absolute_error",
        "area_relative_error",
    ]

    subject_metrics = (
        slice_metrics
        .groupby(
            [
                "subject",
                "group",
            ]
        )[
            metric_columns
        ]
        .mean()
        .reset_index()
    )

    subject_metrics.to_csv(
        output_directory
        / "subject_metrics.csv",
        index=False,
    )

    group_metrics = (
        subject_metrics
        .groupby(
            "group"
        )[
            metric_columns
        ]
        .agg(
            [
                "mean",
                "std",
                "median",
            ]
        )
    )

    group_metrics.to_csv(
        output_directory
        / "group_metrics.csv"
    )

    all_probabilities = np.concatenate(
        sampled_probabilities
    )

    all_targets = np.concatenate(
        sampled_targets
    )

    probability_metrics = {}

    if np.unique(
        all_targets
    ).size == 2:

        probability_metrics[
            "pixel_auroc"
        ] = float(
            roc_auc_score(
                all_targets,
                all_probabilities,
            )
        )

        probability_metrics[
            "pixel_auprc"
        ] = float(
            average_precision_score(
                all_targets,
                all_probabilities,
            )
        )

    summary = {
        "architecture": architecture,
        "encoder": encoder,
        "split": args.split_name,
        "threshold": threshold,
        "number_of_slices": int(
            len(slice_metrics)
        ),
        "number_of_subjects": int(
            subject_metrics[
                "subject"
            ].nunique()
        ),
    }

    summary.update(
        probability_metrics
    )

    for metric in metric_columns:

        summary[
            f"slice_mean_{metric}"
        ] = float(
            slice_metrics[
                metric
            ].mean()
        )

        summary[
            f"subject_mean_{metric}"
        ] = float(
            subject_metrics[
                metric
            ].mean()
        )

        summary[
            f"subject_std_{metric}"
        ] = float(
            subject_metrics[
                metric
            ].std()
        )

    with open(
        output_directory
        / "summary.json",
        "w",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
        )

    pd.DataFrame(
        [summary]
    ).to_csv(
        output_directory
        / "summary.csv",
        index=False,
    )

    print(
        f"Architecture: {architecture}",
        flush=True,
    )

    print(
        "Subject-level mean Dice:",
        subject_metrics[
            "dice"
        ].mean(),
        flush=True,
    )

    print(
        "Subject-level mean IoU:",
        subject_metrics[
            "iou"
        ].mean(),
        flush=True,
    )

    print(
        "Subject-level mean HD95:",
        subject_metrics[
            "hd95_pixels"
        ].mean(),
        flush=True,
    )

    print(
        "Subject-level thickness MAE:",
        subject_metrics[
            "thickness_mae"
        ].mean(),
        flush=True,
    )

    print(
        f"Saved results to: "
        f"{output_directory}",
        flush=True,
    )


if __name__ == "__main__":
    main()