#!/usr/bin/env python3

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


def pad_to_multiple(array, multiple=32):
    """
    Pad the bottom and right sides so height and width are divisible by 32.
    Duke images are 512x1000 and become 512x1024.
    """
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
            f"Unsupported array shape: {array.shape}"
        )

    return np.pad(
        array,
        padding,
        mode="constant",
        constant_values=0,
    )


class RPEDCDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image = np.asarray(
            Image.open(row["image_path"]).convert("L"),
            dtype=np.uint8,
        )

        mask = np.asarray(
            Image.open(row["rpedc_mask_path"]).convert("L"),
            dtype=np.uint8,
        )

        mask = (mask > 0).astype(np.float32)

        if image.shape != mask.shape:
            raise RuntimeError(
                f"Shape mismatch for {row['image_path']}: "
                f"image={image.shape}, mask={mask.shape}"
            )

        if self.transform is not None:

            augmented = self.transform(
                image=image,
                mask=mask,
            )

            image = augmented["image"]
            mask = augmented["mask"]

        image = image.astype(np.float32) / 255.0

        # Convert grayscale into three channels for ImageNet encoders.
        image = np.stack(
            [image, image, image],
            axis=0,
        )

        mask = mask.astype(np.float32)[None, :, :]

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
                np.ascontiguousarray(image)
            ),
            "mask": torch.from_numpy(
                np.ascontiguousarray(mask)
            ),
            "subject": str(row["subject"]),
            "group": str(row["group"]),
            "bscan_idx": int(row["bscan_idx"]),
        }


def build_transforms(training):

    if not training:
        return A.Compose([])

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),

            A.ShiftScaleRotate(
                shift_limit=0.02,
                scale_limit=0.05,
                rotate_limit=3,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.12,
                contrast_limit=0.12,
                p=0.4,
            ),
        ]
    )


def build_model(
    architecture,
    encoder,
    encoder_weights,
):

    common_arguments = {
        "encoder_name": encoder,
        "encoder_weights": encoder_weights,
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
        f"Unsupported architecture: {architecture}"
    )


class DiceBCELoss(nn.Module):

    def __init__(
        self,
        bce_weight=0.4,
        dice_weight=0.6,
        pos_weight=3.0,
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.register_buffer(
            "positive_weight",
            torch.tensor(
                [pos_weight],
                dtype=torch.float32,
            ),
        )

    def forward(self, logits, targets):

        bce_loss = (
            nn.functional.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=self.positive_weight,
            )
        )

        probabilities = torch.sigmoid(logits)

        dimensions = (
            0,
            2,
            3,
        )

        intersection = torch.sum(
            probabilities * targets,
            dim=dimensions,
        )

        denominator = torch.sum(
            probabilities + targets,
            dim=dimensions,
        )

        soft_dice = (
            (2.0 * intersection + 1e-6)
            /
            (denominator + 1e-6)
        )

        dice_loss = 1.0 - soft_dice.mean()

        total_loss = (
            self.bce_weight * bce_loss
            +
            self.dice_weight * dice_loss
        )

        return (
            total_loss,
            bce_loss.detach(),
            dice_loss.detach(),
        )


def calculate_metrics(
    true_positive,
    false_positive,
    false_negative,
    true_negative,
):

    epsilon = 1e-8

    dice = (
        2 * true_positive
        /
        (
            2 * true_positive
            + false_positive
            + false_negative
            + epsilon
        )
    )

    iou = (
        true_positive
        /
        (
            true_positive
            + false_positive
            + false_negative
            + epsilon
        )
    )

    precision = (
        true_positive
        /
        (
            true_positive
            + false_positive
            + epsilon
        )
    )

    recall = (
        true_positive
        /
        (
            true_positive
            + false_negative
            + epsilon
        )
    )

    specificity = (
        true_negative
        /
        (
            true_negative
            + false_positive
            + epsilon
        )
    )

    accuracy = (
        true_positive + true_negative
        /
        (
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

    mcc_denominator = math.sqrt(
        max(
            (
                true_positive + false_positive
            )
            *
            (
                true_positive + false_negative
            )
            *
            (
                true_negative + false_positive
            )
            *
            (
                true_negative + false_negative
            ),
            0.0,
        )
    ) + epsilon

    mcc = (
        (
            true_positive * true_negative
        )
        -
        (
            false_positive * false_negative
        )
    ) / mcc_denominator

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "mcc": float(mcc),
    }


def run_epoch(
    model,
    dataloader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    threshold=0.5,
    use_amp=False,
):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_bce_loss = 0.0
    total_dice_loss = 0.0
    total_samples = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    for batch in dataloader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        amp_context = (
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            )
            if use_amp
            else nullcontext()
        )

        with torch.set_grad_enabled(training):

            with amp_context:

                logits = model(images)

                loss, bce_loss, dice_loss = criterion(
                    logits,
                    masks,
                )

            if training:

                if scaler is not None:

                    scaler.scale(loss).backward()

                    scaler.unscale_(
                        optimizer
                    )

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=5.0,
                    )

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                else:

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=5.0,
                    )

                    optimizer.step()

        predictions = (
            torch.sigmoid(logits)
            >= threshold
        )

        targets = masks >= 0.5

        total_tp += torch.logical_and(
            predictions,
            targets,
        ).sum().item()

        total_fp += torch.logical_and(
            predictions,
            ~targets,
        ).sum().item()

        total_fn += torch.logical_and(
            ~predictions,
            targets,
        ).sum().item()

        total_tn += torch.logical_and(
            ~predictions,
            ~targets,
        ).sum().item()

        batch_size = images.shape[0]

        total_loss += (
            loss.item() * batch_size
        )

        total_bce_loss += (
            bce_loss.item() * batch_size
        )

        total_dice_loss += (
            dice_loss.item() * batch_size
        )

        total_samples += batch_size

    metrics = calculate_metrics(
        total_tp,
        total_fp,
        total_fn,
        total_tn,
    )

    total_samples = max(
        total_samples,
        1,
    )

    metrics.update(
        {
            "loss": (
                total_loss / total_samples
            ),
            "bce_loss": (
                total_bce_loss
                / total_samples
            ),
            "dice_loss": (
                total_dice_loss
                / total_samples
            ),
        }
    )

    return metrics


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
        "--out_dir",
        required=True,
    )

    parser.add_argument(
        "--arch",
        required=True,
        choices=[
            "unet",
            "unetpp",
            "fpn",
            "deeplabv3plus",
        ],
    )

    parser.add_argument(
        "--encoder",
        default="resnet34",
    )

    parser.add_argument(
        "--encoder_weights",
        default="imagenet",
        choices=[
            "imagenet",
            "none",
        ],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
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
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--bce_weight",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--dice_weight",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--pos_weight",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--amp",
        action="store_true",
    )

    args = parser.parse_args()

    seed_everything(
        args.seed
    )

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

    if len(data) != len(manifest):

        raise RuntimeError(
            "Some manifest rows were lost "
            f"during the split merge: "
            f"{len(manifest)} -> {len(data)}"
        )

    train_dataframe = data[
        data["split"] == "train"
    ].copy()

    validation_dataframe = data[
        data["split"] == "val"
    ].copy()

    train_subjects = set(
        train_dataframe[
            "subject"
        ].astype(str)
    )

    validation_subjects = set(
        validation_dataframe[
            "subject"
        ].astype(str)
    )

    subject_overlap = (
        train_subjects
        &
        validation_subjects
    )

    if subject_overlap:

        raise RuntimeError(
            "Subject leakage detected: "
            f"{list(subject_overlap)[:5]}"
        )

    train_dataset = RPEDCDataset(
        train_dataframe,
        transform=build_transforms(
            training=True
        ),
    )

    validation_dataset = RPEDCDataset(
        validation_dataframe,
        transform=build_transforms(
            training=False
        ),
    )

    pin_memory = (
        torch.cuda.is_available()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=(
            args.workers > 0
        ),
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=(
            args.workers > 0
        ),
        drop_last=False,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    encoder_weights = (
        None
        if args.encoder_weights == "none"
        else args.encoder_weights
    )

    model = build_model(
        architecture=args.arch,
        encoder=args.encoder,
        encoder_weights=encoder_weights,
    )

    model = model.to(
        device
    )

    criterion = DiceBCELoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        pos_weight=args.pos_weight,
    )

    criterion = criterion.to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
        )
    )

    use_amp = (
        args.amp
        and device.type == "cuda"
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )
        if use_amp
        else None
    )

    configuration = vars(
        args
    ).copy()

    configuration.update(
        {
            "device": str(device),
            "training_slices": len(
                train_dataframe
            ),
            "validation_slices": len(
                validation_dataframe
            ),
            "training_subjects": int(
                train_dataframe[
                    "subject"
                ].nunique()
            ),
            "validation_subjects": int(
                validation_dataframe[
                    "subject"
                ].nunique()
            ),
        }
    )

    with open(
        output_directory
        / "config.json",
        "w",
    ) as handle:

        json.dump(
            configuration,
            handle,
            indent=2,
        )

    print(
        f"Device: {device}",
        flush=True,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
            flush=True,
        )

    print(
        f"Architecture: {args.arch}",
        flush=True,
    )

    print(
        f"Training slices: "
        f"{len(train_dataframe)}",
        flush=True,
    )

    print(
        f"Validation slices: "
        f"{len(validation_dataframe)}",
        flush=True,
    )

    history = []

    best_validation_dice = -1.0
    epochs_without_improvement = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        epoch_start = time.time()

        training_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            threshold=args.threshold,
            use_amp=use_amp,
        )

        with torch.no_grad():

            validation_metrics = run_epoch(
                model=model,
                dataloader=validation_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                scaler=None,
                threshold=args.threshold,
                use_amp=use_amp,
            )

        scheduler.step(
            validation_metrics["dice"]
        )

        elapsed_seconds = (
            time.time()
            -
            epoch_start
        )

        learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        history_row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "seconds": elapsed_seconds,
        }

        history_row.update(
            {
                f"train_{key}": value
                for key, value
                in training_metrics.items()
            }
        )

        history_row.update(
            {
                f"val_{key}": value
                for key, value
                in validation_metrics.items()
            }
        )

        history.append(
            history_row
        )

        pd.DataFrame(
            history
        ).to_csv(
            output_directory
            / "training_history.csv",
            index=False,
        )

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": (
                optimizer.state_dict()
            ),
            "epoch": epoch,
            "architecture": args.arch,
            "encoder": args.encoder,
            "encoder_weights": (
                encoder_weights
            ),
            "threshold": args.threshold,
            "validation_metrics": (
                validation_metrics
            ),
            "configuration": (
                configuration
            ),
        }

        torch.save(
            checkpoint,
            output_directory
            / "last.pt",
        )

        improved = (
            validation_metrics["dice"]
            >
            best_validation_dice
        )

        if improved:

            best_validation_dice = (
                validation_metrics["dice"]
            )

            epochs_without_improvement = 0

            torch.save(
                checkpoint,
                output_directory
                / "best.pt",
            )

        else:

            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss="
            f"{training_metrics['loss']:.4f} "
            f"train_dice="
            f"{training_metrics['dice']:.4f} | "
            f"val_loss="
            f"{validation_metrics['loss']:.4f} "
            f"val_dice="
            f"{validation_metrics['dice']:.4f} "
            f"val_iou="
            f"{validation_metrics['iou']:.4f} "
            f"val_precision="
            f"{validation_metrics['precision']:.4f} "
            f"val_recall="
            f"{validation_metrics['recall']:.4f} "
            f"val_specificity="
            f"{validation_metrics['specificity']:.4f} "
            f"val_mcc="
            f"{validation_metrics['mcc']:.4f} | "
            f"lr={learning_rate:.2e} "
            f"time={elapsed_seconds:.1f}s "
            f"{'[BEST]' if improved else ''}",
            flush=True,
        )

        if (
            epochs_without_improvement
            >= args.patience
        ):

            print(
                "Early stopping at "
                f"epoch {epoch}. "
                "Best validation Dice: "
                f"{best_validation_dice:.4f}",
                flush=True,
            )

            break

    print(
        "Training completed. "
        "Best validation Dice: "
        f"{best_validation_dice:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()