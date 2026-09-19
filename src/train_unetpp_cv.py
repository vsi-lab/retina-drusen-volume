cat > src/train_unetpp_cv.py <<'PY'
#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# Dataset
# ============================================================

class RPEDCDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image = np.asarray(
            Image.open(
                row["image_path"]
            ).convert("L"),
            dtype=np.float32,
        ) / 255.0

        mask = np.asarray(
            Image.open(
                row["rpedc_mask_path"]
            ).convert("L")
        )

        mask = (
            mask > 0
        ).astype(np.float32)

        height, width = image.shape

        # Replicate grayscale into RGB,
        # matching the previous validated pipeline.
        image = np.stack(
            [image, image, image],
            axis=0,
        )

        mask = mask[None, :, :]

        # ----------------------------------------------------
        # Pad only. NEVER resize.
        # 512 x 1000 -> 512 x 1024
        # ----------------------------------------------------

        pad_h = (
            32 - height % 32
        ) % 32

        pad_w = (
            32 - width % 32
        ) % 32

        if pad_h > 0 or pad_w > 0:

            image = np.pad(
                image,
                (
                    (0, 0),
                    (0, pad_h),
                    (0, pad_w),
                ),
                mode="constant",
                constant_values=0,
            )

            mask = np.pad(
                mask,
                (
                    (0, 0),
                    (0, pad_h),
                    (0, pad_w),
                ),
                mode="constant",
                constant_values=0,
            )

        return {
            "image": torch.from_numpy(
                image
            ).float(),

            "mask": torch.from_numpy(
                mask
            ).float(),

            "height": height,
            "width": width,
        }


# ============================================================
# Loss
# ============================================================

class BCEDiceLoss(nn.Module):

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
            "pos_weight",
            torch.tensor(
                [pos_weight],
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        logits,
        targets,
    ):

        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
        )

        probabilities = torch.sigmoid(
            logits
        )

        smooth = 1e-6

        intersection = (
            probabilities * targets
        ).sum(
            dim=(1, 2, 3)
        )

        denominator = (
            probabilities.sum(
                dim=(1, 2, 3)
            )
            +
            targets.sum(
                dim=(1, 2, 3)
            )
        )

        dice = (
            2.0 * intersection + smooth
        ) / (
            denominator + smooth
        )

        dice_loss = (
            1.0 - dice
        ).mean()

        total = (
            self.bce_weight * bce
            +
            self.dice_weight * dice_loss
        )

        return total


# ============================================================
# Metrics
# ============================================================

def binary_counts(
    prediction,
    target,
):

    prediction = prediction.astype(
        np.bool_
    )

    target = target.astype(
        np.bool_
    )

    tp = np.logical_and(
        prediction,
        target,
    ).sum()

    fp = np.logical_and(
        prediction,
        np.logical_not(target),
    ).sum()

    fn = np.logical_and(
        np.logical_not(prediction),
        target,
    ).sum()

    tn = np.logical_and(
        np.logical_not(prediction),
        np.logical_not(target),
    ).sum()

    return (
        int(tp),
        int(fp),
        int(fn),
        int(tn),
    )


def metrics_from_counts(
    tp,
    fp,
    fn,
    tn,
):

    eps = 1e-8

    dice = (
        2 * tp
    ) / (
        2 * tp
        + fp
        + fn
        + eps
    )

    iou = (
        tp
    ) / (
        tp
        + fp
        + fn
        + eps
    )

    precision = (
        tp
    ) / (
        tp
        + fp
        + eps
    )

    recall = (
        tp
    ) / (
        tp
        + fn
        + eps
    )

    specificity = (
        tn
    ) / (
        tn
        + fp
        + eps
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
    }


# ============================================================
# Training epoch
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    use_amp,
):

    model.train()

    running_loss = 0.0
    samples = 0

    for batch in loader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.cuda.amp.autocast(
            enabled=use_amp
        ):

            logits = model(images)

            loss = criterion(
                logits,
                masks,
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        batch_size = images.size(0)

        running_loss += (
            float(loss.item())
            * batch_size
        )

        samples += batch_size

    return (
        running_loss
        /
        max(samples, 1)
    )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    threshold,
    use_amp,
):

    model.eval()

    total_loss = 0.0
    samples = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    for batch in loader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        with torch.cuda.amp.autocast(
            enabled=use_amp
        ):

            logits = model(images)

            loss = criterion(
                logits,
                masks,
            )

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities
            >= threshold
        )

        batch_size = images.size(0)

        total_loss += (
            float(loss.item())
            * batch_size
        )

        samples += batch_size

        # Crop padded 512x1024 prediction
        # back to original 512x1000.
        for i in range(batch_size):

            height = int(
                batch["height"][i]
            )

            width = int(
                batch["width"][i]
            )

            pred = (
                predictions[
                    i,
                    0,
                    :height,
                    :width,
                ]
                .cpu()
                .numpy()
            )

            truth = (
                masks[
                    i,
                    0,
                    :height,
                    :width,
                ]
                .cpu()
                .numpy()
                > 0.5
            )

            tp, fp, fn, tn = (
                binary_counts(
                    pred,
                    truth,
                )
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    metrics = metrics_from_counts(
        total_tp,
        total_fp,
        total_fn,
        total_tn,
    )

    metrics["loss"] = (
        total_loss
        /
        max(samples, 1)
    )

    return metrics


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--fold",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--out_dir",
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch_size",
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
        "--patience",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--encoder",
        default="resnet34",
    )

    parser.add_argument(
        "--encoder_weights",
        default="imagenet",
    )

    args = parser.parse_args()


    set_seed(args.seed)


    manifest_path = Path(
        args.manifest
    )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    df = pd.read_csv(
        manifest_path
    )


    # ========================================================
    # Leakage-safe role selection
    # ========================================================

    train_df = df[
        df["role"]
        ==
        "model_train"
    ].copy()

    val_df = df[
        df["role"]
        ==
        "model_val"
    ].copy()

    test_df = df[
        df["role"]
        ==
        "outer_test"
    ].copy()


    if len(train_df) == 0:
        raise RuntimeError(
            "No model_train slices."
        )

    if len(val_df) == 0:
        raise RuntimeError(
            "No model_val slices."
        )

    if len(test_df) == 0:
        raise RuntimeError(
            "No outer_test slices."
        )


    train_subjects = set(
        train_df["subject_key"]
    )

    val_subjects = set(
        val_df["subject_key"]
    )

    test_subjects = set(
        test_df["subject_key"]
    )


    assert train_subjects.isdisjoint(
        val_subjects
    )

    assert train_subjects.isdisjoint(
        test_subjects
    )

    assert val_subjects.isdisjoint(
        test_subjects
    )


    # Explicitly prove outer test
    # is not used by the DataLoaders below.
    print()
    print("=" * 80)
    print("LEAKAGE-SAFE U-NET++ TRAINING")
    print("=" * 80)

    print("Fold:", args.fold)
    print("Seed:", args.seed)

    print(
        "Training subjects:",
        len(train_subjects),
    )

    print(
        "Validation subjects:",
        len(val_subjects),
    )

    print(
        "Outer-test subjects RESERVED:",
        len(test_subjects),
    )

    print(
        "Training slices:",
        len(train_df),
    )

    print(
        "Validation slices:",
        len(val_df),
    )

    print(
        "Outer-test slices NOT USED:",
        len(test_df),
    )

    print(
        "Outer-test leakage check: PASS"
    )


    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print()
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )


    # ========================================================
    # Data loaders
    # ========================================================

    generator = torch.Generator()

    generator.manual_seed(
        args.seed
    )


    train_loader = DataLoader(
        RPEDCDataset(train_df),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=(
            args.workers > 0
        ),
    )


    val_loader = DataLoader(
        RPEDCDataset(val_df),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=(
            args.workers > 0
        ),
    )


    # ========================================================
    # Model
    # ========================================================

    encoder_weights = (
        None
        if args.encoder_weights.lower()
        ==
        "none"
        else args.encoder_weights
    )


    model = smp.UnetPlusPlus(
        encoder_name=args.encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)


    criterion = BCEDiceLoss(
        bce_weight=0.4,
        dice_weight=0.6,
        pos_weight=3.0,
    ).to(device)


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
            patience=2,
        )
    )


    use_amp = (
        device.type == "cuda"
    )


    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )


    # ========================================================
    # Training
    # ========================================================

    best_dice = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    history = []


    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )


        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
            use_amp=use_amp,
        )


        scheduler.step(
            val_metrics["dice"]
        )


        current_lr = optimizer.param_groups[
            0
        ]["lr"]


        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss":
                val_metrics["loss"],
            "val_dice":
                val_metrics["dice"],
            "val_iou":
                val_metrics["iou"],
            "val_precision":
                val_metrics["precision"],
            "val_recall":
                val_metrics["recall"],
            "val_specificity":
                val_metrics["specificity"],
            "lr":
                current_lr,
        }

        history.append(row)


        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"dice={val_metrics['dice']:.6f} | "
            f"iou={val_metrics['iou']:.6f} | "
            f"lr={current_lr:.2e}"
        )


        if (
            val_metrics["dice"]
            >
            best_dice
        ):

            best_dice = (
                val_metrics["dice"]
            )

            best_epoch = epoch

            epochs_without_improvement = 0


            checkpoint = {
                "architecture":
                    "unetpp",

                "encoder":
                    args.encoder,

                "encoder_weights":
                    encoder_weights,

                "in_channels":
                    3,

                "classes":
                    1,

                "threshold":
                    args.threshold,

                "fold":
                    args.fold,

                "seed":
                    args.seed,

                "best_epoch":
                    best_epoch,

                "best_val_dice":
                    best_dice,

                "validation_metrics":
                    val_metrics,

                "training_subjects":
                    len(train_subjects),

                "validation_subjects":
                    len(val_subjects),

                "outer_test_subjects":
                    len(test_subjects),

                "training_slices":
                    len(train_df),

                "validation_slices":
                    len(val_df),

                "model_state_dict":
                    model.state_dict(),
            }


            torch.save(
                checkpoint,
                out_dir / "best.pt",
            )


            print(
                "  -> Saved new best checkpoint"
            )


        else:

            epochs_without_improvement += 1


        pd.DataFrame(
            history
        ).to_csv(
            out_dir / "history.csv",
            index=False,
        )


        if (
            epochs_without_improvement
            >=
            args.patience
        ):

            print(
                f"Early stopping at epoch "
                f"{epoch}."
            )

            break


    # ========================================================
    # Save run metadata
    # ========================================================

    config = {
        "fold": args.fold,
        "seed": args.seed,
        "architecture": "unetpp",
        "encoder": args.encoder,
        "encoder_weights": encoder_weights,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "lr": args.lr,
        "weight_decay":
            args.weight_decay,
        "patience": args.patience,
        "threshold": args.threshold,
        "bce_weight": 0.4,
        "dice_weight": 0.6,
        "pos_weight": 3.0,
        "best_epoch": best_epoch,
        "best_val_dice": best_dice,
        "training_subjects":
            len(train_subjects),
        "validation_subjects":
            len(val_subjects),
        "outer_test_subjects":
            len(test_subjects),
        "training_slices":
            len(train_df),
        "validation_slices":
            len(val_df),
        "outer_test_slices":
            len(test_df),
        "outer_test_used_for_training":
            False,
        "resized":
            False,
        "original_height":
            512,
        "original_width":
            1000,
        "network_input_height":
            512,
        "network_input_width":
            1024,
        "axial_spacing_um":
            3.23,
        "lateral_spacing_um":
            6.54,
        "bscan_spacing_um":
            67.0,
    }


    with open(
        out_dir / "config.json",
        "w",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )


    print()
    print("=" * 80)
    print("TRAINING FINISHED")
    print("=" * 80)

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation Dice:",
        best_dice,
    )

    print(
        "Saved:",
        out_dir / "best.pt",
    )


if __name__ == "__main__":
    main()