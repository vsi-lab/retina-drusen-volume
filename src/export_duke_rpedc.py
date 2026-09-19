#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image
from tqdm import tqdm


def build_region_mask(
    upper_boundary: np.ndarray,
    lower_boundary: np.ndarray,
    height: int,
    width: int,
):
    mask = np.zeros((height, width), dtype=np.uint8)
    valid_columns = np.zeros(width, dtype=np.uint8)

    for x in range(width):
        upper = upper_boundary[x]
        lower = lower_boundary[x]

        if not np.isfinite(upper) or not np.isfinite(lower):
            continue

        # MATLAB boundary coordinates are 1-based.
        upper_idx = int(round(float(upper))) - 1
        lower_idx = int(round(float(lower))) - 1

        upper_idx = int(np.clip(upper_idx, 0, height - 1))
        lower_idx = int(np.clip(lower_idx, 0, height - 1))

        if lower_idx <= upper_idx:
            continue

        mask[upper_idx : lower_idx + 1, x] = 1
        valid_columns[x] = 1

    return mask, valid_columns


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)

    low = float(np.percentile(image, 1))
    high = float(np.percentile(image, 99))

    if high <= low:
        return np.clip(image, 0, 255).astype(np.uint8)

    image = (image - low) / (high - low)
    image = np.clip(image, 0.0, 1.0)

    return (image * 255.0).astype(np.uint8)


def export_subject(
    mat_path: Path,
    group: str,
    out_root: Path,
    upper_channel: int,
    lower_channel: int,
    minimum_valid_columns: int,
):
    data = sio.loadmat(
        mat_path,
        squeeze_me=True,
        struct_as_record=False,
    )

    if "images" not in data or "layerMaps" not in data:
        raise KeyError(
            f"Missing images or layerMaps in {mat_path.name}"
        )

    images = np.asarray(data["images"])
    layers = np.asarray(data["layerMaps"])
    age = float(np.asarray(data.get("Age", np.nan)).squeeze())

    if images.ndim != 3:
        raise ValueError(
            f"Unexpected images shape {images.shape} in {mat_path.name}"
        )

    if layers.ndim != 3:
        raise ValueError(
            f"Unexpected layerMaps shape {layers.shape} in {mat_path.name}"
        )

    if upper_channel >= layers.shape[2]:
        raise ValueError(
            f"upper_channel={upper_channel} is outside layerMaps shape "
            f"{layers.shape}"
        )

    if lower_channel >= layers.shape[2]:
        raise ValueError(
            f"lower_channel={lower_channel} is outside layerMaps shape "
            f"{layers.shape}"
        )

    height, width, num_bscans = images.shape

    if layers.shape[0] != num_bscans:
        raise ValueError(
            f"B-scan count mismatch in {mat_path.name}: "
            f"images={num_bscans}, layerMaps={layers.shape[0]}"
        )

    if layers.shape[1] != width:
        raise ValueError(
            f"Width mismatch in {mat_path.name}: "
            f"images={width}, layerMaps={layers.shape[1]}"
        )

    subject = mat_path.stem

    image_dir = out_root / "images"
    mask_dir = out_root / "rpedc_masks"
    boundary_dir = out_root / "boundaries"

    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    boundary_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for bscan_idx in range(num_bscans):
        image = images[:, :, bscan_idx]

        upper = layers[bscan_idx, :, upper_channel]
        lower = layers[bscan_idx, :, lower_channel]

        mask, valid_columns = build_region_mask(
            upper_boundary=upper,
            lower_boundary=lower,
            height=height,
            width=width,
        )

        valid_count = int(valid_columns.sum())
        valid_fraction = float(valid_count / width)

        if valid_count < minimum_valid_columns:
            continue

        stem = f"{subject}_b{bscan_idx:03d}"

        image_path = image_dir / f"{stem}.png"
        mask_path = mask_dir / f"{stem}.png"
        boundary_path = boundary_dir / f"{stem}.npz"

        Image.fromarray(
            normalize_image(image),
            mode="L",
        ).save(image_path)

        Image.fromarray(
            (mask * 255).astype(np.uint8),
            mode="L",
        ).save(mask_path)

        np.savez_compressed(
            boundary_path,
            upper_boundary=upper,
            lower_boundary=lower,
            valid_columns=valid_columns,
            upper_channel=np.int32(upper_channel),
            lower_channel=np.int32(lower_channel),
        )

        rows.append(
            {
                "subject": subject,
                "group": group,
                "age": age,
                "bscan_idx": bscan_idx,
                "image_path": str(image_path.resolve()),
                "rpedc_mask_path": str(mask_path.resolve()),
                "boundary_path": str(boundary_path.resolve()),
                "height": height,
                "width": width,
                "valid_columns": valid_count,
                "valid_fraction": valid_fraction,
                "mask_pixels": int(mask.sum()),
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--amd_dir", required=True)
    parser.add_argument("--control_dir", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--manifest", required=True)

    parser.add_argument(
        "--upper_channel",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--lower_channel",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--minimum_valid_columns",
        type=int,
        default=250,
    )

    args = parser.parse_args()

    amd_dir = Path(args.amd_dir)
    control_dir = Path(args.control_dir)
    out_root = Path(args.out_root)
    manifest_path = Path(args.manifest)

    if not amd_dir.exists():
        raise FileNotFoundError(amd_dir)

    if not control_dir.exists():
        raise FileNotFoundError(control_dir)

    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []

    groups = [
        ("AMD", amd_dir),
        ("Control", control_dir),
    ]

    for group, root in groups:
        mat_files = sorted(root.rglob("*.mat"))

        print(f"{group} files found: {len(mat_files)}")

        for mat_path in tqdm(
            mat_files,
            desc=f"Exporting {group}",
        ):
            try:
                subject_rows = export_subject(
                    mat_path=mat_path,
                    group=group,
                    out_root=out_root,
                    upper_channel=args.upper_channel,
                    lower_channel=args.lower_channel,
                    minimum_valid_columns=args.minimum_valid_columns,
                )

                rows.extend(subject_rows)

            except Exception as exc:
                failures.append(
                    {
                        "group": group,
                        "mat_path": str(mat_path.resolve()),
                        "error": str(exc),
                    }
                )

                print(f"[SKIP] {mat_path.name}: {exc}")

    if not rows:
        raise RuntimeError(
            "No slices were exported. Check the channel indices and "
            "minimum_valid_columns threshold."
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)

    failure_path = manifest_path.with_name(
        manifest_path.stem + "_failures.csv"
    )

    pd.DataFrame(failures).to_csv(
        failure_path,
        index=False,
    )

    print()
    print("Saved manifest:", manifest_path)
    print("Saved failure log:", failure_path)
    print("Exported slices:", len(manifest))
    print()

    print("Slices by group:")
    print(manifest.groupby("group").size())
    print()

    print("Subjects by group:")
    print(manifest.groupby("group")["subject"].nunique())
    print()

    print("Valid-column summary:")
    print(manifest["valid_columns"].describe())
    print()

    print("Mask-pixel summary:")
    print(manifest["mask_pixels"].describe())


if __name__ == "__main__":
    main()
