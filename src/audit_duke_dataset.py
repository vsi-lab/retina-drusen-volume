
#!/usr/bin/env python3



import argparse

from pathlib import Path



import numpy as np

import pandas as pd

import scipy.io as sio

from tqdm import tqdm





def inspect_file(path: Path, group: str):

    row = {

        "subject": path.stem,

        "group": group,

        "mat_path": str(path.resolve()),

        "read_status": "failed",

        "error": "",

    }



    try:

        data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)



        required = {"images", "layerMaps", "Age"}

        missing = required.difference(data.keys())



        if missing:

            raise KeyError(f"Missing keys: {sorted(missing)}")



        images = np.asarray(data["images"])

        layers = np.asarray(data["layerMaps"])

        age = np.asarray(data["Age"]).squeeze()



        if images.ndim != 3:

            raise ValueError(f"images ndim={images.ndim}, expected 3")



        if layers.ndim != 3:

            raise ValueError(f"layerMaps ndim={layers.ndim}, expected 3")



        height, width, num_bscans = images.shape



        if layers.shape[0] != num_bscans:

            raise ValueError(

                f"B-scan mismatch: images={num_bscans}, "

                f"layerMaps={layers.shape[0]}"

            )



        if layers.shape[1] != width:

            raise ValueError(

                f"Width mismatch: images={width}, "

                f"layerMaps={layers.shape[1]}"

            )



        valid_counts = [

            int(np.isfinite(layers[:, :, channel]).sum())

            for channel in range(layers.shape[2])

        ]



        medians = [

            float(np.nanmedian(layers[:, :, channel]))

            for channel in range(layers.shape[2])

        ]



        row.update(

            {

                "age": float(age),

                "height": height,

                "width": width,

                "num_bscans": num_bscans,

                "num_boundaries": layers.shape[2],

                "valid_boundary_0": valid_counts[0],

                "valid_boundary_1": valid_counts[1],

                "valid_boundary_2": valid_counts[2],

                "median_boundary_0": medians[0],

                "median_boundary_1": medians[1],

                "median_boundary_2": medians[2],

                "image_min": int(images.min()),

                "image_max": int(images.max()),

                "image_mean": float(images.mean()),

                "read_status": "ok",

                "error": "",

            }

        )



    except Exception as exc:

        row["error"] = str(exc)



    return row





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--amd_dir", required=True)

    parser.add_argument("--control_dir", required=True)

    parser.add_argument("--out_csv", required=True)

    args = parser.parse_args()



    rows = []



    amd_files = sorted(Path(args.amd_dir).rglob("*.mat"))

    control_files = sorted(Path(args.control_dir).rglob("*.mat"))



    print("AMD files:", len(amd_files))

    print("Control files:", len(control_files))



    for path in tqdm(amd_files, desc="Auditing AMD"):

        rows.append(inspect_file(path, "AMD"))



    for path in tqdm(control_files, desc="Auditing Control"):

        rows.append(inspect_file(path, "Control"))



    df = pd.DataFrame(rows)



    out_path = Path(args.out_csv)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)



    print()

    print("Saved:", out_path)

    print()

    print("Read status:")

    print(df["read_status"].value_counts(dropna=False))

    print()

    print("Group counts:")

    print(df.groupby(["group", "read_status"]).size())





if __name__ == "__main__":

    main()

