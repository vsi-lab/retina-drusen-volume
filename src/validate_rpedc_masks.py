
#!/usr/bin/env python3



import argparse



import numpy as np

import pandas as pd

from PIL import Image





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)

    args = parser.parse_args()



    df = pd.read_csv(args.manifest)



    rows = []



    for _, row in df.iterrows():

        image = np.array(Image.open(row["image_path"]))

        mask = np.array(Image.open(row["rpedc_mask_path"]))



        unique_values = np.unique(mask)

        foreground = mask > 0



        rows.append(

            {

                "subject": row["subject"],

                "group": row["group"],

                "bscan_idx": row["bscan_idx"],

                "same_shape": image.shape[:2] == mask.shape[:2],

                "unique_values": "|".join(map(str, unique_values.tolist())),

                "foreground_pixels": int(foreground.sum()),

                "foreground_fraction": float(foreground.mean()),

                "all_background": bool(foreground.sum() == 0),

            }

        )



    report = pd.DataFrame(rows)



    print("Total masks:", len(report))

    print("Shape mismatch:", (~report["same_shape"]).sum())

    print("All-background masks:", report["all_background"].sum())

    print()

    print("Foreground fraction summary:")

    print(report["foreground_fraction"].describe())

    print()

    print("By group:")

    print(

        report.groupby("group")["foreground_fraction"]

        .agg(["count", "mean", "median", "min", "max"])

    )





if __name__ == "__main__":

    main()

