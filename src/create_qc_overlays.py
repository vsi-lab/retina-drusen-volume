
#!/usr/bin/env python3



import argparse

from pathlib import Path



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from PIL import Image





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)

    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--samples_per_group", type=int, default=20)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()



    df = pd.read_csv(args.manifest)

    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)



    samples = []



    for group, group_df in df.groupby("group"):

        n = min(args.samples_per_group, len(group_df))



        samples.append(

            group_df.sample(

                n=n,

                random_state=args.seed,

            )

        )



    sampled = pd.concat(samples, ignore_index=True)



    for _, row in sampled.iterrows():

        image = np.array(Image.open(row["image_path"]).convert("L"))

        mask = np.array(Image.open(row["rpedc_mask_path"])) > 0



        boundary_data = np.load(row["boundary_path"])

        upper = boundary_data["upper_boundary"]

        lower = boundary_data["lower_boundary"]



        valid_upper = np.isfinite(upper)

        valid_lower = np.isfinite(lower)



        fig, axes = plt.subplots(1, 3, figsize=(18, 6))



        axes[0].imshow(image, cmap="gray")

        axes[0].set_title("OCT image")

        axes[0].axis("off")



        axes[1].imshow(image, cmap="gray")

        axes[1].plot(

            np.arange(len(upper))[valid_upper],

            upper[valid_upper] - 1,

            linewidth=1,

            label="RPEDC upper",

        )

        axes[1].plot(

            np.arange(len(lower))[valid_lower],

            lower[valid_lower] - 1,

            linewidth=1,

            label="Bruch's membrane",

        )

        axes[1].set_ylim(image.shape[0], 0)

        axes[1].set_title("Boundary overlay")

        axes[1].legend()

        axes[1].axis("off")



        axes[2].imshow(image, cmap="gray")

        axes[2].imshow(mask, alpha=0.35)

        axes[2].set_title("RPEDC mask overlay")

        axes[2].axis("off")



        plt.suptitle(

            f"{row['subject']} | {row['group']} | "

            f"B-scan {int(row['bscan_idx'])}"

        )

        plt.tight_layout()



        output_name = (

            f"{row['group']}_{row['subject']}_"

            f"b{int(row['bscan_idx']):03d}.png"

        )



        plt.savefig(out_dir / output_name, dpi=150)

        plt.close()



    print("Saved overlays:", len(sampled))

    print("Output folder:", out_dir)





if __name__ == "__main__":

    main()

