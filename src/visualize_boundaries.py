
#!/usr/bin/env python3



import argparse

from pathlib import Path



import matplotlib.pyplot as plt

import numpy as np

import scipy.io as sio





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--mat_file", required=True)

    parser.add_argument("--bscan", type=int, default=50)

    parser.add_argument("--out_png", required=True)

    args = parser.parse_args()



    mat_path = Path(args.mat_file)

    out_path = Path(args.out_png)



    if not mat_path.exists():

        raise FileNotFoundError(mat_path)



    data = sio.loadmat(

        mat_path,

        squeeze_me=True,

        struct_as_record=False,

    )



    images = np.asarray(data["images"])

    layer_maps = np.asarray(data["layerMaps"])



    if images.ndim != 3:

        raise RuntimeError(f"Unexpected image shape: {images.shape}")



    if layer_maps.ndim != 3:

        raise RuntimeError(f"Unexpected layerMaps shape: {layer_maps.shape}")



    if not 0 <= args.bscan < images.shape[2]:

        raise ValueError(

            f"B-scan index must be between 0 and {images.shape[2] - 1}"

        )



    image = images[:, :, args.bscan]



    plt.figure(figsize=(16, 7))

    plt.imshow(image, cmap="gray")



    for channel in range(layer_maps.shape[2]):

        boundary = layer_maps[args.bscan, :, channel]

        valid = np.isfinite(boundary)



        x = np.arange(boundary.shape[0])[valid]



        # MATLAB boundary positions are normally one-based.

        y = boundary[valid] - 1



        plt.plot(

            x,

            y,

            linewidth=1.5,

            label=f"Channel {channel}",

        )



    plt.xlim(0, image.shape[1])

    plt.ylim(image.shape[0], 0)

    plt.xlabel("A-scan column")

    plt.ylabel("Axial depth")

    plt.title(f"{mat_path.name}, B-scan {args.bscan}")

    plt.legend()

    plt.tight_layout()



    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path, dpi=200)

    plt.close()



    print("Saved:", out_path.resolve())





if __name__ == "__main__":

    main()

