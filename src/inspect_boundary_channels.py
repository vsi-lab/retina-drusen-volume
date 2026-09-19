
#!/usr/bin/env python3



import argparse

from pathlib import Path



import numpy as np

import scipy.io as sio





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--mat_file", required=True)

    args = parser.parse_args()



    path = Path(args.mat_file)



    if not path.exists():

        raise FileNotFoundError(path)



    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)



    if "images" not in data or "layerMaps" not in data:

        raise RuntimeError(

            f"Required variables are missing. Available keys: "

            f"{[key for key in data if not key.startswith('__')]}"

        )



    images = np.asarray(data["images"])

    layer_maps = np.asarray(data["layerMaps"])



    print("File:", path)

    print("Image shape:", images.shape)

    print("Layer map shape:", layer_maps.shape)

    print()



    if layer_maps.ndim != 3:

        raise RuntimeError(

            f"Expected layerMaps with three dimensions, found {layer_maps.shape}"

        )



    medians = []



    for channel in range(layer_maps.shape[2]):

        values = layer_maps[:, :, channel]

        valid = np.isfinite(values)



        if valid.sum() == 0:

            median = np.nan

            mean = np.nan

            minimum = np.nan

            maximum = np.nan

        else:

            median = float(np.nanmedian(values))

            mean = float(np.nanmean(values))

            minimum = float(np.nanmin(values))

            maximum = float(np.nanmax(values))



        medians.append(median)



        print(f"Channel {channel}")

        print(" valid points:", int(valid.sum()))

        print(" missing points:", int((~valid).sum()))

        print(" valid fraction:", float(valid.mean()))

        print(" median depth:", median)

        print(" mean depth:", mean)

        print(" minimum:", minimum)

        print(" maximum:", maximum)

        print()



    valid_medians = np.asarray(medians, dtype=np.float64)



    if np.all(np.isfinite(valid_medians)):

        order = np.argsort(valid_medians)



        print(

            "Channels ordered from shallowest to deepest:",

            order.tolist(),

        )

        print()

        print("Expected interpretation:")

        print(f"ILM candidate: channel {order[0]}")

        print(f"RPEDC upper-boundary candidate: channel {order[1]}")

        print(f"Bruch's membrane candidate: channel {order[2]}")

    else:

        print("One or more channels had no valid annotations.")





if __name__ == "__main__":

    main()

