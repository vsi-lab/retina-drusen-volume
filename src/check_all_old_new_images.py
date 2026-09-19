
#!/usr/bin/env python3



from pathlib import Path



import numpy as np

import pandas as pd

from PIL import Image





OLD_MANIFEST = Path(

    "reports/pre_cv_archive/"

    "duke_rpedc_manifest_185subjects.csv"

)



NEW_MANIFEST = Path(

    "data/manifests/"

    "duke_full_rpedc_manifest.csv"

)



OUTPUT = Path(

    "reports/full_cohort_qc/"

    "old_new_image_differences.csv"

)





if not OLD_MANIFEST.exists():

    raise FileNotFoundError(

        f"Old manifest not found: {OLD_MANIFEST}"

    )



if not NEW_MANIFEST.exists():

    raise FileNotFoundError(

        f"New manifest not found: {NEW_MANIFEST}"

    )





old = pd.read_csv(OLD_MANIFEST)

new = pd.read_csv(NEW_MANIFEST)





merged = old.merge(

    new,

    on=[

        "subject",

        "bscan_idx",

    ],

    how="inner",

    suffixes=(

        "_old",

        "_new",

    ),

    validate="one_to_one",

)





print("=" * 80)

print("EXHAUSTIVE OLD VS FULL IMAGE COMPARISON")

print("=" * 80)



print("Old manifest rows:", len(old))

print("Matched rows:", len(merged))





if len(merged) != len(old):

    raise RuntimeError(

        f"Expected {len(old)} matched rows, "

        f"found {len(merged)}"

    )





rows = []





for i, row in merged.iterrows():



    old_path = Path(

        row["image_path_old"]

    )



    new_path = Path(

        row["image_path_new"]

    )





    if not old_path.exists():

        raise FileNotFoundError(

            f"Missing old image: {old_path}"

        )



    if not new_path.exists():

        raise FileNotFoundError(

            f"Missing new image: {new_path}"

        )





    old_img = np.asarray(

        Image.open(old_path)

    )



    new_img = np.asarray(

        Image.open(new_path)

    )





    if old_img.shape != new_img.shape:



        rows.append(

            {

                "subject": row["subject"],

                "bscan_idx": row["bscan_idx"],

                "different_pixels": np.nan,

                "different_fraction": np.nan,

                "max_abs_difference": np.nan,

                "mean_abs_difference": np.nan,

                "shape_mismatch": True,

                "old_shape": str(

                    old_img.shape

                ),

                "new_shape": str(

                    new_img.shape

                ),

                "old_image_path": str(

                    old_path

                ),

                "new_image_path": str(

                    new_path

                ),

            }

        )



        continue





    if not np.array_equal(

        old_img,

        new_img,

    ):



        diff = (

            old_img.astype(np.int32)

            -

            new_img.astype(np.int32)

        )



        different = (

            diff != 0

        )





        rows.append(

            {

                "subject": row["subject"],

                "bscan_idx": row["bscan_idx"],



                "different_pixels":

                    int(

                        different.sum()

                    ),



                "different_fraction":

                    float(

                        different.mean()

                    ),



                "max_abs_difference":

                    int(

                        np.abs(diff).max()

                    ),



                "mean_abs_difference":

                    float(

                        np.abs(diff).mean()

                    ),



                "shape_mismatch":

                    False,



                "old_shape":

                    str(

                        old_img.shape

                    ),



                "new_shape":

                    str(

                        new_img.shape

                    ),



                "old_image_path":

                    str(old_path),



                "new_image_path":

                    str(new_path),

            }

        )





    if (

        (i + 1) % 1000 == 0

        or

        (i + 1) == len(merged)

    ):



        print(

            f"Checked "

            f"{i + 1}/{len(merged)}"

        )





result = pd.DataFrame(rows)





OUTPUT.parent.mkdir(

    parents=True,

    exist_ok=True,

)





# Always create a readable CSV, even if there

# are zero image differences.



if len(result) == 0:



    result = pd.DataFrame(

        columns=[

            "subject",

            "bscan_idx",

            "different_pixels",

            "different_fraction",

            "max_abs_difference",

            "mean_abs_difference",

            "shape_mismatch",

            "old_shape",

            "new_shape",

            "old_image_path",

            "new_image_path",

        ]

    )





result.to_csv(

    OUTPUT,

    index=False,

)





print()

print("=" * 80)

print("RESULT")

print("=" * 80)



print(

    "Total images compared:",

    len(merged)

)



print(

    "Different images:",

    len(result)

)





if len(result) > 0:



    print(

        "Maximum absolute difference:",

        result[

            "max_abs_difference"

        ].max()

    )



    print(

        "Maximum differing fraction:",

        result[

            "different_fraction"

        ].max()

    )



    print(

        "Median differing fraction:",

        result[

            "different_fraction"

        ].median()

    )



    print(

        "Maximum mean absolute difference:",

        result[

            "mean_abs_difference"

        ].max()

    )





    print()

    print(

        "Top discrepancies:"

    )



    print(

        result.sort_values(

            [

                "max_abs_difference",

                "different_fraction",

            ],

            ascending=False,

        )

        .head(20)

        .to_string(index=False)

    )





print()

print(

    "Saved:",

    OUTPUT

)

