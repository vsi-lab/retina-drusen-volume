
#!/usr/bin/env python3



import argparse

import json

from pathlib import Path



import pandas as pd





def main():



    parser = argparse.ArgumentParser()



    parser.add_argument(

        "--reports_root",

        required=True,

    )



    parser.add_argument(

        "--out_csv",

        required=True,

    )



    args = parser.parse_args()



    rows = []



    reports_root = Path(

        args.reports_root

    )



    for summary_file in sorted(

        reports_root.glob(

            "*/summary.json"

        )

    ):



        with open(

            summary_file

        ) as handle:



            rows.append(

                json.load(handle)

            )



    if not rows:



        raise RuntimeError(

            "No summary.json files found "

            f"under {reports_root}"

        )



    results = pd.DataFrame(

        rows

    )



    preferred_columns = [

        "architecture",

        "encoder",

        "number_of_subjects",



        "subject_mean_dice",

        "subject_std_dice",



        "subject_mean_iou",



        "subject_mean_precision",

        "subject_mean_recall",

        "subject_mean_specificity",



        "subject_mean_balanced_accuracy",

        "subject_mean_mcc",



        "subject_mean_hd95_pixels",

        "subject_mean_assd_pixels",



        "subject_mean_thickness_mae",

        "subject_mean_thickness_rmse",



        "subject_mean_area_relative_error",



        "pixel_auroc",

        "pixel_auprc",

    ]



    available_columns = [

        column

        for column

        in preferred_columns

        if column

        in results.columns

    ]



    results = results[

        available_columns

    ]



    results = results.sort_values(

        by="subject_mean_dice",

        ascending=False,

    )



    output_path = Path(

        args.out_csv

    )



    output_path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )



    results.to_csv(

        output_path,

        index=False,

    )



    print(

        results.to_string(

            index=False

        )

    )



    print(

        f"\nSaved: {output_path}"

    )





if __name__ == "__main__":

    main()

