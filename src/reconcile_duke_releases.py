
#!/usr/bin/env python3



from pathlib import Path

import pandas as pd





OLD_PATH = Path(

    "reports/cohort_audit/current_release_inventory.csv"

)



FULL_PATH = Path(

    "reports/cohort_audit/full_release_inventory.csv"

)



OUT_PATH = Path(

    "reports/cohort_audit/release_reconciliation.csv"

)





old = pd.read_csv(OLD_PATH)

full = pd.read_csv(FULL_PATH)





old_cols = [

    "subject_key",

    "subject_id",

    "group",

    "filename",

    "age",

    "num_bscans",

    "image_shape",

    "layer_shape",

    "sha256",

    "inclusion_status",

    "full_path",

]



full_cols = old_cols.copy()





old2 = old[old_cols].copy()

full2 = full[full_cols].copy()





merged = full2.merge(

    old2,

    on=[

        "subject_key",

        "subject_id",

        "group",

    ],

    how="left",

    suffixes=(

        "_full",

        "_old",

    ),

    indicator=True,

)





def classify(row):



    if row["_merge"] == "left_only":

        return "newly_recovered_in_full_release"



    old_hash = str(

        row.get("sha256_old", "")

    )



    full_hash = str(

        row.get("sha256_full", "")

    )



    if (

        old_hash

        and full_hash

        and old_hash != "nan"

        and full_hash != "nan"

        and old_hash == full_hash

    ):

        return "present_in_both_identical"



    return "present_in_both_check_file_difference"





merged[

    "release_status"

] = merged.apply(

    classify,

    axis=1,

)





OUT_PATH.parent.mkdir(

    parents=True,

    exist_ok=True,

)



merged.to_csv(

    OUT_PATH,

    index=False,

)





print("\n=== RELEASE STATUS ===")

print(

    merged[

        "release_status"

    ].value_counts()

)



print("\n=== RELEASE STATUS BY GROUP ===")

print(

    merged.groupby(

        [

            "group",

            "release_status",

        ]

    ).size()

)



print("\n=== FULL COHORT ===")

print(

    merged.groupby("group").size()

)



print("\nSaved:", OUT_PATH)

