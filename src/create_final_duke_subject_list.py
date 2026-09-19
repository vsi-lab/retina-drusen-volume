
#!/usr/bin/env python3



from pathlib import Path

import pandas as pd





FULL = Path(

    "reports/cohort_audit/full_release_inventory.csv"

)



RECON = Path(

    "reports/cohort_audit/release_reconciliation.csv"

)



OUT = Path(

    "reports/cohort_audit/final_subject_inclusion_list.csv"

)





full = pd.read_csv(FULL)



recon = pd.read_csv(RECON)[

    [

        "subject_key",

        "release_status",

    ]

].copy()





df = full.merge(

    recon,

    on="subject_key",

    how="left",

    validate="one_to_one",

)





# Default: all structurally valid official subjects included.

df["paper_inclusion_status"] = "included"

df["paper_exclusion_reason"] = ""





# Special 82-B-scan control:

# preserve in master cohort but do not use for quantitative

# CV until physical slow-scan spacing is verified.

mask = (

    df["subject_key"]

    ==

    "Control_1088"

)



df.loc[

    mask,

    "paper_inclusion_status"

] = "pending_spacing_verification"



df.loc[

    mask,

    "paper_exclusion_reason"

] = (

    "Official file contains 82 B-scans rather than 100; "

    "slow-scan physical spacing must be verified before "

    "quantitative volume analysis."

)





columns = [

    "subject_key",

    "subject_id",

    "group",

    "age",

    "filename",

    "source_tag",

    "release_status",

    "image_shape",

    "layer_shape",

    "num_bscans",

    "image_height",

    "num_ascans",

    "layer_channels",

    "read_status",

    "inclusion_status",

    "paper_inclusion_status",

    "paper_exclusion_reason",

    "sha256",

    "full_path",

]





df = df[

    columns

].sort_values(

    [

        "group",

        "subject_id",

    ]

)





OUT.parent.mkdir(

    parents=True,

    exist_ok=True,

)



df.to_csv(

    OUT,

    index=False,

)





print("\n=== FINAL MASTER COHORT ===")

print("Total subjects:", len(df))



print("\nBy group:")

print(

    df.groupby("group").size()

)



print("\nPaper inclusion status:")

print(

    df.groupby(

        [

            "group",

            "paper_inclusion_status",

        ]

    ).size()

)



print("\nPending/non-included subjects:")

pending = df[

    df["paper_inclusion_status"]

    !=

    "included"

]



if len(pending) == 0:

    print("None")

else:

    print(

        pending[

            [

                "subject_key",

                "group",

                "age",

                "num_bscans",

                "paper_inclusion_status",

                "paper_exclusion_reason",

            ]

        ].to_string(index=False)

    )



print("\nSaved:", OUT)

