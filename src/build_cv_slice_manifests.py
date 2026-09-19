
#!/usr/bin/env python3



import re

from pathlib import Path

import pandas as pd





FULL_MANIFEST = Path(

    "data/manifests/duke_full_rpedc_manifest.csv"

)



SUBJECT_SPLITS = Path(

    "data/splits/duke_5fold_subject_roles.csv"

)



OUT_DIR = Path(

    "data/manifests/cv"

)



OUT_DIR.mkdir(

    parents=True,

    exist_ok=True,

)





PATTERN = re.compile(

    r"Farsiu_Ophthalmology_2013_"

    r"(AMD|Control)_Subject_(\d+)"

)





def make_subject_key(name):

    match = PATTERN.search(str(name))



    if match is None:

        raise ValueError(

            f"Cannot parse subject name: {name}"

        )



    return (

        f"{match.group(1)}_"

        f"{int(match.group(2))}"

    )





manifest = pd.read_csv(FULL_MANIFEST)

splits = pd.read_csv(SUBJECT_SPLITS)



manifest["subject_key"] = (

    manifest["subject"]

    .apply(make_subject_key)

)





print("=" * 80)

print("BUILDING FIVE-FOLD SLICE MANIFESTS")

print("=" * 80)





for fold in range(1, 6):



    fold_split = splits[

        splits["fold"] == fold

    ].copy()



    merged = manifest.merge(

        fold_split[

            [

                "subject",

                "role",

                "atlas_eligible",

            ]

        ],

        left_on="subject_key",

        right_on="subject",

        how="inner",

        suffixes=("", "_split"),

        validate="many_to_one",

    )



    # ---------------------------------

    # Subject-level leakage checks

    # ---------------------------------



    train_subjects = set(

        merged[

            merged["role"] == "model_train"

        ]["subject_key"]

    )



    val_subjects = set(

        merged[

            merged["role"] == "model_val"

        ]["subject_key"]

    )



    test_subjects = set(

        merged[

            merged["role"] == "outer_test"

        ]["subject_key"]

    )



    assert train_subjects.isdisjoint(

        val_subjects

    )



    assert train_subjects.isdisjoint(

        test_subjects

    )



    assert val_subjects.isdisjoint(

        test_subjects

    )



    # ---------------------------------

    # Atlas safety

    # ---------------------------------



    atlas = merged[

        merged["atlas_eligible"]

    ]



    assert (

        atlas["group"] == "Control"

    ).all()



    assert (

        atlas["role"] == "model_train"

    ).all()



    # ---------------------------------

    # Save

    # ---------------------------------



    out = OUT_DIR / f"fold_{fold}.csv"



    merged.to_csv(

        out,

        index=False,

    )



    print()

    print("=" * 65)

    print(f"FOLD {fold}")

    print("=" * 65)



    print("\nSUBJECT COUNTS")

    print(

        merged.groupby(

            ["role", "group"]

        )["subject_key"].nunique()

    )



    print("\nSLICE COUNTS")

    print(

        merged.groupby(

            ["role", "group"]

        ).size()

    )



    print(

        "\nAtlas controls:",

        atlas["subject_key"].nunique()

    )



    print(

        "Total slices:",

        len(merged)

    )



    print(

        "Saved:",

        out

    )





print()

print("=" * 80)

print("ALL FIVE FOLD MANIFESTS CREATED: PASS")

print("=" * 80)

