
#!/usr/bin/env python3



from pathlib import Path



import pandas as pd



from sklearn.model_selection import (

    StratifiedKFold,

    StratifiedShuffleSplit,

)





INPUT = Path(

    "reports/cohort_audit/final_subject_inclusion_list.csv"

)



OUTPUT = Path(

    "data/splits/duke_5fold_subject_roles.csv"

)





N_FOLDS = 5

SEED = 42



# Internal validation only for early stopping/model selection.

VAL_FRACTION = 0.10





df = pd.read_csv(INPUT)





# Only currently approved quantitative-analysis subjects.

df = df[

    df["paper_inclusion_status"]

    ==

    "included"

].copy()





if df["subject_key"].duplicated().any():

    raise RuntimeError(

        "Duplicate subject keys detected."

    )





print("\n=== CV COHORT ===")

print("Total:", len(df))



print(

    df.groupby("group").size()

)





outer_cv = StratifiedKFold(

    n_splits=N_FOLDS,

    shuffle=True,

    random_state=SEED,

)





rows = []





for fold, (

    trainval_idx,

    test_idx,

) in enumerate(

    outer_cv.split(

        df["subject_key"],

        df["group"],

    ),

    start=1,

):



    trainval = df.iloc[

        trainval_idx

    ].reset_index(drop=True)



    outer_test = df.iloc[

        test_idx

    ].reset_index(drop=True)





    # Internal validation split is drawn only

    # from the OUTER TRAINING fold.

    inner_split = StratifiedShuffleSplit(

        n_splits=1,

        test_size=VAL_FRACTION,

        random_state=SEED + fold,

    )





    model_train_idx, model_val_idx = next(

        inner_split.split(

            trainval["subject_key"],

            trainval["group"],

        )

    )





    model_train = trainval.iloc[

        model_train_idx

    ]



    model_val = trainval.iloc[

        model_val_idx

    ]





    # -----------------------------

    # MODEL TRAIN

    # -----------------------------

    for _, row in model_train.iterrows():



        rows.append(

            {

                "fold": fold,

                "subject": row["subject_key"],

                "subject_id": row["subject_id"],

                "group": row["group"],

                "age": row["age"],

                "role": "model_train",



                # Atlas comes ONLY from controls

                # inside the model-training subjects.

                "atlas_eligible": (

                    row["group"]

                    ==

                    "Control"

                ),

            }

        )





    # -----------------------------

    # MODEL VALIDATION

    # -----------------------------

    for _, row in model_val.iterrows():



        rows.append(

            {

                "fold": fold,

                "subject": row["subject_key"],

                "subject_id": row["subject_id"],

                "group": row["group"],

                "age": row["age"],

                "role": "model_val",

                "atlas_eligible": False,

            }

        )





    # -----------------------------

    # OUTER TEST

    # -----------------------------

    for _, row in outer_test.iterrows():



        rows.append(

            {

                "fold": fold,

                "subject": row["subject_key"],

                "subject_id": row["subject_id"],

                "group": row["group"],

                "age": row["age"],

                "role": "outer_test",

                "atlas_eligible": False,

            }

        )





result = pd.DataFrame(rows)





OUTPUT.parent.mkdir(

    parents=True,

    exist_ok=True,

)



result.to_csv(

    OUTPUT,

    index=False,

)





print("\n=== FOLD DISTRIBUTION ===")



print(

    result.groupby(

        [

            "fold",

            "role",

            "group",

        ]

    ).size()

)





print("\n=== ATLAS CONTROLS PER FOLD ===")



print(

    result[

        result["atlas_eligible"]

    ]

    .groupby("fold")

    .size()

)





print("\n=== LEAKAGE CHECKS ===")





for fold in range(

    1,

    N_FOLDS + 1,

):



    f = result[

        result["fold"]

        ==

        fold

    ]





    model_train = set(

        f[

            f["role"]

            ==

            "model_train"

        ]["subject"]

    )



    model_val = set(

        f[

            f["role"]

            ==

            "model_val"

        ]["subject"]

    )



    outer_test = set(

        f[

            f["role"]

            ==

            "outer_test"

        ]["subject"]

    )



    atlas = set(

        f[

            f["atlas_eligible"]

        ]["subject"]

    )





    assert model_train.isdisjoint(

        model_val

    )



    assert model_train.isdisjoint(

        outer_test

    )



    assert model_val.isdisjoint(

        outer_test

    )



    assert atlas.issubset(

        model_train

    )





    atlas_rows = f[

        f["atlas_eligible"]

    ]





    assert (

        atlas_rows["group"]

        ==

        "Control"

    ).all()



    assert (

        atlas_rows["role"]

        ==

        "model_train"

    ).all()





    print(

        f"Fold {fold}: PASS"

    )





# ----------------------------------

# Verify each subject is test exactly once

# ----------------------------------



outer = result[

    result["role"]

    ==

    "outer_test"

]



test_counts = (

    outer.groupby("subject")

    .size()

)





assert len(test_counts) == len(df)



assert (

    test_counts == 1

).all()





print(

    "\nEvery CV subject is held out "

    "exactly once: PASS"

)



print("\nSaved:", OUTPUT)

