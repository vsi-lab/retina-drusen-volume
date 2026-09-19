
#!/usr/bin/env python3



import argparse

from pathlib import Path



import pandas as pd

from sklearn.model_selection import train_test_split





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)

    parser.add_argument("--out_csv", required=True)

    parser.add_argument("--test_size", type=float, default=0.15)

    parser.add_argument("--val_size", type=float, default=0.15)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()



    manifest = pd.read_csv(args.manifest)



    subjects = (

        manifest[["subject", "group"]]

        .drop_duplicates()

        .reset_index(drop=True)

    )



    train_val, test = train_test_split(

        subjects,

        test_size=args.test_size,

        random_state=args.seed,

        stratify=subjects["group"],

    )



    val_fraction_of_train_val = args.val_size / (1.0 - args.test_size)



    train, val = train_test_split(

        train_val,

        test_size=val_fraction_of_train_val,

        random_state=args.seed,

        stratify=train_val["group"],

    )



    train = train.copy()

    val = val.copy()

    test = test.copy()



    train["split"] = "train"

    val["split"] = "val"

    test["split"] = "test"



    splits = pd.concat(

        [train, val, test],

        ignore_index=True,

    ).sort_values(["split", "group", "subject"])



    out_path = Path(args.out_csv)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    splits.to_csv(out_path, index=False)



    print("Saved:", out_path)

    print()

    print(splits.groupby(["split", "group"]).size())

    print()

    print("Total subjects:", len(splits))





if __name__ == "__main__":

    main()

