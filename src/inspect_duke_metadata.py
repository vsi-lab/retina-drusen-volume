
#!/usr/bin/env python3



from pathlib import Path

import numpy as np

import scipy.io as sio





ROOTS = [

    Path("data/raw/duke_amd/extracted/amd"),

    Path("data/raw/duke_amd/extracted/control"),

]





def describe_value(key, value):

    print(f"  {key}")



    try:

        print(f"    shape: {value.shape}")

        print(f"    dtype: {value.dtype}")

    except Exception:

        print(f"    type: {type(value)}")

        return



    if value.size <= 20:

        try:

            print(f"    value: {value}")

        except Exception:

            pass



    if value.dtype.kind in {"U", "S", "O"}:

        try:

            print(f"    content: {value.squeeze()}")

        except Exception:

            pass





for root in ROOTS:



    files = sorted(root.rglob("*.mat"))



    print()

    print("=" * 90)

    print("DIRECTORY:", root)

    print("FILES:", len(files))

    print("=" * 90)



    for path in files[:5]:



        print()

        print("FILE:", path.name)



        data = sio.loadmat(

            path,

            squeeze_me=False,

            struct_as_record=False,

        )



        for key, value in data.items():



            if key.startswith("__"):

                continue



            describe_value(

                key,

                value,

            )

