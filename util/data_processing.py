import pandas as pd

import os


def one_hot(df: pd.DataFrame, col: str, verbose=False):
    if col not in df.columns:
        raise ValueError(
            f'No such column as "{col}" in the columns of dataframe.\nThe columns are:\n{", ".join(df.columns)}'
        )
    keys = pd.unique(df[col].dropna())
    for key in keys:
        new_col = col + "_is_" + str(key)
        df[new_col] = df[col].apply(lambda s: s == key)
        if verbose:
            print(f'Created new feature "{new_col}" from cols')
    if verbose:
        term_size = os.get_terminal_size().columns
        print("=" * term_size)
        print(f'Created a total of {len(keys)} features from column "col"')
