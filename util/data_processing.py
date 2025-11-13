import pandas as pd

import os
import sys
import random


def one_hot(df: pd.DataFrame, col: str, verbose=False, infix="_is_"):
    """
    Encodes the specified column in a "one-hot" manner:
    Each of the values of the column's value set gets its own column, with boolean
    values indicating whether or not the record had that value in the original column.
    This prevents the model from interpreting string columns as numerical ones.
    """
    if col not in df.columns:
        raise ValueError(
            f'No such column as "{col}" in the columns of dataframe.\nThe columns are:\n{", ".join(df.columns)}'
        )
    keys = pd.unique(df[col].dropna())
    for key in keys:
        new_col = col + infix + str(key)
        df[new_col] = df[col].apply(lambda s: s == key)
        if verbose:
            print(f'Created new feature "{new_col}" from cols')
    if verbose:
        term_size = os.get_terminal_size().columns
        print("=" * term_size)
        print(f'Created a total of {len(keys)} features from column "col"')


def split(
    df: pd.DataFrame, percentages: list[float], random_state=None
) -> list[pd.DataFrame]:
    """
    Splits the dataframe into parts, the split percentages are given in "percentages",
    represented as floating-point numbers in the range [0, 1].
    """
    if abs(sum(percentages) - 1) > 0.01:
        print(
            f"Warning: the sum of percentages in split should be equal to 1 (was {sum(percentages)})",
            file=sys.stderr,
        )

    result = []
    df_percentage = 1
    if random_state is None:
        random_state = random.randint(0, 2**32 - 1)
    for percentage in percentages:
        # calculate percentage as portion of the remaining rows
        true_percentage = percentage / df_percentage

        slice: pd.DataFrame = df.sample(frac=true_percentage, random_state=random_state)
        df = df.drop(slice.index.tolist()).reset_index(drop=True)
        slice = slice.reset_index(drop=True)

        result.append(slice)
        df_percentage -= percentage

    return result
