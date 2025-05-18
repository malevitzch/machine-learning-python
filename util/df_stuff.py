import pandas as pd


def split_df(df, percentage, random_state=42):
    """
    Splits a dataframe into training and verification
    dataframes based on given percentage and potentially a random state
    """

    training_df = df.sample(
        frac=percentage, random_state=random_state).reset_index(drop=True)
    verification_df = df.drop(training_df.index).reset_index(drop=True)
    return training_df, verification_df
