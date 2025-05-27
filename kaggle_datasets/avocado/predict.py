import matplotlib.pyplot as plt
import pandas as pd
import sys


def read_df_from_csv(name):
    try:
        df = pd.read_csv(name)
    except FileNotFoundError:
        print(f'Cannot find the dataset \'{name}\'')
        sys.exit(1)
    return df


def index_by_dates(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)


# Read data and split it into conventional and organic avocado
# cause as far as I'm concerned they are completely different products

df = read_df_from_csv('avocado.csv')

index_by_dates(df)

con_df = df[df['type'] == 'conventional']
org_df = df[df['type'] == 'organic']

totalUS_con = con_df[con_df['region'] == 'TotalUS']
