import matplotlib.pyplot as plt
import pandas as pd
import sys


def index_by_dates(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)


# Read data and split it into conventional and organic avocado
# cause as far as I'm concerned they are completely different products
try:
    df = pd.read_csv('avocado.csv')
except FileNotFoundError:
    print("Cannot find the dataset 'email_phishing_data.csv'")
    sys.exit(1)

index_by_dates(df)

con_df = df[df['type'] == 'conventional']
org_df = df[df['type'] == 'organic']

totalUS_con = con_df[con_df['region'] == 'TotalUS']
