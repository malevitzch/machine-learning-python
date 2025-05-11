import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd

import sys


def show_feature_relationship(df):
    plt.figure(figsize=(10, 8))
    corr = df[['AveragePrice', 'Total Volume',
               'Total Bags', 'Large Bags', 'Small Bags']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.tight_layout()
    plt.show()


def data_graph(df, name, graph_name=None):
    if graph_name is None:
        graph_name = name
    series = df[[name]]
    print(series.head())
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=series, x=series.index, y=name, marker='o')
    plt.title(graph_name)
    plt.xlabel("Date")
    plt.ylabel(name)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45, fontsize=8)
    plt.show()


def process(df):
    df = df.iloc[:, 1:]
    df = df.asfreq('W', method='bfill')
    print(df.head())
    data_graph(df, 'AveragePrice',
               'Average Price of avocado betwene 2015 and 2018')


try:
    df = pd.read_csv('avocado.csv')
except FileNotFoundError:
    print("Cannot find the dataset 'email_phishing_data.csv'")
    sys.exit(1)

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df.sort_index(inplace=True)

con_df = df[df['type'] == 'conventional']
org_df = df[df['type'] == 'organic']

con_total_df = con_df[con_df['region'] == 'TotalUS']

process(con_total_df)
