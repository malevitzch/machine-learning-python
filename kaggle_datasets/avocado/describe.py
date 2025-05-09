import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import sys

try:
    df = pd.read_csv('avocado.csv')
except FileNotFoundError:
    print("Cannot find the dataset 'email_phishing_data.csv'")
    sys.exit(1)

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df.sort_index(inplace=True)

# the problem here is that there are multiple regions in the dataset

# df = df.asfreq('D', method='bfill')

plt.figure(figsize=(10, 8))
corr = df[['AveragePrice', 'Total Volume',
           'Total Bags', 'Large Bags', 'Small Bags']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.tight_layout()
plt.show()
