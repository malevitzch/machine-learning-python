import matplotlib.pyplot as plt
import pandas as pd


# Read data and split it into conventional and organic avocado
# cause as far as I'm concerned they are completely different products
df = pd.read_csv("avocado.csv")

con_df = df[df['type'] == 'conventional']
org_df = df[df['type'] == 'organic']
