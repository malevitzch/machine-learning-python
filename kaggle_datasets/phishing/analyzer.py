import pandas as pd
import sys
try:
    data = pd.read_csv('email_phishing_data.csv')
except FileNotFoundError:
    print("Cannot find the dataset 'email_phishing_data.csv'")
    sys.exit(1)

print(data.shape)
