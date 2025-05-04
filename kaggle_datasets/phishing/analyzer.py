import pandas as pd
import sys
try:
    df = pd.read_csv('email_phishing_data.csv')
except FileNotFoundError:
    print("Cannot find the dataset 'email_phishing_data.csv'")
    sys.exit(1)

training_df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
verification_df = df.drop(training_df.index).reset_index(drop=True)


print(training_df.shape)
print(verification_df.shape)
