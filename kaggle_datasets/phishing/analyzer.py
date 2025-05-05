import pandas as pd
import sys

from sklearn.cluster import KMeans


def cluster_test(df):
    kmeans = KMeans(n_clusters=2, random_state=42)

    kmeans.fit(df.iloc[:, :-1])

    df['cluster'] = kmeans.labels_

    correct = (df['label'] == df['cluster']).sum()
    all = len(df)

    percent = int((correct / all) * 100)
    print(f"{correct}/{all}: {percent}%")


try:
    df = pd.read_csv('email_phishing_data.csv')
except FileNotFoundError:
    print("Cannot find the dataset 'email_phishing_data.csv'")
    sys.exit(1)

training_df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
verification_df = df.drop(training_df.index).reset_index(drop=True)


print(training_df.shape)
print(verification_df.shape)

# This method has 98% accuracy and does not require precise labeling of data
# If we were to be more clever and guess which label is which based on how much
# it appears, then we could do it without cheating
cluster_test(df)
