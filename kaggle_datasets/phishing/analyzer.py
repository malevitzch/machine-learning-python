import pandas as pd
import sys

from sklearn.cluster import KMeans


def cluster_test(df):
    kmeans = KMeans(n_clusters=2, random_state=42)

    kmeans.fit(df.iloc[:, :-1])

    df['cluster'] = kmeans.labels_
    safe_cluster = df['cluster'].mode()[0]
    if safe_cluster != 0:
        df['cluster'] = df['cluster'].map({0: 1, 1: 0})
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
# Rather it requires a vague idea of what the clusters represent
cluster_test(training_df)
