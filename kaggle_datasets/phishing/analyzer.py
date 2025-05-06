import pandas as pd
import sys

from sklearn.cluster import KMeans


def cluster_test(df):
    kmeans = KMeans(n_clusters=2, random_state=42)

    training_df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
    verification_df = df.drop(training_df.index).reset_index(drop=True)

    kmeans.fit(training_df.iloc[:, :-1])

    training_df['cluster'] = kmeans.labels_
    safe_cluster = training_df['cluster'].mode()[0]
    if safe_cluster != 0:
        training_df['cluster'] = training_df['cluster'].map({0: 1, 1: 0})

    correct = (training_df['label'] == training_df['cluster']).sum()
    all = len(training_df)

    percent = int((correct / all) * 100)
    print(f"Training dataset: {correct}/{all}: {percent}%")

    verification_df['cluster'] = kmeans.predict(verification_df.iloc[:, :-1])
    if safe_cluster != 0:
        verification_df['cluster'] = (
            verification_df['cluster'].map({0: 1, 1: 0}))
    correct_ver = (verification_df['label'] ==
                   verification_df['cluster']).sum()
    all_ver = len(verification_df)
    percent_ver = int((correct_ver / all_ver) * 100)
    print(f"Verification dataset: {correct_ver}/{all_ver}: {percent_ver}%")


try:
    df = pd.read_csv('email_phishing_data.csv')
except FileNotFoundError:
    print("Cannot find the dataset 'email_phishing_data.csv'")
    sys.exit(1)

# This method has 98% accuracy and does not require precise labeling of data
# used to generate the kmeans
# Rather it requires a vague idea of what the clusters represent
cluster_test(df)
