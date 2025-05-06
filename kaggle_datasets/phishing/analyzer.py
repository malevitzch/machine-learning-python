import pandas as pd
import sys

from sklearn.cluster import KMeans

# This is very bad for the problem since the clusters are not exclusive
# very bad idea


def cluster_test(df):
    kmeans = KMeans(n_clusters=2, random_state=42, init='k-means++')

    training_df = df.sample(
        frac=0.8, random_state=42).reset_index(drop=True)
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

    tp = ((verification_df['cluster'] == 0) & (
        verification_df['label'] == 0)).sum()
    fp = ((verification_df['cluster'] == 0) & (
        verification_df['label'] == 1)).sum()
    tn = ((verification_df['cluster'] == 1) & (
        verification_df['label'] == 1)).sum()
    fn = ((verification_df['cluster'] == 1) & (
        verification_df['label'] == 0)).sum()

    confusion_table = pd.DataFrame({
        'Predicted 0': [tp, fp],
        'Predicted 1': [fn, tn]
    }, index=['Actual 0', 'Actual 1'])

    print("Confusion Matrix:")
    print(confusion_table)


try:
    df = pd.read_csv('email_phishing_data.csv')
except FileNotFoundError:
    print("Cannot find the dataset 'email_phishing_data.csv'")
    sys.exit(1)

cluster_test(df)
