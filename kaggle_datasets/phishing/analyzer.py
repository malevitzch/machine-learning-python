import pandas as pd
import sys

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# This is very bad for the problem since the clusters are not split well
# very bad idea in general, classifies everything as nonphishing


def cluster_test(df):
    print("Running the KMeans cluster test")
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


# This model is very bad at identifying phishing emails,
# 8% chance for good answer on phishing
# 92% chance for good answer on regular
def isolation_forest_test(df):

    print("Running the isolation forest test")
    model = IsolationForest(contamination=0.08, random_state=42)

    training_df = df.sample(
        frac=0.5, random_state=42).reset_index(drop=True)
    verification_df = df.drop(training_df.index).reset_index(drop=True)

    model.fit(training_df.iloc[:, :-1])
    verification_df['anomaly'] = model.predict(verification_df.iloc[:, :-1])
    verification_df['anomaly'] = verification_df['anomaly'].map({1: 0, -1: 1})

    correct = (verification_df['label'] == verification_df['anomaly']).sum()
    all = len(verification_df)

    percent = int((correct / all) * 100)
    print(f"Verification dataset: {correct}/{all}: {percent}%")

    tp = ((verification_df['anomaly'] == 1) & (
        verification_df['label'] == 1)).sum()
    fp = ((verification_df['anomaly'] == 1) & (
        verification_df['label'] == 0)).sum()
    tn = ((verification_df['anomaly'] == 0) & (
        verification_df['label'] == 0)).sum()
    fn = ((verification_df['anomaly'] == 0) & (
        verification_df['label'] == 1)).sum()

    confusion_table = pd.DataFrame({
        'Predicted 0': [tn, fn],
        'Predicted 1': [fp, tp]
    }, index=['Actual 0', 'Actual 1'])

    print("Confusion Matrix (Phishing = 1):")
    print(confusion_table)
    positive_accuracy = (tp*100 // (fn + tp))
    negative_accuracy = (tn*100 // (tn + fp))

    print(f"Positive assessment: {positive_accuracy}%\n"
          f"negative assessment: {negative_accuracy}%")


try:
    df = pd.read_csv('email_phishing_data.csv')
except FileNotFoundError:
    print("Cannot find the dataset 'email_phishing_data.csv'")
    sys.exit(1)

isolation_forest_test(df)
