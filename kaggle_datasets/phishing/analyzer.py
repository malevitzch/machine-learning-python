import pandas as pd
import sys

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def split_df(df, percentage, random_state=42):
    training_df = df.sample(
        frac=percentage, random_state=random_state).reset_index(drop=True)
    verification_df = df.drop(training_df.index).reset_index(drop=True)
    return training_df, verification_df


def run_assessment(df):
    correct_ver = (df['label'] ==
                   df['predict']).sum()
    all_ver = len(df)
    percent_ver = int((correct_ver / all_ver) * 100)
    print(f"Verification dataset: {correct_ver}/{all_ver}: {percent_ver}%")

    tp = ((df['predict'] == 1) & (
        df['label'] == 1)).sum()
    fp = ((df['predict'] == 1) & (
        df['label'] == 0)).sum()
    tn = ((df['predict'] == 0) & (
        df['label'] == 0)).sum()
    fn = ((df['predict'] == 0) & (
        df['label'] == 1)).sum()

    confusion_table = pd.DataFrame({
        'Predicted 0': [tn, fn],
        'Predicted 1': [fp, tp]
    }, index=['Actual 0', 'Actual 1'])

    print("Confusion Matrix (Phishing = 1):")
    print(confusion_table)
    positive_accuracy = (tp*100 // (fn + tp))
    negative_accuracy = (tn*100 // (tn + fp))

    print(f"Positive assessment: {positive_accuracy}%\n"
          f"Negative assessment: {negative_accuracy}%")


# This is very bad for the problem since the clusters are not split well
# very bad idea in general, classifies everything as nonphishing
def cluster_test(df):
    print("Running the KMeans cluster test")
    kmeans = KMeans(n_clusters=2, random_state=42, init='k-means++')

    training_df, verification_df = split_df(df, 0.8)

    kmeans.fit(training_df.iloc[:, :-1])

    training_df['predict'] = kmeans.labels_
    safe_cluster = training_df['predict'].mode()[0]
    if safe_cluster != 0:
        training_df['predict'] = training_df['predict'].map({0: 1, 1: 0})

    correct = (training_df['label'] == training_df['predict']).sum()
    all = len(training_df)

    percent = int((correct / all) * 100)
    print(f"Training dataset: {correct}/{all}: {percent}%")

    verification_df['predict'] = kmeans.predict(verification_df.iloc[:, :-1])
    if safe_cluster != 0:
        verification_df['predict'] = (
            verification_df['predict'].map({0: 1, 1: 0}))

    run_assessment(verification_df)


# This model is very bad at identifying phishing emails,
# around 8% chance for good answer on phishing
# around 92% chance for good answer on regular emails
def isolation_forest_test(df):

    print("Running the isolation forest test")
    model = IsolationForest(contamination=0.08, random_state=42)

    training_df, verification_df = split_df(df, 0.8)

    model.fit(training_df.iloc[:, :-1])
    verification_df['predict'] = model.predict(verification_df.iloc[:, :-1])
    verification_df['predict'] = verification_df['predict'].map({1: 0, -1: 1})

    correct = (verification_df['label'] == verification_df['predict']).sum()
    all = len(verification_df)

    percent = int((correct / all) * 100)
    print(f"Verification dataset: {correct}/{all}: {percent}%")

    run_assessment(verification_df)


# This one at least gives moderately interesting results, it can be made to
# predict some stuff but the imbalanced weights make it useless
def SGD_classifier_test(df):
    training_df, verification_df = split_df(df, 0.8)
    clf = make_pipeline(
        StandardScaler(),
        SGDClassifier(max_iter=100000, tol=1e-3, class_weight={0: 1, 1: 67}))
    clf.fit(training_df.iloc[:, :-1], training_df['label'])
    verification_df['predict'] = clf.predict(verification_df.iloc[:, :-1])

    run_assessment(verification_df)


try:
    df = pd.read_csv('email_phishing_data.csv')
except FileNotFoundError:
    print("Cannot find the dataset 'email_phishing_data.csv'")
    sys.exit(1)
SGD_classifier_test(df)
# isolation_forest_test(df)
