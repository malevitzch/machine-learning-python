import pandas as pd
from sklearn.cluster import KMeans


def split_df(df, percentage, random_state=42):
    traindf = df.sample(
        frac=percentage, random_state=random_state).reset_index(drop=True)
    verification_df = df.drop(traindf.index).reset_index(drop=True)
    return traindf, verification_df


# Clustering doesn't work too well with strings, at least
# the standard algorithms don't


def cluster_test(df):
    traindf, verdf = split_df(df, 0.4)
    kmeans = KMeans(n_clusters=2, random_state=42, init='k-means++')
    kmeans.fit(traindf.drop(
        columns=['Transported', 'HomePlanet', 'Cabin', 'Destination', 'Name']))
    traindf['predict'] = kmeans.labels_
    safe_cluster = traindf['predict'].mode()[0]
    if safe_cluster != 0:
        traindf['predict'] = traindf['predict'].map({0: 1, 1: 0})

    correct = (traindf['Transported'] == traindf['predict']).sum()
    all = len(traindf)

    percent = int((correct / all) * 100)
    print(f"Training dataset: {correct}/{all}: {percent}%")


tdf = pd.read_csv('train.csv').dropna()
cluster_test(tdf)
