import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report, f1_score, recall_score
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

import numpy as np


def encode(df: pd.DataFrame, string_cols: [str]) -> pd.DataFrame:
    for col in string_cols:
        df[col] = df[col].astype('category').cat.codes
    return df


def extract_interesting(df: pd.DataFrame) -> pd.DataFrame:
    """
    We currently ignore the following:
    - Name (potentially the number of travels might matter)
    - Index inside the group
    """

    df = df.copy()

    # Convert truth values to regular 0/1 integers
    if 'Transported' in df.columns:
        df['Transported'] = df['Transported'].astype(int)
    df['CryoSleep'] = df['CryoSleep'].astype(int)

    # Extract group and group size from each passenger
    df['Group'] = df.apply(lambda row: row['PassengerId'][:4], axis=1)
    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')

    # Extract the three characteristics from Cabin
    df['Deck'] = df.apply(lambda row: 'Unknown'
                          if row['Cabin'] == 'Unknown' else row['Cabin'][0], axis=1)
    df['Side'] = df.apply(lambda row: 'Unknown'
                          if row['Cabin'] == 'Unknown' else row['Cabin'][4], axis=1)
    df['Number'] = df.apply(lambda row: -1
                            if row['Cabin'] == 'Unknown' else int(row['Cabin'][2]), axis=1)

    num = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['Total'] = df[num].sum(axis=1)

    return df


def split_df(df, percentage, random_state=47):
    traindf = df.sample(
        frac=percentage, random_state=random_state).reset_index(drop=True)
    verification_df = df.drop(traindf.index).reset_index(drop=True)
    return traindf, verification_df


def assessment(df):
    actual = df['Transported']
    pred = df['Predict']
    print("Accuracy:", accuracy_score(actual, pred))
    print("Precision:", precision_score(actual, pred, average='weighted'))
    print("Recall:", recall_score(actual, pred, average='weighted'))
    print("F1 Score:", f1_score(actual, pred, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(actual, pred))
    print("Classification Report:\n", classification_report(actual, pred))

# This requires some data preparation before trying


def cluster_test(df, split=True):
    if split:
        traindf, verdf = split_df(df, 0.4)
    else:
        traindf = df
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


def random_forest_test(df: pd.DataFrame, split: bool = True) -> RandomForestClassifier:
    if split:
        training_df, verification_df = split_df(df, 0.4)
    else:
        training_df = df
    model = RandomForestClassifier(
        n_estimators=50, random_state=42, n_jobs=8)
    feature_cols = [col for col in df.columns if col != 'Transported']
    model.fit(training_df[feature_cols], training_df['Transported'])
    if split:
        verification_df['Predict'] = model.predict(
            verification_df[feature_cols])
        assessment(verification_df)
    return model


def xgboost_test(df: pd.DataFrame) -> XGBClassifier:
    training_df, verification_df = split_df(df, 0.6)
    cat_cols = ['Deck', 'HomePlanet', 'Destination', 'Side', 'Group']
    for col in cat_cols:
        df[col] = df[col].astype('category')
    model = XGBClassifier(n_estimators=3000,
                          max_depth=11,
                          min_child_weight=0,
                          learning_rate=0.01,
                          objective='binary:logistic',
                          early_stopping_rounds=50,
                          gamma=0,
                          enable_categorical=True)
    feature_cols = [col for col in df.columns if col not in [
        'Transported', 'PassengerId']]
    model.fit(training_df[feature_cols],
              training_df['Transported'],
              eval_set=[(verification_df[feature_cols], verification_df['Transported'])])
    verification_df['Predict'] = model.predict(
        verification_df[feature_cols])
    assessment(verification_df)
    return model


def fill_na(df: pd.DataFrame) -> pd.DataFrame:

    # df['HomePlanet'] = df['HomePlanet'].fillna('Unknown')

    df['CryoSleep'] = df['CryoSleep'].fillna(False)

    df['Cabin'] = df['Cabin'].fillna('Unknown')

    # df['Destination'] = df['Destination'].fillna('Unknown')

    df['Age'] = df['Age'].fillna(df['Age'].median())

    df['VIP'] = df['VIP'].fillna(False)

    df['RoomService'] = df['RoomService'].fillna(df['RoomService'].median())

    df['FoodCourt'] = df['FoodCourt'].fillna(df['FoodCourt'].median())

    df['ShoppingMall'] = df['ShoppingMall'].fillna(df['ShoppingMall'].median())

    df['Spa'] = df['Spa'].fillna(df['Spa'].median())

    df['VRDeck'] = df['VRDeck'].fillna(df['VRDeck'].median())

    df['Name'] = df['Name'].fillna('Unknown')
    return df


df = extract_interesting(fill_na(pd.read_csv('train.csv')))
taskdf = extract_interesting(fill_na(pd.read_csv('test.csv')))

combined = pd.concat([df, taskdf], keys=['train', 'test'])

string_cols = ['HomePlanet', 'Deck', 'Side',
               'Destination', 'Name', 'Group', 'Cabin']
encode(combined, string_cols)
df = combined.xs('train')
taskdf = combined.xs('test').drop(columns=['Transported'])
cat_cols = ['Deck', 'HomePlanet', 'Destination', 'Side', 'Group']
for col in cat_cols:
    taskdf[col] = taskdf[col].astype('category')

model = xgboost_test(df)
feature_cols = [col for col in df.columns if col not in [
    'Transported', 'PassengerId']]

taskdf['Transported'] = model.predict(
    taskdf[feature_cols])
result = taskdf[['PassengerId', 'Transported']]
result['Transported'] = result['Transported'].astype(bool)
# print(result)
result.to_csv('result_xgb.csv', index=False)
# random_forest_test(
#    encode(tdf, ['PassengerId', 'HomePlanet', 'Cabin', 'Name', 'Destination']))
