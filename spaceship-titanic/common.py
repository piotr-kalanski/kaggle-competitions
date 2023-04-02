import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# TODO - change to sklearn pipeline
# TODO - use fit and transform
def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    def map_age(df):
        df.Age = df.Age.fillna(28) # 28 - mean of age
        df.loc[ df['Age'] <= 16, 'Age'] = 0
        df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
        df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
        df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
        df.loc[ df['Age'] > 64, 'Age'] = 4

    def map_vip(df):
        df["IsVIP"] = df["VIP"].apply(lambda x: 1 if x else 0)

    def map_cabin(df):
        df["Cabin_deck"] = df["Cabin"].apply(lambda x: x.split("/")[0] if type(x) == str else 'UNKNOWN')
        df["Cabin_num"] = df["Cabin"].apply(lambda x: x.split("/")[1] if type(x) == str else 0)
        #df["Cabin_side"] = df["Cabin"].apply(lambda x: x.split("/")[2] if type(x) == str else 'UNKNOWN')
        df["Cabin_side_is_port"] = df["Cabin"].apply(lambda x: (1 if x.split("/")[2] == 'P' else 0) if type(x) == str else 0)

    def map_numbers_to_categories(df, column: str, buckets: int):
        df[column].fillna(np.mean(df[column]))
        df[column] = pd.qcut(df[column], buckets, labels=False, duplicates="drop")

    mapped_features = [
        "Age",
        "IsVIP",
        #"Cabin_num",
        "Cabin_side_is_port",
    ]

    label_encoded_features = []

    numbers_to_categories_features = [
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
    ]

    dummy_features = [
        "HomePlanet",
        "Destination",
        "Cabin_deck",
        #"Cabin_side",
    ]

    df_copy = df.copy()
    map_age(df_copy)
    map_vip(df_copy)
    map_cabin(df_copy)
    for c in numbers_to_categories_features:
        map_numbers_to_categories(df_copy, c, 10)
    features = numbers_to_categories_features + label_encoded_features + dummy_features + mapped_features
    return pd.get_dummies(df_copy[features], columns=dummy_features)


# class Preprocessor:

#     FEATURES_TO_IMPUTE = [
#         "RoomService",
#         "FoodCourt",
#         "ShoppingMall",
#         "Spa",
#         "VRDeck"
#     ]

#     def __init__(self):
#         self._imputer = SimpleImputer(strategy="median")

#     def fit(self, train_data):
#         self._imputer.fit(train_data[self.FEATURES_TO_IMPUTE])

#     def transform(self, data):
#         data = calculate_features(data)
#         return self._imputer.transform(data)
