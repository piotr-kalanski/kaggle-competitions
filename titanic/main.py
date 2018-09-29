import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from scipy.stats import mode
from sklearn.pipeline import make_pipeline, make_union


def read_data():
    return pd.read_csv("train.csv")


def split_train_test(X, y):
    return train_test_split(X, y, test_size=0.25)


def calculate_features(df):
    #setting silly values to nan
    df.Fare = df.Fare.map(lambda x: np.nan if x==0 else x)

    df.Fare = df.Fare.fillna(np.mean(df.Fare))

    df['Categorical_Fare'] = pd.qcut(df['Fare'], 4)

    df['Has_Cabin'] = df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    #Special case for cabins as nan may be signal
    df.Cabin = df.Cabin.fillna('Unknown')

    df['Embarked'] = df['Embarked'].fillna('S')

    def substrings_in_string(big_string, substrings):
        if big_string is not np.nan:
            for substring in substrings:
                if str.find(big_string, substring) != -1:
                    return substring
        return np.nan

    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                'Don', 'Jonkheer']

    df['Title'] = df['Name'].map(lambda x: substrings_in_string(x, title_list))

    def replace_titles(x):
        title = x['Title']
        if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Countess', 'Mme']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title =='Dr':
            if x['Sex']=='Male':
                return 'Mr'
            else:
                return 'Mrs'
        else:
            return title

    df['Title'] = df.apply(replace_titles, axis=1)

    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    df['Deck'] = df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1

    df['Is_Alone'] = 0
    df.loc[df['Family_Size'] == 1, 'Is_Alone'] = 1

    df.Age = df.Age.fillna(np.mean(df.Age))

    df['Categorical_Age'] = pd.qcut(df['Age'], 5)

    df['Age*Class'] = df['Age']*df['Pclass']

    df['Fare_Per_Person'] = df['Fare']/(df['Family_Size']+1)

    df['Name_length'] = df['Name'].apply(len)

    # Mapping Fare
    df.loc[ df['Fare'] <= 7.91, 'Fare'] 						        = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
    df.loc[ df['Fare'] > 31, 'Fare'] 							        = 3
    df['Fare'] = df['Fare'].astype(int)

    # Mapping Age
    df.loc[ df['Age'] <= 16, 'Age'] 					       = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[ df['Age'] > 64, 'Age'] = 4

    # result = pd.concat([
    #     df[["Pclass", "SibSp", "Parch"]],
    #     pd.get_dummies(df.Sex, "Sex"),
    #     pd.get_dummies(df.Embarked, "Embarked")
    # ], axis=1)
    # Fare, Age
    #result["Fare"][result.Fare.isnull()] = np.mean(result.Fare)

    result = pd.concat([
        #pd.get_dummies(df.Categorical_Fare, "Categorical_Fare"),
        #pd.get_dummies(df.Categorical_Age, "Categorical_Age"),
        pd.get_dummies(df.Title, 'Title'),
        #pd.get_dummies(df.Deck, 'Deck'),
        df[[
            'Age',
            'Fare',
            'Family_Size',
            #'Age*Class',
        #    'Fare_Per_Person',
        #    "SibSp",
            "Parch",
            'Is_Alone',
            'Has_Cabin',
            'Name_length'
        ]],
        pd.get_dummies(df.Sex, "Sex"),
        pd.get_dummies(df.Embarked, "Embarked"),
        pd.get_dummies(df.Pclass, "Pclass")
    ], axis=1)

    return result


def train_model(X_train, y_train):
    #model = LogisticRegression()
    #model = RandomForestClassifier()
    #model = ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.9000000000000001, min_samples_leaf=4, min_samples_split=15)
    #model = GradientBoostingClassifier(max_depth=2, max_features=0.25, min_samples_leaf=13, min_samples_split=15, n_estimators=100, subsample=0.4)
    #model = DecisionTreeClassifier(max_depth=7, min_samples_leaf=16, min_samples_split=10)
    #model = RandomForestClassifier(bootstrap=False, max_features=0.6500000000000001, min_samples_leaf=9, min_samples_split=19)

    model = VotingClassifier(estimators=[
        ('lr', LogisticRegression()),
        ('et', ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.9000000000000001, min_samples_leaf=4, min_samples_split=15)),
        ('gb', GradientBoostingClassifier(max_depth=2, max_features=0.25, min_samples_leaf=13, min_samples_split=15, n_estimators=100, subsample=0.4)),
        ('dt', DecisionTreeClassifier(max_depth=7, min_samples_leaf=16, min_samples_split=10)),
        ('rf', RandomForestClassifier(bootstrap=False, max_features=0.6500000000000001, min_samples_leaf=9, min_samples_split=19)),
        ('gb2', GradientBoostingClassifier(learning_rate=0.01, max_depth=2, max_features=0.8, min_samples_leaf=11, min_samples_split=10, subsample=0.7000000000000001)),
        ('gb3', GradientBoostingClassifier(max_depth=7, max_features=0.15000000000000002, min_samples_leaf=5, min_samples_split=17, n_estimators=100, subsample=0.6500000000000001)),
        ('pip1', make_pipeline(
            StackingEstimator(estimator=LinearSVC(dual=False, loss="squared_hinge", tol=1e-05)),
            StandardScaler(),
            DecisionTreeClassifier(criterion="entropy", max_depth=6, min_samples_leaf=7, min_samples_split=9)
        ))
        #('rf2', RandomForestClassifier(criterion="entropy", max_features=0.25, min_samples_split=8, n_estimators=100))
    ], voting='hard')

    model.fit(X_train, y_train)
    return model


def calculate_model_quality(model, test_data, test_labels):
    y_true = test_labels
    y_pred = model.predict(test_data)
    return {
        "accuracy": accuracy_score(y_true, y_pred)
    }


def calculate_predictions(model):
    test_raw = pd.read_csv("test.csv")
    test = calculate_features(test_raw)
    #test['Deck_T'] = 0 # TODO - powinno byc obslugiwane w feature engineering
    pred = model.predict(test)
    test_raw["Survived"] = pred
    result = test_raw[["PassengerId", "Survived"]]
    result.to_csv("titanic_submission.csv", index=False)


def auto_ml(X_train, X_test, y_train, y_test):
    tpot = TPOTClassifier(generations=30, population_size=200, verbosity=2)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_pipeline.py')


if __name__ == 'Main':
    train_raw = read_data()
    train = calculate_features(train_raw)
    X_train, X_test, y_train, y_test = split_train_test(train, train_raw["Survived"])
    model = train_model(X_train, y_train)
    measures = calculate_model_quality(model, X_test, y_test)
    for k, m in measures.items():
        print(k)
        print(m)
    calculate_predictions(model)
