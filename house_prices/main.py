import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
import math
import seaborn as sns
import matplotlib.pyplot as plt
from tpot import TPOTRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator


def read_data():
    return pd.read_csv("train.csv")


def split_train_test(X, y):
    return train_test_split(X, y, test_size=0.25, random_state=0)


def dataset_intitial_analysis(df):
    df.info()


def detailed_analysis(df):
    sns.pairplot(df, diag_kind='kde')


def clean_dataset(df):
    categorical_features_for_median_imput = [
        'Electrical',
        'KitchenQual'
    ]

    for f in categorical_features_for_median_imput:
        df[f].fillna(df[f].mode()[0], inplace=True)

    return df


def calculate_features(df):
    raw_features = [
        "LotArea",
        "OverallQual",
        "OverallCond",
        "YearBuilt",
        "YrSold",
        "MoSold"
    ]

    features_for_label_encoding = [
        "HeatingQC",
        "Electrical",
        "KitchenQual"
    ]
    label_encoded_features = []
    for f in features_for_label_encoding:
        f_code = f + '_code'
        df[f_code] = LabelEncoder().fit_transform(df[f])
        label_encoded_features.append(f_code)

    dummy_features = [
        "CentralAir"
    ]

    features = raw_features + label_encoded_features + dummy_features

    return pd.get_dummies(df[features])


def train_model(X_train, y_train):
    #model = linear_model.LinearRegression()
    model = make_pipeline(
        StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.45, tol=0.001)),
        StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.9, learning_rate=1.0, loss="quantile", max_depth=8, max_features=0.15000000000000002, min_samples_leaf=19, min_samples_split=12, n_estimators=100, subsample=0.15000000000000002)),
        VarianceThreshold(threshold=0.0005),
        StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.8, learning_rate=0.5, loss="lad", max_depth=1, max_features=0.8, min_samples_leaf=19, min_samples_split=14, n_estimators=100, subsample=0.8500000000000001)),
        PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
        LassoLarsCV(normalize=True)
    )
    model.fit(X_train, y_train)
    return model


def calculate_model_quality(model, test_data, test_labels):
    y_true = test_labels
    y_pred = model.predict(test_data)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": math.sqrt(mse),
        "mean_squared_error": mse,
        "r2_score": r2_score(y_true, y_pred)
    }


def calculate_predictions(model):
    test_raw = pd.read_csv("test.csv")
    test_cleaned = clean_dataset(test_raw)
    test = calculate_features(test_cleaned)
    pred = model.predict(test)
    test_raw["SalePrice"] = pred
    result = test_raw[["Id", "SalePrice"]]
    result.to_csv("submission.csv", index=False)


def plot_prediction_results(model, X_test, y_test):
    y_pred = model.predict(X_test)
    #sns.scatterplot(x=y_test, y=y_pred)
    max_val = 800000
    plt.plot([0, max_val], [0, max_val], marker='o')
    #plt.hist2d(x=y_test, y=y_pred, bins=100, cmap=plt.cm.jet)
    #plt.colorbar()
    plt.scatter(x=y_test, y=y_pred, alpha=0.5)
    axes = plt.gca()
    axes.set_xlim([0, max_val])
    axes.set_ylim([0, max_val])
    plt.show()


def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    _ = sns.heatmap(
        df.corr(),
        cmap = colormap,
        square=True,
        cbar_kws={'shrink':.9 },
        ax=ax,
        annot=True,
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )

    plt.title('Pearson Correlation of Features', y=1.05, size=15)


def auto_ml(X_train, X_test, y_train, y_test):
    tpot = TPOTRegressor(generations=30, population_size=200, verbosity=2, periodic_checkpoint_folder="tpot_checkpoint/")
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_pipeline.py')


if __name__ == 'Main':
    train_raw = read_data()
    #dataset_intitial_analysis(train_raw)
    train_cleaned = clean_dataset(train_raw)
    train = calculate_features(train_cleaned)
    X_train, X_test, y_train, y_test = split_train_test(train, train_raw["SalePrice"])
    #detailed_analysis(pd.concat([X_train,y_train], axis=1))
    model = train_model(X_train, y_train)
    measures = calculate_model_quality(model, X_test, y_test)
    for k, m in measures.items():
        print(k + (30 - len(k))*' ' + " : " + str(m))
    plot_prediction_results(model, X_test, y_test)
    calculate_predictions(model)
