import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
import math
import seaborn as sns
import matplotlib.pyplot as plt


def read_data():
    return pd.read_csv("train.csv")


def split_train_test(X, y):
    return train_test_split(X, y, test_size=0.25, random_state=0)


def dataset_intitial_analysis(df):
    df.info()

    for c in df.columns:
        sns.distplot(df[c])


def calculate_features(df):
    return df[[
        "LotArea",
        "OverallQual",
        "OverallCond",
        "YearBuilt",
        "YrSold",
        "MoSold"
    ]]


def train_model(X_train, y_train):
    model = linear_model.LinearRegression()
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
    test = calculate_features(test_raw)
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


if __name__ == 'Main':
    train_raw = read_data()
    dataset_intitial_analysis(train_raw)
    train = calculate_features(train_raw)
    X_train, X_test, y_train, y_test = split_train_test(train, train_raw["SalePrice"])
    model = train_model(X_train, y_train)
    measures = calculate_model_quality(model, X_test, y_test)
    for k, m in measures.items():
        print(k + (30 - len(k))*' ' + " : " + str(m))
    plot_prediction_results(model, X_test, y_test)
    calculate_predictions(model)
