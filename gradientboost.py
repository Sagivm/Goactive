import pandas as pd
import numpy as np
from ReadData import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold,KFold
from lightgbm import LGBMClassifier
from operator import itemgetter



def gboost(X:np.ndarray, y:np.ndarray):
    print("Start gboost")
    kf = StratifiedKFold(5, shuffle=True, random_state=42)
    models = list()
    for _ in range(0,2):
        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            print(f"Iteration {i+1}")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = GradientBoostingClassifier(n_estimators=250,
                                             max_depth=10,
                                             max_features=0.2,
                                             learning_rate=0.15,
                                             random_state=42)
            clf.fit(X_train, y_train)
            prediction = clf.predict(X_test)
            acc = accuracy_score(y_test, prediction)
            models.append((clf, acc))
            print("Summery")
            print(f"score: {acc}")
    top_model = max(models, key=itemgetter(1))[0]
    return top_model



def lgboost(X:np.ndarray, y:np.ndarray):
    print("Start lgboost")
    kf = StratifiedKFold(5, shuffle=True, random_state=42)
    models = list()
    for _ in range(0,2):
        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            print(f"Iteration {i+1}")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = LGBMClassifier(n_estimators=2000,
                                 n_jobs=4,
                                 learning_rate=0.2,
                                 max_depth=20,
                                 random_state=42)

            clf.fit(X_train, y_train)
            prediction = clf.predict(X_test)
            acc = accuracy_score(y_test, prediction)
            models.append((clf, acc))
            print("Summery")
            print(f"score: {acc}")

    top_model = max(models, key=itemgetter(1))[0]
    return top_model
