import pandas as pd
import numpy as np
from ReadData import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold,KFold,RepeatedStratifiedKFold,RepeatedKFold
#from lightgbm import LGBMClassifier
from operator import itemgetter



def gboost(X:np.ndarray, y:np.ndarray,n:int):
    print("Start gboost")
    kf = RepeatedStratifiedKFold(n_splits=n, n_repeats=10)
    models = list()
    scores = list()
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        print(f"Iteration {i+1}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = GradientBoostingClassifier(n_estimators=200,
                                         max_depth=6,
                                         max_features=0.3,
                                         learning_rate=0.1,
                                         random_state=42)
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        acc = accuracy_score(y_test, prediction)
        models.append({
            "model": clf,
            "scores": confusion_matrix(y_test, prediction).diagonal() / confusion_matrix(y_test, prediction).sum(
                axis=1)
        })
        scores.append(acc)
        # models.append({
        #     "model": clf,
          #     "acc": acc,
        #     "scores":
        #               })
        #             print("Summery")
        print(f"score: {acc}")
    top_model_index = scores.index(max(scores))
    return list(models[top_model_index].values())
