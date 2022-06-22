import pandas as pd
import numpy as np
from ReadData import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import StratifiedKFold,KFold, RepeatedStratifiedKFold,RepeatedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

from operator import itemgetter,getitem


def tree(X:np.ndarray, y:np.ndarray):
    print("Start tree")
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
    models = list()
    scores = list()
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        print(f"Iteration {i+1}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = ExtraTreesClassifier(n_estimators=300,
                                   max_depth=16,
                                   max_features=0.2,
                                   n_jobs=6,
                                   random_state=42)

        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        acc = accuracy_score(y_test,prediction)
        models.append({
            "model": clf,
            "scores": confusion_matrix(y_test,prediction).diagonal()/confusion_matrix(y_test,prediction).sum(axis=1)
        })
        scores.append(acc)
        print("Summery")
        print(f"score: {acc}")
    top_model_index = scores.index(max(scores))
    return list(models[top_model_index].values())
