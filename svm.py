import pandas as pd
import numpy as np
from ReadData import *
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import StratifiedKFold,KFold, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

from operator import itemgetter,getitem


def svm(X:np.ndarray, y:np.ndarray):
    print("Start svm")
    kf = RepeatedStratifiedKFold(n_splits=10,n_repeats=5)
    models = list()
    scores = list()
    for _ in range(0,1):
        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            print(f"Iteration {i+1}")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = LinearSVC(dual=True,max_iter=2000)
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