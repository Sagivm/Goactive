import pandas as pd
import numpy as np
from ReadData import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot

from operator import itemgetter


def tree(X:np.ndarray, y:np.ndarray):
    print("Start tree")
    kf = StratifiedKFold(5,shuffle=True,random_state=42)
    models = list()
    for _ in range(0,2):
        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            print(f"Iteration {i+1}")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = RandomForestClassifier(n_estimators=400,
                                         max_features=0.2,
                                         n_jobs=4,
                                         random_state=42
                                         )



            clf.fit(X_train, y_train)
            prediction = clf.predict(X_test)
            acc = accuracy_score(y_test,prediction)
            models.append((clf,acc))
            print("Summery")
            print(f"score: {acc}")
    top_model = max(models, key=itemgetter(1))[0]
    return top_model