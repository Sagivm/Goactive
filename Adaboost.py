import pandas as pd
import numpy as np
from ReadData import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier


def adaboost(X:np.ndarray, y:np.ndarray):
    print("Start adaboost")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = AdaBoostClassifier(n_estimators=100,random_state=0)
    clf.fit(X_train,y_train)

    prediction = clf.predict(X_test)

    print("Summery")
    print(f"score: {accuracy_score(y_test,prediction)}")

    return clf