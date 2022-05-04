import pandas as pd
import numpy as np
from ReadData import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier


def gboost(X:np.ndarray, y:np.ndarray):
    print("Start gboost")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = GradientBoostingClassifier(n_estimators=200,random_state=0,learning_rate=0.2)
    clf.fit(X_train,y_train)

    prediction = clf.predict(X_test)

    print("Summery")
    print(f"score: {accuracy_score(y_test,prediction)}")

    return clf