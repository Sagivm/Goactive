import pandas as pd
import numpy as np
from ReadData import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest,chi2
from matplotlib import pyplot



def tree(X:np.ndarray, y:np.ndarray):
    print("Start tree")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    clf = RandomForestClassifier(n_estimators=400,
                                 max_features=0.2,
                                 n_jobs=4,
                                 random_state=42)

    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    print("Summery")
    print(f"score: {accuracy_score(y_test,prediction)}")

    return clf