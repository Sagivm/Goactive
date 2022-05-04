import pandas as pd
import numpy as np
from ReadData import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest,chi2
from matplotlib import pyplot



def tree(X:np.ndarray, y:np.ndarray,best_samples: np.ndarray):
    print("Start tree")
    X_train, X_test, y_train, y_test, best_train, best_test = train_test_split(X, y, best_samples, test_size=0.3, random_state=2)
    weights = np.array(list(map(lambda n: 8 if n else 1, best_train)))
    clf = RandomForestClassifier()

    clf.fit(X_train, y_train,sample_weight=weights)
    prediction = clf.predict(X_test)

    print("Summery")
    print(f"score: {accuracy_score(y_test,prediction)}")

    return clf