import pandas as pd
import numpy as np
from ReadData import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

def nn(X:np.ndarray, y:np.ndarray):
    print("Start ")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = MLPClassifier(alpha=1e-5,
                        hidden_layer_sizes=(20,10), random_state=1)
    clf.fit(X_train,y_train)

    prediction = clf.predict(X_test)

    print("Summery")
    print(f"score: {accuracy_score(y_test,prediction)}")

    return clf