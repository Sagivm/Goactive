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
    print()
    ccp_alpha_values = np.array(range(0,10))/10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    tree_depth_values = [i for i in range(1,100)]
    train_scores = list()
    test_scores = list()
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    train_yhat = clf.predict(X_train)
    train_acc = accuracy_score(y_train, train_yhat)
    train_scores.append(train_acc)
    # evaluate on the test dataset
    #
    test_yhat = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_yhat)
    test_scores.append(test_acc)

    return clf