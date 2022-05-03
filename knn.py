import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def knn(X: np.ndarray, y: np.ndarray):
    n_negih_values = [1,2,3,5,10,20,30]
    for n_negih in n_negih_values:
        clf = KNeighborsClassifier(n_neighbors=n_negih)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        clf.fit(X_train,y_train)
        prediction = clf.predict(X_test)
        print(accuracy_score(y_test,prediction))
    return clf
