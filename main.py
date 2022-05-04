import pandas as pd
import numpy as np
from ReadData import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from trees import tree
from knn import knn
from nn import nn
from gradientboost import gboost
from Adaboost import adaboost

from sklearn.neighbors import KNeighborsClassifier


def find_best_samples(X_train, y_train, X_submit):
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)
    good_samples = clf.kneighbors(X_submit, 1)
    good_samples = np.where(good_samples[0] < 60 , good_samples[1],None)
    #return  np.array(list(filter(lambda sample: sample,good_samples)))
    return good_samples



def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def main():


    print("Read train data")
    X,y = readTrainData()
    ids, X_submit = readTestData()
    print("Finished reading train data")

    # X = NormalizeData(X)
    # best_samples = find_best_samples(X,y,X_submit)

    clf_rf = tree(X, y)
    prediction_rf_submit = clf_rf.predict_proba(X_submit)

    clf_gb = gboost(X,y)
    prediction_gb_submit = clf_gb.predict_proba(X_submit)

    # clf_ab = adaboost(X,y)
    # prediction_ab_submit = clf_ab.predict_proba(X_submit)

    # clf_nn = nn(X,y)
    # prediction_nn_submit = clf_nn.predict_proba(X_submit)

    prediction_submit = prediction_rf_submit + prediction_gb_submit

    prediction_submit = np.array([np.argmax(poss == max(poss)) for poss in prediction_submit])

    result = pd.DataFrame(np.stack((ids, prediction_submit), axis=-1).astype(int),columns=["ID","y_pred"])
    pd.DataFrame.to_csv(result,"pred.csv",columns=result.columns,index=False)

if __name__ == '__main__':
    main()