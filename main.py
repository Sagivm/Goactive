import pandas as pd
import numpy as np
import tensorflow

import random_generator
from ReadData import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from trees import tree
from knn import knn
from nn import nn
from svm import svm
from gradientboost import gboost
from Adaboost import adaboost
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler



def NormalizeData(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

def main():


    print("Read train data")
    X,y = readTrainData()
    ids, X_submit = readTestData()
    print("Finished reading train data")
    #
    # X transformation

    ##
    print("Add mutated data")

    ##
    print("X transformation")

    X_full = np.vstack((X, X_submit))
    X_full= NormalizeData(X_full)
    X_full = X_full + 1
    columns = X.shape[1]

    for i in range(columns):
        np.append(X_full,np.log(X_full[:,i]))
    X = X_full[:15000,:]
    X_submit = X_full[15000:, :]
    print("X transformation finished")
    X_mutations,y_mutations,length = random_generator.combinator(X,
                                                                 X_submit,
                                                                 y, alpha=0.4)
    X = np.row_stack((X,X_mutations))
    y = np.hstack((y,y_mutations))

    n_repeats = 3
    # clf_svm, rel_svm_acc = svm(X, y)
    # prediction_svm_submit = clf_svm.predict_proba(X_submit)
    # clf_nn = nn(X, y)
    # prediction_nn_submit = clf_nn.predict(X_submit)

    clf_rf, rel_rf_acc = tree(X, y, n_repeats)
    prediction_rf_submit = clf_rf.predict_proba(X_submit)
    #
    clf_gb, rel_gb_acc = gboost(X,y,n_repeats)
    prediction_gb_submit = clf_gb.predict_proba(X_submit)
    # #
    # clf_nn = nn(X,y)
    # prediction_nn_submit = clf_nn.predict(X_submit)
    best_pred = pd.read_csv("best_pred.csv")
    best_pred = best_pred.to_numpy()[:, 1]
    best_pred = tensorflow.keras.utils.to_categorical(best_pred)

    prediction_submit = 0.30 * best_pred + 0.35*prediction_rf_submit +0.35* prediction_gb_submit
    #prediction_submit = 0.4 * best_pred + 0.6 * prediction_rf_submit
    prediction_n_submit = np.array([np.argmax(poss == max(poss)) for poss in prediction_submit])

    result = pd.DataFrame(np.stack((ids, prediction_n_submit), axis=-1).astype(int),columns=["ID","y_pred"])
    pd.DataFrame.to_csv(result,f"pred--30.csv",columns=result.columns,index=False)



if __name__ == '__main__':
    main()

