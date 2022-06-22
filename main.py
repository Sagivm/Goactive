import pandas as pd
import numpy as np
import tensorflow
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
    #
    # X transformation
    print("X transformation")
    X_full = np.vstack((X, X_submit))
    X_full = NormalizeData(X_full)
    X_full = X_full + 1
    columns = X.shape[1]

    for i in range(columns):
        np.append(X_full,np.log(X_full[:,i]))

    X = X_full[:15000,:]
    X_submit = X_full[15000:, :]
    print("X transformation finished")

    #
    # clf_svm, rel_svm_acc = svm(X, y)
    # prediction_svm_submit = clf_svm.predict_proba(X_submit)

    clf_rf, rel_rf_acc = tree(X, y)
    prediction_rf_submit = clf_rf.predict_proba(X_submit)
    #
    clf_gb, rel_gb_acc = gboost(X,y)
    prediction_gb_submit = clf_gb.predict_proba(X_submit)
    #
    # clf_lgb = lgboost(X, y)
    # prediction_lgb_submit = clf_lgb.predict_proba(X_submit)

    # clf_ab = adaboost(X,y)
    # prediction_ab_submit = clf_ab.predict_proba(X_submit)

    clf_nn = nn(X,y)
    prediction_nn_submit = clf_nn.predict(X_submit)

    # prediction_submit = prediction_rf_submit * 0.3 + \
    #                     prediction_gb_submit * 0.7
    #
    best_pred = pd.read_csv("best_pred.csv")
    best_pred = best_pred.to_numpy()[:, 1]
    best_pred = tensorflow.keras.utils.to_categorical(best_pred)

    prediction_submit = 0.3 * best_pred + 0.35*prediction_rf_submit +0.35* prediction_gb_submit
    prediction_submit = np.array([np.argmax(poss == max(poss)) for poss in prediction_submit])

    result = pd.DataFrame(np.stack((ids, prediction_submit), axis=-1).astype(int),columns=["ID","y_pred"])
    pd.DataFrame.to_csv(result,"pred.csv",columns=result.columns,index=False)

if __name__ == '__main__':
    main()

