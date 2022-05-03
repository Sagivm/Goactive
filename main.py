import pandas as pd
import numpy as np
from ReadData import *
from sklearn.tree import DecisionTreeClassifier
from trees import tree
from knn import knn
from nn import nn
from gradientboost import gboost
def main():


    print("Read train data")
    X,y = readTrainData()
    ids, X_submit = readTestData()
    print("Finished reading train data")
    clf_rf = tree(X,y)
    #clf_gb = gboost(X,y)
    #clf_nn = nn(X,y)

    #prediction_rf_submit = clf_rf.predict_proba(X_submit)
    prediction_submit = clf_rf.predict_proba(X_submit)
    #prediction_gb_submit = clf_gb.predict_proba(X_submit)
    #prediction_nn_submit = clf_nn.predict_proba(X_submit)
    #prediction_submit = prediction_rf_submit + prediction_nn_submit + prediction_gb_submit

    prediction_submit = np.array([np.argmax(poss == max(poss)) for poss in prediction_submit])

    result = pd.DataFrame(np.stack((ids, prediction_submit), axis=-1).astype(int),columns=["ID","y_pred"])
    pd.DataFrame.to_csv(result,"pred.csv",columns=result.columns,index=False)

if __name__ == '__main__':
    main()