import pandas as pd
import numpy as np
import tensorflow
import shap
from auc_graph import plot_auc
import random_generator
from ReadData import *
from trees import tree
from gradientboost import gboost
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def NormalizeData(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

def main():
    print("Read train data")
    X,y = readTrainData()
    ids, X_submit = readTestData()
    print("Finished reading train data")

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
    print("Add mutated data")

    X_mutations,y_mutations,length = random_generator.combinator(X,
                                                                 X_submit,
                                                                 y, alpha=0.3)
    X = np.row_stack((X,X_mutations))
    y = np.hstack((y,y_mutations))

    print("Train models")
    n_repeats = 3

    clf_rf, rel_rf_acc = tree(X, y, n_repeats)
    prediction_rf_submit = clf_rf.predict_proba(X_submit)

    clf_gb, rel_gb_acc = gboost(X,y,n_repeats-1)
    prediction_gb_submit = clf_gb.predict_proba(X_submit)

    best_pred = pd.read_csv("best_pred.csv")
    best_pred = best_pred.to_numpy()[:, 1]
    best_pred = tensorflow.keras.utils.to_categorical(best_pred)


    print("Evaluate")
    prediction_submit = 0.25 * best_pred + 0.35*prediction_rf_submit + 0.40* prediction_gb_submit
    prediction_n_submit = np.array([np.argmax(poss == max(poss)) for poss in prediction_submit])

    result = pd.DataFrame(np.stack((ids, prediction_n_submit), axis=-1).astype(int),columns=["ID","y_pred"])
    pd.DataFrame.to_csv(result,f"pred.csv",columns=result.columns,index=False)

    # # Create object that can calculate shap values
    explainer = shap.TreeExplainer(clf_rf)

    plt.figure()
    # Calculate Shap values
    #shap_values = explainer(X_submit)

    shap_values = explainer.shap_values(X_submit[:2000,:])
    shap.summary_plot(shap_values, max_display=5)
    shap.dependence_plot(1137, shap_values, X)
    shap.dependence_plot(1136, shap_values, X)
    shap.dependence_plot(1103, shap_values, X)
    shap.dependence_plot(378, shap_values, X)
    shap.dependence_plot(1144, shap_values, X)

    # shap.initjs()
    # shap.force_plot(explainer.expected_value[1], shap_values.values[1], X_submit,show=True)
    # shap.plots.force(explainer.expected_value[1], shap_values.values[1], X_submit[:200,:])
    plt.plot()
    plt.savefig('tmp.svg')
    plt.close()

   #  X_train, y_train = readTrainData()
   #  X_train =X_train[:2000,:]
   #  y_train = tensorflow.keras.utils.to_categorical(y_train)[:2000,:]
   #  rf_train_prediction = clf_rf.predict_proba(X_train)
   #  # gb_train_prediction = clf_gb.predict_proba(X_train)
   # # prediction = 0.25 * y_train + 0.35 * rf_train_prediction + 0.40 * gb_train_prediction
   #  prediction = 0.25 * y_train + 0.35 * rf_train_prediction
   #  print("plot")
   #  plot_auc(X_train,y_train)


if __name__ == '__main__':
    main()

