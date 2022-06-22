import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def compare(path1,path2):
    data1 = pd.read_csv(path1)
    data1 = data1.to_numpy()

    data2 = pd.read_csv(path2)
    data2 = data2.to_numpy()
    print(accuracy_score(data1[:,1],data2[:,1]))


def scramble_data(path, alpha=0.02):
    data = pd.read_csv(path)
    data = data.to_numpy()
    randomize_data = list()
    for sample in data:
        if sample[1]==1:
            if np.random.random() < alpha:
                new_val = 2
            else:
                new_val = sample[1]
        else:
            new_val = sample[1]
        rand_sample = [sample[0],new_val]
        randomize_data.append(rand_sample)
    randomize_data = np.array(randomize_data)
    print(accuracy_score(data[:,1],randomize_data[:,1]))
    result = pd.DataFrame(randomize_data.astype(int), columns=["ID", "y_pred"])
    pd.DataFrame.to_csv(result, "random_pred.csv", columns=result.columns, index=False)
    return None


scramble_data("best_pred.csv");
compare("pred.csv","best_pred.csv")