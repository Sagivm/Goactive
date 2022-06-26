import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def find_best_samples(X_train, X_submit,y_train):
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)
    good_samples = clf.kneighbors(X_submit, 1)
    good_samples = np.where(good_samples[0] < 5 , good_samples[1],None)
    #return  np.array(list(filter(lambda sample: sample,good_samples)))
    return good_samples

def combinator(X,X_submit,y_train,alpha):
    best_samples_indexs = find_best_samples(X,X_submit,y_train)
    X_mutations = list()
    y_mutations = list()
    for i,sample in enumerate(best_samples_indexs):
        if sample[0] is not None:
            X_mutations.append(X_submit[i] * alpha + X[i]*(1-alpha))
            y_mutations.append(y_train[i])
    mutations = np.array(X_mutations)
    return mutations,y_mutations,mutations.shape[0]


def compare(path1,path2,alpha=0.2):
    data1 = pd.read_csv(path1)
    data1 = data1.to_numpy()

    data2 = pd.read_csv(path2)
    data2 = data2.to_numpy()

    combined_data = list()
    for i in range(data1.shape[0]):
        if (data1[i,1] != data2[i,1]):
            if(data1[i,1]==1 and np.random.random()<alpha):
                new_val= data2[i,1]
            else:
                new_val = data1[i, 1]
        else:
            new_val=data1[i,1]
        combined_data.append([data1[i,0],new_val])

    combined_data = np.row_stack(combined_data)
    print("Base " +str(accuracy_score(data1[:,1],data2[:,1])))
    print("Changed " + str(accuracy_score(data1[:, 1], combined_data[:, 1])))
    result = pd.DataFrame(combined_data.astype(int), columns=["ID", "y_pred"])
    pd.DataFrame.to_csv(result, "com_pred.csv", columns=result.columns, index=False)


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


# scramble_data("best_pred.csv");
# compare("pred35.csv","best_pred.csv")