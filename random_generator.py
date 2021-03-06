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

def scramble_data(path, alpha=0.02):
    data = pd.read_csv(path)
    data = data.to_numpy()
    randomize_data = list()
    for sample in data:
        if np.random.random() < alpha:
                new_val = np.random.random(3)
        else:
            new_val = sample[1]
        rand_sample = [sample[0],new_val]
        randomize_data.append(rand_sample)
    randomize_data = np.array(randomize_data)
    print(accuracy_score(data[:,1],randomize_data[:,1]))
    result = pd.DataFrame(randomize_data.astype(int), columns=["ID", "y_pred"])
    pd.DataFrame.to_csv(result, "random_pred.csv", columns=result.columns, index=False)
    return None

