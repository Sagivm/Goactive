import pandas as pd
import numpy as np
import tensorflow as ts
import keras
import tensorflow.keras.utils

from ReadData import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.optimizers import SGD

def nn(X:np.ndarray, y:np.ndarray):
    print("Start ANN")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0,shuffle=True)
    y_train = tensorflow.keras.utils.to_categorical(y_train)
    clf = Sequential()
    clf.add(Input(X.shape[1]))
    clf.add(Dense(512,activation='relu'))
    clf.add(Dense(256, activation='relu'))
    clf.add(Dense(64, activation='relu'))
    clf.add(Dense(3,activation='softmax'))
    sgd = SGD(lr=0.1)
    clf.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    clf.fit(X_train, y_train, epochs=50, batch_size=32)
    prediction = clf.predict(X_test)
    prediction = np.array([np.argmax(poss == max(poss)) for poss in prediction])

    acc = accuracy_score(y_test, prediction)
    # models.append({
    #     "model": clf,
    #     "scores": confusion_matrix(y_test, prediction).diagonal() / confusion_matrix(y_test, prediction).sum(axis=1)
    # })
    # scores.append(acc)
    print("Summery")
    print(f"score: {acc}")
    #clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return clf