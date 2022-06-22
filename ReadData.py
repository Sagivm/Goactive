import numpy as np
import pandas as pd
import os
#import matplotlib.pyplot as plt
#import pylab
from scipy import stats


def readTrainData():
    data = pd.read_csv(os.path.join('Data', 'X_y_train.csv'))
    data = data.to_numpy()
    data_attrib = data[:,:-1]
    data_target = data[:, -1]
    return data_attrib, data_target

def readTestData():
    data = pd.read_csv(os.path.join('Data', 'X_test.csv'))
    data = data.to_numpy()
    data_ids = data[:,0]
    data_attrib = data[:,1:]
    return data_ids, data_attrib


def readsubExample():
    data = pd.read_csv(os.path.join('Data', 'y_test_submission_example.csv'))
    data = data.to_numpy()
    data_ids = data[:,0]
    data_target= data[:,1:]
    return data_ids, data_target

