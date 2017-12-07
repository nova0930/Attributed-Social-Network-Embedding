import random

import pandas as pd
import numpy as np
from sklearn import preprocessing

from utils import *
import pickle
import os
# Create Data Loader


def load_data(datafile):
    """
    This function loads data set
    :param datafile:
    :return expression data:
    """
    # Load data file
    df = pd.read_csv(datafile, sep='\t', header= 0)
    df.columns = [int(x[1:])-1 for x in df.columns]
    t_data = df.T
    return(t_data)

def load_gold_standard(gold_standard):
    # Load gold standard relation file
    x = np.array([a[1:] for a in gold_standard.iloc[:,0]]).astype(int)
    y = np.array([b[1:] for b in gold_standard.iloc[:,1]]).astype(int)
    z = np.array(gold_standard.iloc[:,2])

    X = np.zeros((len(x), 3))
    for i in range(len(x)):
        X[i,0] = int(x[i]) - 1
        X[i,1] = int(y[i]) - 1
        X[i, 2] = int(z[i])
    return X


def convertdata(train_file, test_file, validation_file, test_size=0.1):
    os.remove(test_file)
    os.remove(train_file)
    os.remove(validation_file)

    data = load_data("./data/yeast/net4_expression_data.tsv")

    edgelist = pd.read_csv("./data/yeast/yeast_edgelist_biogrid.txt", sep=" ", header=None)
    adj = convertSortedRankTSVToAdjMatrix(edgelist, data.shape[0])
    adj = convertAdjMatrixToSortedRankTSV(adj)
    neg_adj = adj[adj.iloc[:,2]==0]
    print(neg_adj.shape)
    neg_adj.to_csv("output/yeast/neg_sample.txt", header=False, columns=None, index=False )

    # edgelist = edgelist.sample(frac=1).reset_index(drop=True)
    x_train, x_test = train_test_split(edgelist, test_size=test_size)
    n = x_test.shape[0]


    ind = random.sample(range(len(neg_adj)), n)

    #
    X_test_0 = neg_adj.iloc[ind,:]

    X_test = pd.DataFrame(np.vstack((x_test, X_test_0))).astype(int)

    X_test, X_validation = train_test_split(X_test, test_size=0.33)

    X_test.to_csv(test_file, header=None, sep=' ', index = False,  mode='a')

    X_validation.to_csv(validation_file, header=None, sep=' ', index = False,  mode='a')

    x_train.to_csv(train_file, header=None, sep=' ', index = False,  mode='a')

