import random

import pandas as pd
import numpy as np
from utils import *
import pickle

# Create Data Loader
from sklearn.preprocessing import MinMaxScaler


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


# print('Loading expression data')
data = load_data("./data/net4_expression_data.tsv")
scaler = MinMaxScaler()
X_data = pd.DataFrame(scaler.fit_transform(data))
X_data.to_csv('./data/yeast_data_normalized.txt', header=None, sep=' ', mode='a')

# edgelist = pd.read_csv("./data/yeast_edgelist_biogrid.txt", sep=" ", header=0)
# x_train, x_test = train_test_split(edgelist, test_size=0.1)
# n = x_test.shape[0]
# # print(x_train.shape)
#
# edgeMatrix = convertSortedRankTSVToAdjMatrix(edgelist, data.shape[0])
#
# # rows = []
# # cols = []
# # for row in range(edgeMatrix.shape[0]): # df is the DataFrame
# #          for col in range(edgeMatrix.shape[1]):
# #              if edgeMatrix.get_value(row,col) ==  0 and row != col:
# #                  rows.append(row)
# #                  cols.append(col)
#
# # pickle.dump(rows, open("temp/rows.pkl", "wb"))
# # pickle.dump(cols, open("temp/cols.pkl", "wb"))
#
# rows = pickle.load( open( "temp/rows.pkl", "rb" ) )
# cols = pickle.load( open( "temp/cols.pkl", "rb" ) )
# ind = random.sample(rows, n)
#
# indexed_rows = [rows[i] for i in ind]
# indexed_cols = [cols[i] for i in ind]
# #
# X_test_0 = np.zeros((n, 3))
# X_test_0[:,0] = indexed_rows
# X_test_0[:,1] = indexed_cols
# X_test_0[:,2] = 0
# print (X_test_0.shape)
# # #
# X_test = pd.DataFrame(np.vstack((x_test, X_test_0))).astype(int)
# print (X_test.shape)
# X_test.to_csv('./data/yeast_edgelist_test.txt', header=None, sep=' ', index = False,  mode='a')
#
# #
# # (54466, 3)
# (490185, 3)