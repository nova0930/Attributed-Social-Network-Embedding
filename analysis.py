import random
import pickle

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

import evaluation
from utils import *


def adjToList(adj):
    import networkx as nx
    G = nx.from_numpy_matrix(adj)
    return nx.to_edgelist(G)

embeddings = pd.read_csv("./output/yeast/embeddings.txt", header=0, index_col=0)

sim = cosine_similarity(embeddings)

edgelist= pd.read_csv("./data/yeast/yeast_edgelist_biogrid.txt", sep=" ", header=None)
print(edgelist.shape)
n = edgelist.shape[0]

rows = pickle.load( open( "temp/yeast/rows.pkl", "rb"))
cols = pickle.load( open( "temp/yeast/cols.pkl", "rb"))
ind = random.sample(rows, n)

indexed_rows = [rows[i] for i in ind]
indexed_cols = [cols[i] for i in ind]

X_test_0 = np.zeros((n, 3))
X_test_0[:,0] = indexed_rows
X_test_0[:,1] = indexed_cols
X_test_0[:,2] = 0

X_test = np.array(np.vstack((edgelist, X_test_0))).astype(int)
print(X_test.shape)
y_true = [X_test[i,2] for i in range(len(X_test))]
y_predict = [sim[X_test[i,0], X_test[i,1]] for i in range(len(X_test))]
roc = roc_auc_score(y_true, y_predict)
if roc < 0.5:
    roc = 1 - roc

print(roc)

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_true, y_predict)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))