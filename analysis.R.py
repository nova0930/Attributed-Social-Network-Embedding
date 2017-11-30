import networkx as nx

from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from utils import *


def adjToList(adj):
    G = nx.from_numpy_matrix(adj)
    return nx.to_edgelist(G)



embeddings = np.array(pd.read_csv("./output/embeddings.txt", header=0, index_col=0))

sim = cosine_similarity(embeddings)
simList = adjToList(sim)
simList = simList.sort_values([ 'c1', 'c2'], ascending=[ 1, 1])
print(simList.shape)


edgelist = pd.read_csv("./data/yeast_edgelist_biogrid.txt", sep=" ", header=None)
adj = convertSortedRankTSVToAdjMatrix(edgelist, 5950)
# G=nx.read_adjlist("./data/yeast_edgelist_biogrid.txt")
# adj = nx.adjacency_matrix(G)
adjlist = convertAdjMatrixToSortedRankTSV(adj)
adjlist = adjlist.sort_values([ 'c1', 'c2'], ascending=[ 1, 1])
print(simList.shape)
print(adjlist.shape)

y_true = adjlist[:,2]
y_predict = simList[:,2]
roc = roc_auc_score(y_true, y_predict)
print(roc)

