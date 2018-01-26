import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter


# Data analysis for yeast
G=nx.read_edgelist("./data/yeast/edgelist_biogrid.txt", nodetype=int, data=(('weight',float),), delimiter=' ')

edge_degree = G.degree()

y = np.zeros([5950, 1])
for degree in edge_degree:
    y[degree[0]] = degree[1]

y_list = y.tolist()
print(y_list)
y_sorted = sorted(G.degree(),key=itemgetter(1),reverse=True)
print(y_sorted)

y_rev_sort = y_list.sort(reverse=True)
print(y_rev_sort)
# plt.plot(y_rev_sort)
# plt.show()