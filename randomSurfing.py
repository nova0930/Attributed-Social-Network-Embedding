from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import networkx as nx
import numpy as np
import pandas as pd

from utils import convertSortedRankTSVToAdjMatrix
from convertdata import load_data

def read_graph(inputMatrix, filename,g_type):
    if(inputMatrix is None):
        with open('data/'+filename,'rb') as f:
            if g_type == "undirected":
                G = nx.read_weighted_edgelist(f)
            else:
                G = nx.read_weighted_edgelist(f,create_using=nx.DiGraph())
    else:
        G = nx.from_numpy_matrix(inputMatrix)
    node_idx = G.nodes()
    adj_matrix = np.asarray(nx.adjacency_matrix(G, nodelist=None,weight='weight').todense())
    return adj_matrix, node_idx

def scale_sim_mat(mat):
    # Scale Matrix by row
    mat  = mat - np.diag(np.diag(mat))
    D_inv = np.diag(np.reciprocal(np.sum(mat,axis=0)))
    mat = np.nan_to_num(np.dot(D_inv,  mat))

    return mat

def random_surfing(adj_matrix,max_step,alpha):
    # Random Surfing
    nm_nodes = len(adj_matrix)
    adj_matrix = scale_sim_mat(adj_matrix)
    # print(adj_matrix.dtype)
    P0 = np.eye(nm_nodes, dtype='float32')
    M = np.zeros((nm_nodes,nm_nodes),dtype='float32')
    P = np.eye(nm_nodes, dtype='float32')
    for i in range(0,max_step):
        P = alpha * np.dot(P, adj_matrix) + (1 - alpha) * P0
        M = M + P
    return M

def PPMI_matrix(M):
    M = scale_sim_mat(M)
    nm_nodes = len(M)

    col_s = np.sum(M, axis=0).reshape(1,nm_nodes)
    row_s = np.sum(M, axis=1).reshape(nm_nodes,1)
    D = np.sum(col_s)
    rowcol_s = np.dot(row_s,col_s)
    PPMI = np.log(np.divide(D*M,rowcol_s))
    PPMI[np.isnan(PPMI)] = 0.0
    PPMI[np.isinf(PPMI)] = 0.0
    PPMI[np.isneginf(PPMI)] = 0.0
    PPMI[PPMI<0] = 0.0

    return PPMI


def main():
    parser = ArgumentParser('DNGR',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--graph_type', default='undirected',
                        help='Undirected or directed graph as edgelist')

    parser.add_argument('--random_surfing_steps', default=10, type=int,
                        help='Number of steps for random surfing')

    parser.add_argument('--random_surfing_rate', default=0.98, type=float,
                        help='alpha random surfing')

    args = parser.parse_args()


    filename = "edgelist_biogrid.txt"

    edgelist = pd.read_csv("./data/edgelist_biogrid.txt", sep=" ", header=None)
    data = load_data("./data/net4_expression_data.tsv")
    edgeMatrix = convertSortedRankTSVToAdjMatrix(edgelist, data.shape[0])


    graph_type = args.graph_type
    Ksteps = args.random_surfing_steps
    alpha = args.random_surfing_rate
    data_mat, node_idx = read_graph(edgeMatrix, filename, graph_type)
    data = random_surfing(data_mat, Ksteps, alpha)
    data = PPMI_matrix(data)
    pd.DataFrame(data).to_csv("PPMI.txt", sep=",", header=None, index=None)

if __name__ == '__main__':
    main()