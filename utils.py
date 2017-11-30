#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 10:26:11 2017

@author: kishan_kc
"""
import csv
from itertools import chain
from random import randint

import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import networkx as nx


def plot_correlation_matrix(corr):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(corr, interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Gene Expression Correlation')
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=np.arange(-1.0, 1.0, 0.05))
    plt.show()


def transformGraphToAdj(graph):
    n = graph.number_of_nodes()
    adj = np.zeros((n, n))

    for (src, dst, w) in graph.edges(data="weight", default=1):
        adj[src, dst] = w

    return w


def saveGraphToEdgeListTxt(graph, file_name):
    with open(file_name, 'w') as f:
        f.write('#d\n' % graph.number_of_nodes())
        f.write('#d\n' % graph.number_of_edges())
        for i, j, w in graph.edges(data='weight', default=1):
            f.write('#d #d #f\n' % (i, j, w))


def saveGraphToEdgeListTxtn2v(graph, file_name):
    with open(file_name, 'w') as f:
        for i, j, w in graph.edges(data='weight', default=1):
            f.write('#d #d #f\n' % (i, j, w))


def splitGraphToTrainTest(graph, train_ratio, is_undirected=True):
    train_graph = graph.copy()
    test_graph = graph.copy()
    node_num = graph.number_of_nodes()
    for (st, ed, w) in graph.edges(data='weight', default=1):
        if (is_undirected and st >= ed):
            continue
        if (np.random.uniform() <= train_ratio):
            test_graph.remove_edge(st, ed)
            if (is_undirected):
                test_graph.remove_edge(ed, st)
        else:
            train_graph.remove_edge(st, ed)
            if (is_undirected):
                train_graph.remove_edge(ed, st)
    return (train_graph, test_graph)


def convertEdgeListToAdj(adj, threshold=0.0, is_undirected=True, edge_pairs=None):
    result = []
    node_num = adj.shape[0]
    if edge_pairs:
        for (st, ed) in edge_pairs:
            if adj[st, ed] >= threshold:
                result.append((st, ed, adj[st, ed]))
    else:
        for i in range(node_num):
            for j in range(node_num):
                if (j == i):
                    continue
                if (is_undirected and i >= j):
                    continue
                if adj[i, j] > threshold:
                    result.append((i, j, adj[i, j]))
    return result


def splitDataToTrainTest(data, selection=None):
    n = np.sum(data[:, -1]) * 2
    if selection:
        X = data[0:int(n), 0:-1]
        y = data[0:int(n), -1]
    else:
        X = data[:, 0:-1]
        y = data[:, -1]

    # scaler = StandardScaler()
    # X_train_scaler = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y,   stratify=y,  test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test


def balancedErrorRate(actual, output):
    cm = confusion_matrix(actual, output)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    return 0.5 * ((float(FN) / float(TP + FN)) + (float(FP) / float(FP + TN)))

def saveToFile(filename, data):
    fileHandler = open(filename,'w')
    fileHandler.write(data)
    fileHandler.close()

# Create Data Loader
def load_data(datafile):
    """
    This function loads data set
    :param datafile:
    :return expression data:
    """
    # Load data file
    df = pd.read_csv(datafile, sep='\t', header= 0)
    t_data = df.T
    return(t_data)


def loadEmbedding(file_name):
    with open(file_name, 'r') as f:
        n, d = f.readline().strip().split()
        X = np.zeros((int(n), int(d)))
        for line in f:
            emb = line.strip().split()
            emb_fl = [float(emb_i) for emb_i in emb[1:]]
            X[int(emb[0]), :] = emb_fl
    return X

def load_gold_standard(file_name, sep="\t"):
    df = pd.read_csv(file_name, sep=sep, header=None)
    # Load gold standard relation file
    df[0] = df[0].apply(lambda x: x.replace('g', '').replace('G', ''))
    df[1] = df[1].apply(lambda x: x.replace('g', '').replace('G', ''))
    df = df.astype(float)  # imoprtant for later to check for equality
    df[0] = df[0].apply(lambda x: x - 1)
    df[1] = df[1].apply(lambda x: x - 1)
    df = df.astype(float)  # imoprtant for later to check for equality
    return df

# Create Graph based on Correlation Matrix

def generateMultipleGraphs(correlation_matrix, num_of_thresholds):
    """
    :param correlation_matrix:
    :param num_of_thresholds:
    :param range:
    :return: Multiple adjacency matrix based on thresholds
    """
    mat = np.abs(correlation_matrix).copy()
    mat = mat - np.diag(np.ones(mat.shape[0]))
    final = nx.Graph()
    for i in range(num_of_thresholds):
        threshold = randint(20, 90)/100
        graph = generateGraphs(mat, threshold)
        G = nx.Graph(graph)
        final = nx.compose(final, G)
    return final

def generateGraphs(correlation_matrix, threshold):
    """
    :param correlation_matrix:
    :param threshold:
    :return: Adjacency matrix based on correlation threshold
    """
    threshold_matrix = np.abs(correlation_matrix).copy()
    threshold_matrix = threshold_matrix - np.diag(np.ones(threshold_matrix.shape[0]))
    threshold_matrix[np.abs(correlation_matrix)<threshold] = 0
    return (threshold_matrix)


# def biogrid_goldStandard(gold_standard, gene_lists):
def biogrid_goldStandard():
    gene_list = pd.read_csv("../data/net4_gene_ids.tsv")
    file_name = '../data/BIOGRID-ALL-3.4.153.tab2.txt'
    gold_standard_data = pd.read_csv(file_name, sep="\t")
    yeast_data = gold_standard_data[(gold_standard_data['Organism Interactor A'] == 559292) & (gold_standard_data['Organism Interactor B'] == 559292)]
    yeast_data = yeast_data[['Systematic Name Interactor A', 'Systematic Name Interactor B']]
    yeast_data.to_csv("../data/edgelist_biogrid.csv", index=False, header=False)

def adj_to_list(A, output_filename, delimiter):
    '''Takes the adjacency matrix on file input_filename into a list of edges and saves it into output_filename'''
    List = [('Source', 'Target', 'Weight')]
    for source in A.index.values:
        for target in A.index.values:
            List.append((target, source, A[source][target]))
    with open(output_filename, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(List)
    return List


# Implement DeepWalk, SDNE, GCN, GVAE

# Implement Hadamard, Average, L1, L2 approach to combine two nodes
def average(u, v):
    return np.add(u, v)/2

def hadamard(u, v):
    return np.multiply(u, v)

def weighted_L1(u, v):
    return np.abs(u - v)

def weighted_L2(u, v):
    return np.square(u - v)


# Create Feature for Classification using embedding
def createFeaturesFromEmbedding(emb, gold_standard):

    X = np.zeros((emb.shape[0], emb.shape[0]))

    for i in range(gold_standard.shape[0]):
            X[int(gold_standard[i,0]), int(gold_standard[i,1])] = int(gold_standard[i,2])

    data = np.zeros((gold_standard.shape[0], int(emb.shape[1])))
    labels = gold_standard[:, 2]

    for i in range(gold_standard.shape[0]):
        data[i] = average(emb[int(gold_standard[i,0]),:], emb[int(gold_standard[i,1]),:])

    return data, labels


def convertAdjMatrixToSortedRankTSV(inputFile=None, outputFilename=None, desc=True):


    tbl = inputFile

    rownames = range(tbl.shape[0])
    # First column -> repeat the predictors
    firstCol = np.repeat(rownames, tbl.shape[1]).reshape((tbl.shape[0]*tbl.shape[1], 1))
    # Second column -> repeat the targets
    secondCol = []
    for i in range(tbl.shape[1]):
        # print(i)
        secondCol = np.append(secondCol, range(tbl.shape[0]))

    secondCol = secondCol.reshape((tbl.shape[0]*tbl.shape[1], 1))
    thirdCol = np.matrix.flatten(np.matrix(tbl)).reshape((tbl.shape[0]*tbl.shape[1], 1))
    # Gets the indices from a desc sort on the adjancy measures

    # Glue everything together
    result = np.column_stack((firstCol, secondCol, thirdCol))
    # Convert to dataframe
    result = pd.DataFrame(result)
    result.columns = ['c1','c2', 'c3']

    result = pd.DataFrame(result[result['c1']!=result['c2']])
    # Sort it using the indices obtained before
    result =  result.sort_values(['c3', 'c1', 'c2'], ascending=[0, 1, 1])
    result[['c1', 'c2', 'c3']] = result[['c1', 'c2', 'c3']].astype(int)
    # print("Write to file if filename is given")
    # result.to_csv(outputFilename, header=False, columns=None, index=False )
    # else write to function output
    return (result)


def convertSortedRankTSVToAdjMatrix (input=None, nodes=None):
    tbl = pd.DataFrame(input).drop_duplicates()
    tbl.columns = ['c1', 'c2', 'c3']
    tbl = tbl[tbl['c3']==1]

    tbl = tbl.sort_values(['c2'], ascending=1)
    tbl[['c1', 'c2', 'c3']] = tbl[['c1', 'c2', 'c3']].astype(int)
    # Pre allocate return matrix
    m = np.zeros((nodes, nodes))
    # Get the duplicates
    dups = tbl['c2'].duplicated()

    # # Get the startIndices of another column
    startIndices = list(np.where(dups== False)[0])

    for i in range(len(startIndices)-1):
        # print(i)
        colIndex = tbl.iloc[startIndices[i], 1]
        if startIndices[i]==(startIndices[i + 1] - 1):
            rowIndexes = tbl.iloc[startIndices[i], 0]
            valuesToAdd = tbl.iloc[startIndices[i], 2]
        else:
            rowIndexes = tbl.iloc[startIndices[i]:(startIndices[i + 1]), 0].values
            valuesToAdd = tbl.iloc[startIndices[i]:(startIndices[i + 1] ), 2].values
        m[rowIndexes, colIndex] = valuesToAdd


    colIndex = tbl.iloc[startIndices[len(startIndices)-1], 1]
    rowIndexes = tbl.iloc[startIndices[len(startIndices)-1]:len(tbl.iloc[:, 1]), 0]
    valuesToAdd = tbl.iloc[startIndices[len(startIndices)-1]:len(tbl.iloc[:, 1]), 2]

    m[rowIndexes, colIndex] = valuesToAdd
    # m = pd.DataFrame(m)
    # m.to_csv(outputFilename, header=False, columns=None, index=False )
    # else write to function output
    return (m)




# def evaluate(prediction, gold_standard_file, totalPredictionsAccepted):
#
#     prediction = pd.DataFrame(prediction)
#     prediction = prediction.sort_values([0,1], ascending=[True, True])
#     gold_standard_data = load_gold_standard(gold_standard_file)
#     gold_standard_data = gold_standard_data.sort_values([0,1], ascending=[True, True])
#
#     y_true = gold_standard_data.values[0:totalPredictionsAccepted,2]
#     y_preds = prediction.values[0:totalPredictionsAccepted,2]
#     print(classification_report(y_true,y_preds))
#     print(confusion_matrix(y_true, y_preds))
#     # gold = convertRankedListToAdjMatrix(gold_standard_data)

def evaluate(prediction, gold_standard_file, totalPredictionsAccepted):
    prediction = pd.DataFrame(prediction)
    # if(len(prediction)>totalPredictionsAccepted):
    #  prediction = prediction[0:totalPredictionsAccepted,:]

    gold_standard_data = load_gold_standard(gold_standard_file)
    gold = convertSortedRankTSVToAdjMatrix(gold_standard_data)

    p = np.sum(gold)
    # print(p)
    n = gold.shape[0] * gold.shape[1] - p - gold.shape[0]
    # print(n)
    t = p + n
    # print(t)

    firstCol = prediction.iloc[:,0]
    secondCol = prediction.iloc[:, 1]

    thirdCol = np.zeros(len(firstCol))
    # Will indicate the rank of this prediction = (-1 if it is not present and [rank] otherwise]
    rank = np.zeros(len(firstCol))
    # Now, loop over all predictions and determine if they are present in gold matri
    correct = 0
    incorrect = 0

    for i in range(len(firstCol)):
     if (int(gold.iloc[int(firstCol[i]), int(secondCol[i])]) == 1):
         correct = correct + 1
         thirdCol[i] = 1
         rank[i] = correct
     else:
         incorrect = incorrect - 1
         thirdCol[i] = 0
         rank[i] =incorrect
    # Check how many of the gold standard edges we predicted. Any other that still remain are discovered at a uniform rate by definition. (remaining_gold_edges/remaining_edges)
    # Gold links predicted so far


    # If some gold links have not been predicted, calculate the random discovery chance, else set to zero
    if (len(firstCol) < t):
     odds = (p-correct) / (t-len(firstCol))
    else:
     odds = 0

    # Each guess you have 'odds' chance of getting one right and '1-odds' chance of getting it wrong , now construct a vector till the end
    random_positive =  np.repeat(odds, t - len(firstCol))
    random_negative = np.repeat(1- odds, t - len(firstCol))
    # Calculate the amount of true positives and false positives at 'k' guesses depth
    positive = np.concatenate([thirdCol, random_positive])
    negative = np.concatenate([abs(thirdCol - 1), random_negative])
    tpk = np.cumsum(positive)
    fpk = np.cumsum(negative)
    # Depth k
    k = range(1, int(t) + 1)

    # Calculate true positive rate, false positive rate, precision and recall
    tpr = tpk / p
    fpr = fpk / n
    rec = tpr
    prec = tpk / k

    predictors = firstCol
    targets = secondCol

    tpk[-1] = round(tpk[-1])
    fpk[-1] = round(fpk[-1])

    # Integrate area under ROC using trapezoidal rule
    # AUROC = 0;
    # for i = 1:T - 1
    # AUROC = AUROC + (FPR(i + 1) - FPR(i)) * (TPR(i + 1) + TPR(i)) / 2;
    # end

    # faster built - in integration function
    AUROC = np.trapz(fpr, tpr)

    # Integrate area under PR curve using(Stolovitzky et al. 2009)

    # initialize the first value
    # if positive_discovery(1)
     # A = 1 / P;
    # else
    # A = 0;
    # end

    # integrate up to L
    # for k = 2:L
    # if positive_discovery(k)
     # A = A + 1 / P * (1 - FPk(k - 1) * log(k / (k - 1)));
    # end
    # end
    #
    # the remainder of the list is monatonic, so use trapezoidalrule
    # A = A + trapz(REC(L + 1:end), PREC(L + 1:end));
    #
    # finally, normalize by max  possible value
    # A = A / (1 - 1 / P);
    # # built - in function
    AUPR = np.trapz(rec, prec) / (1 - 1 / p)

    return AUROC, AUPR


def AVGC(data, cluster, index):
    # print(index)
    sum_corr = 0
    if (len(cluster)>0):
        cluster = np.array([int(d[1:]) for d in cluster])
        for x in cluster:
            if (x != index):
                c = np.corrcoef(data.iloc[index, :], data.iloc[x, :])[0, 1]
                sum_corr = sum_corr + c
    else:
        cluster = [1]
    return sum_corr/len(cluster)



def DCCA(data, iterations):
    '''

    :param coeff_matrix:
    :return: clusters
    '''
    # i = (coeff_matrix).argsort(axis=None, kind='mergesort')
    # j = np.unravel_index(i, coeff_matrix.shape)
    # indices = np.vstack(j).T
    #
    C = 1
    # index = indices[0]
    i = 0

    genes = ['G{0}'.format(i) for i in range(data.shape[0])]
    clusters = {}
    clusters[0] = genes
    final_clusters = {}
    final_clusters.update(clusters)
    no_repulsion = 0

    prev_clusters = {}
    for iter in range(iterations):
        no_repulsion = 0
        final_clusters = clusters.copy()
        for i, value in clusters.items():

            if(len(value)> 1):
                cluster = np.array([int(d[1:]) for d in value])
                coeff_matrix = np.corrcoef(data.iloc[cluster,:])
                # print(cluster)
                rank = (coeff_matrix).argsort(axis=None, kind='mergesort')
                j = np.unravel_index(rank, coeff_matrix.shape)
                indices = np.vstack(j).T
                index = indices[0]
                xi = index[0]
                xj = index[1]
                if (coeff_matrix[xi, xj] < 0):
                    # print(coeff_matrix[xi, xj])
                    del final_clusters[i]
                    # print(final_clusters)
                    temp_clusters ={}
                    temp_clusters[C] = []
                    temp_clusters[C].append(genes[cluster[xi]])

                    temp_clusters[C + 1] = []
                    temp_clusters[C + 1].append(genes[cluster[xj]])

                    for k in range(len(cluster)):
                        xk = cluster[k]
                        corr1 = np.corrcoef(data.iloc[xk,:], data.iloc[xi,:])[0,1]
                        corr2 = np.corrcoef(data.iloc[xk, :], data.iloc[xj, :]) [0,1]
                        # print(xk)
                        if( corr1 > corr2):
                            temp_clusters[C].append(genes[xk])
                        else:
                            temp_clusters[C + 1].append(genes[xk])
                    # print(cluster)
                    final_clusters.update(temp_clusters)
                    # print(temp_clusters)
                    C += 2
                else:
                    no_repulsion += 1

        if(prev_clusters == clusters):
            break
        prev_clusters = clusters.copy()
        empty_keys = [k for k, v in clusters.items() if not v]
        for k in empty_keys:
            del clusters[k]
        # print(no_repulsion, len(clusters))
        clusters = {}
        ind = 0
        for key, value in final_clusters.items():
            clusters[ind] = set(value)
            ind += 1

        # print(clusters)

        CNew ={}
        for p,value in clusters.items():
            CNew[p] = []

        for xk in range(len(genes)):
            avgcpk = []
            for p, value in clusters.items():
                avgcpk.append(AVGC(data, value, xk))
            CNew[avgcpk.index(min(avgcpk))].append(genes[xk])


        for p, value in clusters.items():
            if CNew[p] != clusters[p]:
                clusters[p] = set(CNew[p])
                CNew[p] = []

    return(clusters)


def calc_weight(data, index_x_i, cluster_x_i, index_x_j, cluster_x_j, L1 = 0.3):
    '''
    Given gene i and j and their respective clusters Ci and Cj, weighted confidence scores of interactions between two proteins is calculated as

        W(Xi, Xj) = L1(||Xi-Ci||^2 + ||Xj-Cj||^2 ) + (1 - L1) * ||Ci - Cj||^2

    :param index_x_i: Index of Gene i
    :param cluster_x_i: Cluster in which Gene i belongs
    :param index_x_j: Index of Gene j
    :param cluster_x_j: Cluster in which Gene j belongs
    :param L1: tradeoff parameter
    :return: returns weight that represents association between Genes i and j
    '''

    x_i = data.iloc[index_x_i,]
    C_i = np.mean(data.iloc[cluster_x_i,], axis=0)

    x_j = data.iloc[index_x_j,]
    C_j = np.mean(data.iloc[cluster_x_j,], axis=0)

    distance_x_i = np.math.sqrt(sum([(a - b) ** 2 for a, b in zip(x_i, C_i)]))
    distance_x_j = np.math.sqrt(sum([(a - b) ** 2 for a, b in zip(x_j, C_j)]))
    distance_C = np.math.sqrt(sum([(a - b) ** 2 for a, b in zip(C_i, C_j)]))

    w = L1 * (distance_x_i + distance_x_j) + (1-L1) * (distance_C)
    return w