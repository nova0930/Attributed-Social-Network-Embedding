import numpy as np
import math

import scipy
# from sklearn.cross_validation import cross_val_score
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
import multiprocessing

def calculate_distance( embeddings, type): # N * emb_size
    if type == 'euclidean_distances':
        Y_predict = -1.0 * euclidean_distances(embeddings, embeddings)
        return Y_predict
    if type == 'cosine_similarity':
        Y_predict = cosine_similarity(embeddings, embeddings)
        return Y_predict

def norm(a):
    sum = 0.0
    for i in range(len(a)):
        sum = sum + a[i] * a[i]
    return math.sqrt(sum)

def cosine_similarity( a,  b):
    sum = 0.0
    for i in range(len(a)):
        sum = sum + a[i] * b[i]
    if (norm(a) * norm(b) ==0):
        return 0
    return sum/(norm(a) * norm(b))

def evaluate_ROC(X_test, Embeddings):
    num_cores = multiprocessing.cpu_count()
    y_true = [ X_test[i][2] for i in range(len(X_test))]
    y_predict = Parallel(n_jobs=num_cores)(delayed(cosine_similarity)(Embeddings[X_test[i][0],:], Embeddings[X_test[i][1], :])  for i in range(len(X_test)))
    roc = roc_auc_score(y_true, y_predict)
    if roc < 0.5:
        roc = 1 - roc
    return roc

def evaluate_ROC_euclidean(X_test, Embeddings):
    y_true = [ X_test[i][2] for i in range(len(X_test))]
    y_predict = [ -1.0 * scipy.spatial.distance.euclidean(Embeddings[X_test[i][0],:], Embeddings[X_test[i][1], :]) for i in range(len(X_test))]
    roc = roc_auc_score(y_true, y_predict)
    if roc < 0.5:
        roc = 1 - roc
    return roc


def evaluate_MAP( node_neighbors_map, Embeddings, distance_measure):
    '''
    given the embeddings of nodes and the node_neighbors, return the MAP value
    :param node_neighbors_map: [node_id : neighbors_ids]
    :param nodes: a dictionary, ['node_id']--len(nodes) of id for nodes, one by one; ['node_attr']--a list of attrs for corresponding nodes
    :param Embeddings:  # nodes_number * (id_dim + attr_dim), row sequence is the same as nodes['node_id']
    :return: MAP value
    '''
    MAP = .0
    Y_true = np.zeros((len(node_neighbors_map), len(node_neighbors_map)))
    for node in node_neighbors_map:
        # prepare the y_true
        for neighbor in node_neighbors_map[node]:
            Y_true[node][neighbor] = 1


    Y_predict = calculate_distance(Embeddings,distance_measure)
    for node in node_neighbors_map:
        MAP +=  average_precision_score(Y_true[node,:], Y_predict[node,:])

    return MAP/len(node_neighbors_map)


def load_embedding(embedding_file, N, combineAttribute=False, datafile=None):
    f = open(embedding_file)
    i = 0
    line = f.readline()
    line = line.strip().split(' ')
    d = int(line[1])
    embeddings = np.random.randn(int(N), d)
    line = f.readline()
    while line:
        line = line.strip().split(' ')
        embeddings[int(line[0]),:] = line[1:]
        i = i + 1
        line = f.readline()
    f.close()
    if combineAttribute:
        data = load_datafile(datafile, N)
        # print(data.shape)
        temp = np.hstack((embeddings, data))
        # print(temp.shape)
        embeddings = temp
    return embeddings

def load_datafile(data_file, N):
    f = open(data_file)
    i = 0
    line = f.readline()
    line = line.strip().split(' ')
    d = len(line[1:])
    data = np.zeros([int(N), d])
    while line:
        # print(i)
        data[int(line[0]),:] = line[1:]
        i = i + 1
        line = f.readline()
        if i < N:
            line = line.strip().split(' ')
        else:
            break
    f.close()
    return data


def read_test_link(testlinkfile):
    X_test = []
    f = open(testlinkfile)
    line = f.readline()
    while line:
        line = line.strip().split(' ')
        X_test.append([int(line[0]), int(line[1]), int(line[2])])
        line = f.readline()
    f.close()
    return X_test
