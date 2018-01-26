import random
import argparse
import numpy as np
import pandas as pd
import time
# from autoencoder import Autoencoder
import math 

import LoadData as data
from GNE import GNE
import pickle
# Set random seeds
from evaluation import evaluate_MAP

SEED = 2016
random.seed(SEED)
np.random.seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser(description="Run GNE.")
    parser.add_argument('--organism', nargs='?',                default='ecoli',    help='Input data path')
    parser.add_argument('--id_dim',                 type=int,   default=64,         help='Dimension for id_part.')
    parser.add_argument('--epoch',                  type=int,   default=20,        help='Number of epochs.')
    parser.add_argument('--n_neg_samples',          type=int,   default=64,         help='Number of negative samples.')
    parser.add_argument('--attr_dim',               type=int,   default=64,         help='Dimension for attr_part.')
    parser.add_argument('--batch_size',             type=int,   default=128,        help='Batch size for training GNE.')
    parser.add_argument('--representation_size',    type=int,   default=128,        help='Dimension of representation vector')
    parser.add_argument('--learning_rate',          type=float, default=0.005,      help='Learning rate')
    parser.add_argument('--hidden_layers',          type=float, default=1,          help='Number of hidden layers for joint transformation')
    return parser.parse_args()

#################### Util functions ####################


def run_GNE(path, data, args):
    # alpha is a parameter that adjusts the effect of attributes in the model
    if args.organism == "ecoli":
        a = 1
    elif args.organism == "yeast":
        a = 0.8
    for alpha in [0.0, a]:
        t1      = time.time()
        model   = GNE(path,  data, id_embedding_size=args.id_dim, attr_embedding_size=args.attr_dim, \
                     batch_size=args.batch_size, alpha=alpha, epoch = args.epoch, representation_size=args.representation_size, learning_rate=args.learning_rate)
        embeddings, auroc = model.train( )
        t2 = time.time()
        print("time taken: " + str(t2 - t1))
    return embeddings

if __name__ == '__main__':
    args = parse_args()
    organism = args.organism
    path = './data/' + organism +'/'
    print("data_path: ", path)
    if args.organism == "ecoli":
        organism_id = 3
    elif args.organism == "yeast":
        organism_id = 4

    test_size = 0.2
    print("Test size: ", test_size)
    Data = data.LoadData( path , SEED, test_size, organism_id)
    data_file = open('output/ecoli/processedData_doublelinked.pkl', 'wb')
    pickle.dump(Data, data_file)
    data_file.close()

    # # load saved data file for 80/10/10 train/test/validation split
    # print("Preprocessed file: output/"+ organism + "/processedData_doublelinked.pkl")
    # data_file   = open('output/'+ organism + '/processedData_doublelinked.pkl', 'rb')
    # Data        = pickle.load(data_file)
    # data_file.close()

    print("Total number of nodes: ", Data.id_N)
    print("Total number of attributes: ", Data.attr_M)

    k = args.representation_size
    id = math.sqrt(Data.id_N)
    attr = math.sqrt(Data.attr_M)

    x = args.hidden_layers * float(k)/float(id + attr)
    args.id_dim = round(id * x)
    args.attr_dim = round(attr * x)

    print("Total training links: ", len(Data.links))
    print("Total epoch: ", args.epoch)

    print('id_dim :', args.id_dim)
    print('attr_dim :', args.attr_dim)

    embeddings = run_GNE(path, Data, args)


