import random
import argparse
import numpy as np
import pandas as pd
import time
# from autoencoder import Autoencoder

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
    parser.add_argument('--organism', nargs='?',                default='yeast',    help='Input data path')
    parser.add_argument('--id_dim',                 type=int,   default=64,                 help='Dimension for id_part.')
    parser.add_argument('--epoch',                  type=int,   default=100,                help='Number of epochs.')
    parser.add_argument('--n_neg_samples',          type=int,   default=64,                 help='Number of negative samples.')
    parser.add_argument('--attr_dim',               type=int,   default=64,                 help='Dimension for attr_part.')
    parser.add_argument('--batch_size',             type=int,   default=64,                help='Batch size for training GNE.')
    parser.add_argument('--representation_size',    type=int,   default=32,                 help='Dimension of representation vector')
    return parser.parse_args()

#################### Util functions ####################


def run_GNE(path, data, args):
    # alpha is a parameter that adjusts the effect of attributes in the model
    for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        t1      = time.time()
        model   = GNE(path,  data, id_embedding_size=args.id_dim, attr_embedding_size=args.attr_dim, \
                     batch_size=args.batch_size, alpha=alpha, epoch = args.epoch, representation_size=args.representation_size)
        embeddings, auroc = model.train( )
        t2 = time.time()
        print("time taken: " + str(t2 - t1))
    return embeddings

if __name__ == '__main__':
    args = parse_args()
    organism = args.organism
    path = './data/' + organism +'/'
    print("data_path: ", path)
    # Data = data.LoadData( path , SEED, 0.2, 3)
    # data_file = open('output/yeast/processedData.pkl', 'wb')
    # pickle.dump(Data, data_file)
    # data_file.close()

    # load saved data file for 80/10/10 train/test/validation split
    print("Preprocessed file: output/"+ organism + "/processedData.pkl")
    data_file   = open('output/'+ organism + '/processedData.pkl', 'rb')
    Data        = pickle.load(data_file)
    data_file.close()

    
    print("Total training links: ", len(Data.links))
    print("Total epoch: ", args.epoch)

    print('id_dim :', args.id_dim)
    print('attr_dim :', args.attr_dim)

    embeddings = run_GNE(path, Data, args)


