import random
import argparse
import numpy as np
import LoadData as data
import evaluation
from SNE import SNE
import matplotlib.pyplot as plt
import pandas as pd
from autoencoder import autoencoder

# Set random seeds
SEED = 2016
random.seed(SEED)
np.random.seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser(description="Run SNE.")
    parser.add_argument('--data_path', nargs='?', default='./data/yeast/',
                        help='Input data path')
    parser.add_argument('--id_dim', type=int, default=20,
                        help='Dimension for id_part.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--n_neg_samples', type=int, default=10,
                        help='Number of negative samples.')
    parser.add_argument('--attr_dim', type=int, default=20,
                        help='Dimension for attr_part.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Dimension for attr_part.')
    parser.add_argument('--train_autoencoder', type=bool, default=True,
                        help='Dimension for attr_part.')
    return parser.parse_args()

#################### Util functions ####################


def run_SNE( data, args, attr_weights=None ):
    model = SNE( data, id_embedding_size=args.id_dim, attr_embedding_size=args.attr_dim, epoch=args.epoch, n_neg_samples=args.n_neg_samples, batch_size=args.batch_size, pretrained_weights = attr_weights)
    embeddings, val_roc = model.train( )
    roc = evaluation.evaluate_ROC(data.X_test, embeddings)
    print("Accuracy (ROC) in Test Data set", "{:.9f}".format(roc))

if __name__ == '__main__':
    args = parse_args()
    print("data_path: ", args.data_path)
    path = args.data_path
    Data = data.LoadData( path , SEED, test_size=0.3)
    print("Total training links: ", len(Data.links))
    print("Total epoch: ", args.epoch)

    print('id_dim :', args.id_dim)
    print('attr_dim :', args.attr_dim)

    if(args.train_autoencoder):
        print("Encoding the Gene expression")
        autoencoder = autoencoder(536, batch_size=args.batch_size, num_hidden_1=128, num_hidden_2=20)
        pretrained_weights = autoencoder.train()
        run_SNE( Data, args, pretrained_weights)
    else:
        run_SNE(Data, args)

