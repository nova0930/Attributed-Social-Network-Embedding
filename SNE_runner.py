import random
import argparse
import numpy as np
import LoadData as data
import evaluation
from SNE import SNE
import pickle
# Set random seeds
# from preprocessBiogrid import biogrid

SEED = 2016
random.seed(SEED)
np.random.seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser(description="Run SNE.")
    parser.add_argument('--data_path', nargs='?', default='./data/yeast/',
                        help='Input data path')
    parser.add_argument('--organism_id', type=int, default=4,
                        help='Organism Identifier: 3 for Ecoli and 4 for Yeast')
    parser.add_argument('--id_dim', type=int, default=20,
                        help='Dimension for id_part.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Numbe r of epochs.')
    parser.add_argument('--n_neg_samples', type=int, default=10,
                        help='Number of negative samples.')
    parser.add_argument('--attr_dim', type=int, default=20,
                        help='Dimension for attr_part.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Dimension for attr_part.')
    parser.add_argument('--alpha', type=int, default=1,
                        help='Regularize the attribute embeddings')
    return parser.parse_args()

#################### Util functions ####################

def run_SNE( path, data, args, attr_weights=None ):
    model = SNE(path, data, id_embedding_size=args.id_dim, attr_embedding_size=args.attr_dim, epoch=args.epoch, n_neg_samples=args.n_neg_samples, batch_size=args.batch_size, pretrained_weights = attr_weights, alpha = args.alpha)
    embeddings = model.train( )
    return embeddings

if __name__ == '__main__':
    args = parse_args()
    print("data_path: ", args.data_path)
    path = args.data_path
    # # biogrid(path, 'ecoli')
    # Data = data.LoadData( path , SEED, test_size=0.3, organism_id=args.organism_id)

    data_file = open('output/processedData.pkl', 'rb')
    Data = pickle.load(data_file)
    data_file.close()

    print("Total training links: ", len(Data.links))
    print("Total epoch: ", args.epoch)

    print('id_dim :', args.id_dim)
    print('attr_dim :', args.attr_dim)

    embeddings = run_SNE(path, Data, args)
    roc = evaluation.evaluate_ROC(Data.X_test, embeddings)
    print("Accuracy (ROC) in Test Data set", "{:.9f}".format(roc))

    print("Evaluating Node2Vec embeddings")
    embeddings_file = path + 'node2vec_vec_all.txt'
    emb = evaluation.load_embedding(embeddings_file, Data.id_N)

    roc = evaluation.evaluate_ROC(Data.X_test, emb)
    print("Node2vec Method: Accuracy (ROC) in Test Data set", "{:.9f}".format(roc))

    print("Evaluating LINE embeddings")
    embeddings_file = path + 'line_vec_all.txt'
    emb = evaluation.load_embedding(embeddings_file, Data.id_N)

    roc = evaluation.evaluate_ROC(Data.X_test, emb)
    print("LINE Method: Accuracy (ROC) in Test Data set", "{:.9f}".format(roc))

    print("Evaluating DeepWalk embeddings")
    embeddings_file = path + 'deepwalk_vec_all.txt'
    emb = evaluation.load_embedding(embeddings_file, Data.id_N)

    roc = evaluation.evaluate_ROC(Data.X_test, emb)
    print("DeepWalk: Accuracy (ROC) in Test Data set", "{:.9f}".format(roc))