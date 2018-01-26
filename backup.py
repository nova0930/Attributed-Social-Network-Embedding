'''
Tensorflow implementation of Gene Network Embedding framework (GNE)

@author: Kishan K C (kk3671@rit.edu)

'''

import pandas as pd
import math
import random
import os
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
import evaluation

class GNE(BaseEstimator, TransformerMixin):
    def __init__(self, path, data, id_embedding_size, attr_embedding_size,
                 batch_size=128, alpha = 1, beta = 0.2, n_neg_samples=10,
                epoch=100, random_seed = 2018, representation_size = 128, learning_rate = 0.001):
        # bind params to class
        self.path                   = path
        self.node_neighbors_map     = data.node_neighbors_map
        self.batch_size             = batch_size
        self.node_N                 = data.id_N
        self.attr_M                 = data.attr_M
        self.X_train                = data.X
        self.X_test                 = data.X_test
        self.X_validation           = data.X_validation
        self.nodes                  = data.nodes
        self.id_embedding_size      = id_embedding_size
        self.attr_embedding_size    = attr_embedding_size
        self.alpha                  = alpha
        self.n_neg_samples          = n_neg_samples
        self.epoch                  = epoch
        self.random_seed            = random_seed
        self.hidden_size_1          = 256
        self.hidden_size_2          = 128
        self.beta                   = beta
        self.learning_rate          = learning_rate
        # define the tower structure with later layer having half number of neurons than previous layer
        self.hidden_layer_size_1    = (int)((self.id_embedding_size + self.attr_embedding_size )/2)
        self.hidden_layer_size_2    = (int) ((self.hidden_layer_size_1)/2) 
        # self.hidden_layer_size_3    = (int) ((self.hidden_layer_size_2)/2) 
        self.representation_size    = representation_size

        # init all variables in a tensorflow graph
        self._init_graph()
        print("For threshold ", self.alpha)
        print("For Batch Size ", self.batch_size)


    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/gpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            
            # Define a placeholder for input data
            self.train_data_id      = tf.placeholder(tf.int32, shape=[None])  # batch_size * 1
            self.train_data_attr    = tf.placeholder(tf.float32, shape=[None, self.attr_M])  # batch_size * attr_M
            self.train_labels       = tf.placeholder(tf.int32, shape=[None, 1])  # batch_size * 1
            self.keep_prob          = tf.placeholder(tf.float32)

            # load initialzed variable.
            self.weights            = self._initialize_weights()

            # Model.
            # Look up embeddings for node_id. u = ENC(node_id)
            self.id_embed           =  tf.nn.embedding_lookup(self.weights['in_embeddings'], self.train_data_id) # batch_size * id_dim
            
            # # non linear transformation of attribute information to capture the non-linearities
            # self.attr_input_dropout =  tf.nn.dropout(self.train_data_attr, self.keep_prob)
            # self.attr_layer_1         =  tf.nn.elu(tf.add(tf.matmul(self.train_data_attr, self.weights['attr_hidden_1']), self.weights['attr_bias_1'])) # batch_size * hidden_size_1
            # self.dropout_attr_1     =  tf.nn.dropout(self.attr_layer_1, self.keep_prob)
            # self.attr_layer_2         =  tf.nn.elu(tf.add(tf.matmul(self.attr_layer_1, self.weights['attr_hidden_2']), self.weights['attr_bias_2'])) # hidden_size_1 * hidden_size_2
            # self.dropout_attr_2     =  tf.nn.dropout(self.attr_layer_2, self.keep_prob)
            self.attr_embed         =  tf.nn.elu(tf.add(tf.matmul(self.train_data_attr, self.weights['attr_embeddings']), self.weights['attr_bias'])) # hidden_size_2 * attr_dim
            
            # fusion layer to create a joint representation vector
            self.embed_layer        =  tf.concat([ self.id_embed, self.alpha * (self.attr_embed)], 1) # batch_size * (id_dim + attr_dim)

            # # ## Hidden layers for non-linear transformation of joint representation
            self.dropout_1          = tf.nn.dropout(self.embed_layer, self.keep_prob)
            self.representation_layer = tf.nn.softsign(tf.add(tf.matmul(self.dropout_1, self.weights['representation_layer']), self.weights['representation_layer_bias']))
            # self.dropout_2          = tf.nn.dropout(self.hidden_layer_1, self.keep_prob)
            # self.hidden_layer_2     = tf.nn.tanh(tf.add(tf.matmul(self.dropout_2, self.weights['hidden_layer_2']), self.weights['hidden_bias_2']))
            # self.dropout_3          = tf.nn.dropout(self.hidden_layer_2, self.keep_prob)
            # self.hidden_layer_3     = tf.nn.tanh(tf.add(tf.matmul(self.dropout_3, self.weights['hidden_layer_3']), self.weights['hidden_bias_3']))

            
            # fusion layer to create a joint representation vector
            # self.representation_layer        =  tf.nn.elu(tf.add(tf.matmul(self.hidden_layer_1, self.weights['representation_layer']), self.weights['representation_layer_bias']))

            # ## Hidden layers for non-linear transformation of joint representation
            # self.representation_layer = self.embed_layer
            
            # Compute the loss, using a sample of the negative labels each time.
            self.loss =  tf.reduce_mean(tf.nn.sampled_softmax_loss(self.weights['out_embeddings'], self.weights['biases'],
                                                  self.train_labels, self.representation_layer, self.n_neg_samples, self.node_N))

            # Optimizer.
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            
            # init
            init        = tf.global_variables_initializer()
            self.sess   = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            self.sess.run(init)


    def _initialize_weights(self):
        all_weights = dict()
        all_weights['in_embeddings']    = tf.Variable(tf.random_uniform([self.node_N, self.id_embedding_size], -1.0, 1.0))  # id_N * id_dim
        all_weights['out_embeddings']   = tf.Variable(tf.random_normal([self.node_N, self.representation_size]))
        all_weights['biases']           = tf.Variable(tf.zeros([self.node_N]))

        all_weights['attr_embeddings']  = tf.Variable(tf.random_normal([self.attr_M, self.attr_embedding_size]))  # attr_M * attr_dim
        all_weights['attr_bias']        = tf.Variable(tf.zeros([ self.attr_embedding_size]))

        # Weight initialization for hidden layers for joint representation transformation
        # all_weights['hidden_layer_1']                   = tf.Variable(tf.random_normal([self.id_embedding_size + self.attr_embedding_size, self.hidden_layer_size_1]))
        # all_weights['hidden_bias_1']                    = tf.Variable(tf.zeros([self.hidden_layer_size_1]))
        # all_weights['hidden_layer_2']                   = tf.Variable(tf.random_normal([self.hidden_layer_size_1, self.hidden_layer_size_2]))
        # all_weights['hidden_bias_2']                    = tf.Variable(tf.zeros([self.hidden_layer_size_2]))
        # all_weights['hidden_layer_3']                   = tf.Variable(tf.random_normal([self.hidden_layer_size_2, self.hidden_layer_size_3]))
        # all_weights['hidden_bias_3']                    = tf.Variable(tf.zeros([self.hidden_layer_size_3]))
        all_weights['representation_layer']             = tf.Variable( tf.random_normal([self.id_embedding_size + self.attr_embedding_size, self.representation_size]))
        all_weights['representation_layer_bias']        = tf.Variable(tf.zeros([self.representation_size]))



        # Weight initialization for hidden layers for attribute transformation
        all_weights['attr_hidden_1']    = tf.Variable(tf.random_normal([self.attr_M, self.hidden_size_1]))
        all_weights['attr_bias_1']      = tf.Variable(tf.zeros([self.hidden_size_1]))

        all_weights['attr_hidden_2']    = tf.Variable(tf.random_normal([self.hidden_size_1, self.hidden_size_2]))
        all_weights['attr_bias_2']      = tf.Variable(tf.zeros([self.hidden_size_2]))

        return all_weights

    def partial_fit(self, X): # fit a batch
        feed_dict = {self.train_data_id: X['batch_data_id'], self.train_data_attr: X['batch_data_attr'],
                     self.train_labels: X['batch_data_label'], self.keep_prob : 0.7}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, data, batch_size):
        start_index = np.random.randint(0, len(data) - batch_size)
        return data[start_index:(start_index + batch_size)]

    def train(self): # fit a dataset

        total_iterations = 0
        # Best validation accuracy seen so far.
        best_validation_accuracy = 0.0

        # Iteration-number for last improvement to validation accuracy.
        last_improvement = 0

        # Stop optimization if no improvement found in this many iterations.
        require_improvement = 5


        print('Using in + out embedding')
        for epoch in range( self.epoch ):
            perm = np.random.permutation(len(self.X_train['data_id_list']))
            self.X_train['data_id_list']    = self.X_train['data_id_list'][perm]
            self.X_train['data_attr_list']  = self.X_train['data_attr_list'][perm]
            self.X_train['data_label_list'] = self.X_train['data_label_list'][perm]
            total_batch = int( len(self.X_train['data_id_list']) / self.batch_size)
            # Loop over all batches
            total_iterations += 1
            avg_cost = 0.
            for i in range(total_batch):
                # generate a batch data
                batch_xs = {}
                start_index = np.random.randint(0, len(self.X_train['data_id_list']) - self.batch_size)
                batch_xs['batch_data_id']       = self.X_train['data_id_list'][start_index:(start_index + self.batch_size)]
                batch_xs['batch_data_attr']     = self.X_train['data_attr_list'][
                                                    start_index:(start_index + self.batch_size)]
                batch_xs['batch_data_label']    = self.X_train['data_label_list'][
                                                    start_index:(start_index + self.batch_size)]

                # Fit training using batch data
                cost = self.partial_fit(batch_xs)
                avg_cost += cost / total_batch


            # Display logs per epoch
            Embeddings_out  = self.getEmbedding('out_embedding', self.nodes)
            Embeddings_in   = self.getEmbedding('embed_layer', self.nodes)
            Embeddings      = Embeddings_out + Embeddings_in

            # link prediction test
            roc = evaluation.evaluate_ROC(self.X_validation, Embeddings)
            
            # If validation accuracy is an improvement over best-known.
            if roc > best_validation_accuracy:
                # Update the best-known validation accuracy.
                best_validation_accuracy = roc

                # Set the iteration for the last improvement to current.
                last_improvement = total_iterations

                # Save all variables of the TensorFlow graph to file.
                self.embedding_checkpoints(Embeddings, "save")

                # A string to be printed below, shows improvement found.
                improved_str = '*'
            else:
                # An empty string to be printed below.
                # Shows that no improvement was found.
                improved_str = ''

            # Status-message for printing.
            msg = "Epoch: {0:>6}, Train-Batch Loss: {1:.9f}, Validation AUC: {2:.9f} {3}"

            # Print it.
            print(msg.format(epoch + 1, avg_cost, roc, improved_str))

            # If no improvement found in the required number of iterations.
            if total_iterations - last_improvement > require_improvement:
                print("No improvement found in a while, stopping optimization.")

                # Break out from the for-loop.
                break

        Embeddings = self.embedding_checkpoints(Embeddings, "restore")
        test_roc = evaluation.evaluate_ROC(self.X_test, Embeddings)
        print("Accuracy for alpha: {0:.2f} : {1:.9f}".format(self.alpha, test_roc))
        return Embeddings, test_roc

    def getEmbedding(self, type, nodes):
        if type == 'embed_layer':
            feed_dict = {self.train_data_id: nodes['node_id'], self.train_data_attr: nodes['node_attr'], self.keep_prob : 1}
            Embedding = self.sess.run(self.representation_layer, feed_dict=feed_dict)
            return Embedding
        if type == 'out_embedding':
            Embedding = self.sess.run(self.weights['out_embeddings'])
            return Embedding  # nodes_number * (id_dim + attr_dim)

    def embedding_checkpoints(self, Embeddings, type):
        file = self.path + "Embeddings.txt"
        if type == "save":
            if os.path.isfile(file):
                os.remove(file)
            pd.DataFrame(Embeddings).to_csv(file, index=False, header=False)
        if type == 'restore':
            Embeddings = pd.read_csv(file, header=None)
            return np.array(Embeddings)
