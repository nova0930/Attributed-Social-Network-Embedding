'''
Tensorflow implementation of Social Network Embedding framework (SNE)

@author: Lizi Liao (liaolizi.llz@gmail.com)

'''

import pandas as pd
import math
import random
import os
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
import evaluation

class SNE(BaseEstimator, TransformerMixin):
    def __init__(self, path, data, id_embedding_size, attr_embedding_size, pretrained_weights,
                 batch_size=128, alpha = 1, beta = 0.01, n_neg_samples=10,
                epoch=100, random_seed = 2016):
        # bind params to class
        self.path = path
        self.node_neighbors_map = data.node_neighbors_map
        # self.random_walk_length = data.random_walk_length
        self.batch_size = batch_size
        self.node_N = data.id_N
        self.attr_M = data.attr_M
        self.X_train = data.X
        self.X_test = data.X_test
        self.X_validation = data.X_validation
        self.nodes = data.nodes
        self.id_embedding_size = id_embedding_size
        self.attr_embedding_size = attr_embedding_size
        self.alpha = alpha
        print("For threshold ", self.alpha)
        self.n_neg_samples = n_neg_samples
        self.epoch = epoch
        self.random_seed = random_seed
        self.hidden_size_1 = 128
        self.beta = beta
        self.pretrained_weights = pretrained_weights
        # init all variables in a tensorflow graph
        self._init_graph()


    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():#, tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.train_data_id = tf.placeholder(tf.int32, shape=[None])  # batch_size * 1
            self.train_data_attr = tf.placeholder(tf.float32, shape=[None, self.attr_M])  # batch_size * attr_M
            self.train_labels = tf.placeholder(tf.int32, shape=[None, 1])  # batch_size * 1
            self.keep_prob = tf.placeholder(tf.float32)

            # Variables.
            network_weights = self._initialize_weights()
            self.weights = network_weights

            # Model.
            # Look up embeddings for node_id.
            self.id_embed =  tf.nn.embedding_lookup(self.weights['in_embeddings'], self.train_data_id) # batch_size * id_dim
            # self.attr_layer_1 = tf.nn.relu(tf.add(tf.matmul(self.train_data_attr, self.weights['attr_hidden_1']), self.weights['attr_bias_1']))
            z2_attrib = tf.matmul(self.train_data_attr, self.weights['attr_hidden_1'])
            batch_mean, batch_var = tf.nn.moments(z2_attrib, [0])
            scale2 = tf.Variable(tf.ones([self.hidden_size_1]))
            beta2 = tf.Variable(tf.zeros([self.hidden_size_1]))
            self.batch_norm_layer = tf.nn.batch_normalization(z2_attrib,batch_mean,batch_var,beta2,scale2,1e-3)

            self.attr_layer_1 = tf.nn.relu(self.batch_norm_layer)

            self.attr_embed =  tf.nn.relu(tf.matmul(self.attr_layer_1, self.weights['attr_embeddings'])) # batch_size * attr_dim


            self.embed_layer = tf.concat([ self.id_embed, self.alpha * (self.attr_embed)], 1) # batch_size * (id_dim + attr_dim)

            batch_mean_1, batch_var_1 = tf.nn.moments(self.embed_layer, [0])
            scale3 = tf.Variable(tf.ones([ self.id_embedding_size + self.attr_embedding_size]))
            beta3 = tf.Variable(tf.zeros([ self.id_embedding_size + self.attr_embedding_size]))
            self.batch_norm_layer = tf.nn.batch_normalization(self.embed_layer, batch_mean_1, batch_var_1, beta3, scale3, 1e-3)
            ## can add hidden_layers component here!

            # Compute the loss, using a sample of the negative labels each time.
            self.loss =  tf.reduce_mean(tf.nn.sampled_softmax_loss(self.weights['out_embeddings'], self.weights['biases'],
                                                  self.train_labels,  self.embed_layer,  self.n_neg_samples, self.node_N))

            #Regularizer
            self.regularizer = tf.nn.l2_loss(self.weights['attr_embeddings']) + tf.nn.l2_loss(self.weights['attr_hidden_1']) +  tf.nn.l2_loss(self.weights['attr_bias_1'])  +tf.nn.l2_loss(self.weights['attr_bias_2'])

            self.loss = tf.reduce_mean(self.loss + self.beta * self.regularizer)

            # Optimizer.
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            # init
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['in_embeddings'] = tf.Variable(tf.random_uniform([self.node_N, self.id_embedding_size], -1.0, 1.0))  # id_N * id_dim
        all_weights['out_embeddings'] = tf.Variable(tf.truncated_normal([self.node_N, self.id_embedding_size + self.attr_embedding_size],
                                    stddev=1.0 / math.sqrt(self.id_embedding_size + self.attr_embedding_size)))
        all_weights['biases'] = tf.Variable(tf.zeros([self.node_N]))

        all_weights['attr_embeddings'] = tf.Variable(
            tf.truncated_normal([self.hidden_size_1, self.attr_embedding_size],
                                stddev=1.0 / math.sqrt(self.attr_embedding_size)))  # attr_M * attr_dim
        all_weights['attr_hidden_1'] = tf.Variable(tf.truncated_normal([self.attr_M, self.hidden_size_1],
                                                                       stddev=1.0 / math.sqrt(self.hidden_size_1)))
        all_weights['attr_bias_1'] = tf.Variable(tf.zeros([self.hidden_size_1]))
        all_weights['attr_bias_2'] = tf.Variable(tf.zeros([self.attr_embedding_size]))
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
            total_batch = int( len(self.X_train['data_id_list']) / self.batch_size)
            # Loop over all batches
            total_iterations += 1
            avg_cost = 0.
            for i in range(total_batch):
                # generate a batch data
                batch_xs = {}
                start_index = np.random.randint(0, len(self.X_train['data_id_list']) - self.batch_size)
                batch_xs['batch_data_id'] = self.X_train['data_id_list'][start_index:(start_index + self.batch_size)]
                batch_xs['batch_data_attr'] = self.X_train['data_attr_list'][
                                              start_index:(start_index + self.batch_size)]
                batch_xs['batch_data_label'] = self.X_train['data_label_list'][
                                               start_index:(start_index + self.batch_size)]

                # Fit training using batch data
                cost = self.partial_fit(batch_xs)
                avg_cost += cost / total_batch
            # Display logs per epoch
            Embeddings_out = self.getEmbedding('out_embedding', self.nodes)
            Embeddings_in = self.getEmbedding('embed_layer', self.nodes)
            Embeddings = Embeddings_out + Embeddings_in

            # link prediction test
            roc = evaluation.evaluate_ROC(self.X_validation, Embeddings)
            # print("Epoch:", '%04d' % (epoch + 1), \
            #              "roc=", "{:.9f}".format(roc))

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
        return Embeddings

    def getEmbedding(self, type, nodes):
        if type == 'embed_layer':
            feed_dict = {self.train_data_id: nodes['node_id'], self.train_data_attr: nodes['node_attr'], self.keep_prob : 1}
            Embedding = self.sess.run(self.embed_layer, feed_dict=feed_dict)
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
