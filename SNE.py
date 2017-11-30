'''
Tensorflow implementation of Social Network Embedding framework (SNE)

@author: Lizi Liao (liaolizi.llz@gmail.com)

'''


import math
import random

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
import evaluation

class SNE(BaseEstimator, TransformerMixin):
    def __init__(self, data, id_embedding_size, attr_embedding_size,
                 batch_size=128, alpha = 1, n_neg_samples=10,
                epoch=100, random_seed = 2016):
        # bind params to class
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
        self.n_neg_samples = n_neg_samples
        self.epoch = epoch
        self.random_seed = random_seed
        self.hidden_size_1 = 200
        self.beta = 0.01
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
            # self.train_data_walks = tf.placeholder(tf.float32, shape=[None, self.random_walk_length])  # batch_size * random_walk_length
            self.train_labels = tf.placeholder(tf.int32, shape=[None, 1])  # batch_size * 1
            self.keep_prob = tf.placeholder(tf.float32)

            # Variables.
            network_weights = self._initialize_weights()
            self.weights = network_weights

            # Model.
            # Look up embeddings for node_id.
            self.id_embed =  tf.nn.embedding_lookup(self.weights['in_embeddings'], self.train_data_id) # batch_size * id_dim
            self.attr_layer_1 = tf.nn.relu(tf.add(tf.matmul(self.train_data_attr, self.weights['hidden_1']), self.weights['bias_1']))
            self.drop_out = tf.nn.dropout(self.attr_layer_1, self.keep_prob)
            self.attr_embed =  tf.nn.relu(tf.matmul(self.drop_out, self.weights['attr_embeddings']))  # batch_size * attr_dim
            self.embed_layer = tf.concat([self.id_embed, self.alpha * (self.attr_embed)], 1) # batch_size * (id_dim + attr_dim)

            ## can add hidden_layers component here!

            # Compute the loss, using a sample of the negative labels each time.
            self.loss =  tf.reduce_mean(tf.nn.sampled_softmax_loss(self.weights['out_embeddings'], self.weights['biases'],
                                                  self.train_labels,  self.embed_layer,  self.n_neg_samples, self.node_N))

            #Regularizer
            self.regularizer = tf.nn.l2_loss(self.weights['attr_embeddings']) + tf.nn.l2_loss(self.weights['hidden_1'])

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
        all_weights['attr_embeddings'] = tf.Variable(tf.random_uniform([self.hidden_size_1,self.attr_embedding_size], -1.0, 1.0)) # attr_M * attr_dim
        all_weights['out_embeddings'] = tf.Variable(tf.truncated_normal([self.node_N, self.id_embedding_size + self.attr_embedding_size],
                                    stddev=1.0 / math.sqrt(self.id_embedding_size + self.attr_embedding_size)))
        all_weights['biases'] = tf.Variable(tf.zeros([self.node_N]))

        all_weights['hidden_1'] = tf.Variable(tf.random_normal([self.attr_M, self.hidden_size_1]))
        all_weights['bias_1'] = tf.Variable(tf.zeros([self.hidden_size_1]))

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

        print('Using in + out embedding')
        val_roc = []
        for epoch in range( self.epoch ):
            total_batch = int( len(self.X_train['data_id_list']) / self.batch_size)
            # print('total_batch in 1 epoch: ', total_batch)
            # Loop over all batches
            for i in range(total_batch):
                # generate a batch data
                batch_xs = {}
                start_index = np.random.randint(0, len(self.X_train['data_id_list']) - self.batch_size)
                ind = random.sample(range(len(self.X_train['data_id_list']) ), self.batch_size)
                batch_xs['batch_data_id'] = self.X_train['data_id_list'][ind]
                # batch_xs['batch_random_walk'] = self.X_train['data_random_walks'][ind]
                batch_xs['batch_data_attr'] = self.X_train['data_attr_list'][ind]
                batch_xs['batch_data_label'] = self.X_train['data_label_list'][ind]
                # Fit training using batch data
                cost = self.partial_fit(batch_xs)

            # Display logs per epoch
            Embeddings_out = self.getEmbedding('out_embedding', self.nodes)
            Embeddings_in = self.getEmbedding('embed_layer', self.nodes)
            Embeddings = Embeddings_out + Embeddings_in

            # link prediction test
            roc = evaluation.evaluate_ROC(self.X_validation, Embeddings)
            print("Epoch:", '%04d' % (epoch + 1), \
                         "roc=", "{:.9f}".format(roc))
            val_roc.append(roc)
        return Embeddings, val_roc

    def getEmbedding(self, type, nodes):
        if type == 'embed_layer':
            feed_dict = {self.train_data_id: nodes['node_id'], self.train_data_attr: nodes['node_attr'], self.keep_prob : 1}
            Embedding = self.sess.run(self.embed_layer, feed_dict=feed_dict)
            return Embedding
        if type == 'out_embedding':
            Embedding = self.sess.run(self.weights['out_embeddings'])
            return Embedding  # nodes_number * (id_dim + attr_dim)

